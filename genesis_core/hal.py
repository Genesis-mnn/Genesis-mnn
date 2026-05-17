# genesis/hal.py — Genesis 框架硬件抽象层
#
# 本模块定义了统一的硬件后端接口，并原生支持 NVIDIA CUDA 与华为昇腾 NPU。
# 同时为第三方硬件厂商提供了自行适配的开放接口与完整指南。
#
# 设计原则：
# 1. 所有后端必须继承 GenesisBackend 并实现契约。
# 2. 导入本文件不会因缺失专用驱动而崩溃（尤其针对 Ascend 后端）。
# 3. 第三方可通过注册机制动态添加后端，并被框架自动识别。
# 4. 提供全局活跃后端切换及设备迁移辅助。

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Type

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
_THIRD_PARTY_BACKENDS: Dict[str, Type['GenesisBackend']] = {}
_ACTIVE_BACKEND: Optional['GenesisBackend'] = None


# ============================================================================
# 1. GenesisBackend 基类
# ============================================================================
class GenesisBackend(nn.Module, ABC):
    """所有硬件后端必须遵循的统一接口契约。

    派生类必须实现所有抽象方法，并提供一致的设备管理能力。
    本基类同时提供了通用的 to_device 迁移方法。
    """

    def __init__(self, device_id: int = 0):
        super().__init__()
        self._device_id = device_id
        self._torch_device_name = 'cpu'  # 子类必须在 __init__ 中重写为实际 PyTorch 设备名

    # ------------------------------------------------------------------
    # 抽象方法（契约）
    # ------------------------------------------------------------------
    @abstractmethod
    def synchronize(self) -> None:
        """同步当前设备上的所有未决操作，阻塞直到完成。"""
        ...

    @abstractmethod
    def device_count(self) -> int:
        """返回此类型硬件的可用设备数量。"""
        ...

    @abstractmethod
    def set_device(self, device_id: int) -> None:
        """将指定的设备设为当前活跃设备。"""
        ...

    @abstractmethod
    def get_device_name(self) -> str:
        """返回用户可见的后端标识名（如 'cuda', 'ascend', 'cpu'）。"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查该后端在运行时是否可用（驱动存在且硬件就绪）。"""
        ...

    # ------------------------------------------------------------------
    # 通用工具方法
    # ------------------------------------------------------------------
    def to_device(self,
                  obj: Any,
                  device_id: Optional[int] = None) -> Any:
        """将模型或张量安全地迁移到本后端设备。

        参数:
            obj: nn.Module 或 torch.Tensor 实例。
            device_id: 目标设备索引，若为 None 则使用实例默认设备。
        返回:
            迁移后的对象（原地操作并返回自身）。
        异常:
            RuntimeError: 当本后端不可用时抛出。
            TypeError: 传入不支持的对象类型。
        """
        if not self.is_available():
            raise RuntimeError(
                f"{self.get_device_name()} 后端当前不可用。"
            )
        dev_id = device_id if device_id is not None else self._device_id
        device = torch.device(f"{self._torch_device_name}:{dev_id}")

        if isinstance(obj, nn.Module):
            return obj.to(device)
        elif isinstance(obj, torch.Tensor):
            return obj.to(device)
        else:
            raise TypeError(
                f"不支持的对象类型: {type(obj)}，仅支持 nn.Module 或 Tensor。"
            )


# ============================================================================
# 2. CPU 后端（总是可用）
# ============================================================================
class CPUBackend(GenesisBackend):
    """CPU 后端实现，始终可用，作为默认回退。"""

    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._torch_device_name = 'cpu'

    def synchronize(self) -> None:
        # CPU 无需同步
        pass

    def device_count(self) -> int:
        return 1

    def set_device(self, device_id: int) -> None:
        self._device_id = device_id

    def get_device_name(self) -> str:
        return 'cpu'

    def is_available(self) -> bool:
        return True


# ============================================================================
# 3. CUDA 后端（NVIDIA GPU）
# ============================================================================
class CUDABackend(GenesisBackend):
    """CUDA 后端完整实现，基于 PyTorch 原生 CUDA API。

    在运行时自动检测 CUDA 环境，若不可用则 is_available() 返回 False。
    """

    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._torch_device_name = 'cuda'

    def synchronize(self) -> None:
        if self.is_available():
            torch.cuda.synchronize()

    def device_count(self) -> int:
        if self.is_available():
            return torch.cuda.device_count()
        return 0

    def set_device(self, device_id: int) -> None:
        if self.is_available():
            torch.cuda.set_device(device_id)
            self._device_id = device_id

    def get_device_name(self) -> str:
        return 'cuda'

    def is_available(self) -> bool:
        return torch.cuda.is_available()


# ============================================================================
# 4. 昇腾 NPU 后端（华为 Ascend）
# ============================================================================
class AscendBackend(GenesisBackend):
    """华为昇腾 NPU 后端实现。

    通过尝试导入 torch_npu 包并利用 PyTorch 的 PrivateUse1 机制进行安全注册。
    在缺少驱动或未安装硬件环境的系统中，导入本模块不会崩溃，
    is_available() 将安全地返回 False 并给出明确提示。
    """

    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self._available = False
        self._torch_npu = None
        self._torch_device_name = 'cpu'  # 默认回退

        # ---- 安全尝试导入 torch_npu ----
        try:
            import torch_npu
            self._torch_npu = torch_npu
            self._available = True
            self._torch_device_name = 'npu'  # torch_npu 注册的设备名
        except ImportError:
            # 尝试利用 PrivateUse1 机制注册 ascend 标识（仅用于命名兼容）
            try:
                torch.utils.rename_privateuse1_backend('ascend')
                self._torch_device_name = 'ascend'  # PrivateUse1 重命名后的设备名
            except Exception:
                # 安静失败，保持 _available = False
                pass

    def synchronize(self) -> None:
        if self.is_available() and self._torch_npu is not None:
            self._torch_npu.synchronize()

    def device_count(self) -> int:
        if self.is_available() and self._torch_npu is not None:
            return self._torch_npu.device_count()
        return 0

    def set_device(self, device_id: int) -> None:
        if self.is_available() and self._torch_npu is not None:
            self._torch_npu.set_device(device_id)
            self._device_id = device_id

    def get_device_name(self) -> str:
        # 对外统一标识为 'ascend'
        return 'ascend'

    def is_available(self) -> bool:
        return self._available


# ============================================================================
# 5. 第三方硬件厂商自行适配接口
# ============================================================================
class ThirdPartyBackend(GenesisBackend):
    """第三方硬件后端必须继承的接口契约。

    任何希望接入 Genesis 生态的厂商都应遵循此接口，实现所有抽象方法。
    完成后通过 register_third_party_backend 动态注册，即可被框架识别。
    """

    @abstractmethod
    def is_available(self) -> bool:
        # 虽然基类已定义，但在此显式声明强调其重要性
        ...


def register_third_party_backend(name: str, backend_class: Type[GenesisBackend]) -> None:
    """注册一个第三方硬件后端。

    参数:
        name: 后端标识名（如 'moorethreads', 'cambricon'），需全局唯一。
        backend_class: 继承自 ThirdPartyBackend 的具体后端类（非实例）。

    异常:
        TypeError: 若提供的类不是 ThirdPartyBackend 的子类。
        ValueError: 若名称已被内置或已注册。
    """
    if not issubclass(backend_class, ThirdPartyBackend):
        raise TypeError(
            f"{backend_class.__name__} 必须继承 ThirdPartyBackend。"
        )

    # 避免与内置后端名冲突
    reserved = {'cpu', 'cuda', 'ascend'}
    if name in reserved or name in _THIRD_PARTY_BACKENDS:
        raise ValueError(
            f"后端名称 '{name}' 已被保留或已注册，请使用其他名称。"
        )

    _THIRD_PARTY_BACKENDS[name] = backend_class
    print(f"第三方后端 '{name}' ({backend_class.__name__}) 已注册，将在下次扫描时识别。")


# ============================================================================
# 6. 硬件自动感知与调度中心
# ============================================================================
def get_available_backends() -> Dict[str, GenesisBackend]:
    """自动扫描当前环境，返回所有可用后端实例的字典。

    至少包含 'cpu'，以及可用的 'cuda'、'ascend' 和所有已注册的第三方后端。
    返回:
        {名称: 后端实例} 字典。
    """
    backends: Dict[str, GenesisBackend] = {}

    # CPU 总是可用
    cpu = CPUBackend()
    backends[cpu.get_device_name()] = cpu

    # CUDA
    cuda = CUDABackend()
    if cuda.is_available():
        backends[cuda.get_device_name()] = cuda

    # Ascend
    ascend = AscendBackend()
    if ascend.is_available():
        backends[ascend.get_device_name()] = ascend

    # 已注册的第三方后端
    for name, cls in _THIRD_PARTY_BACKENDS.items():
        if name in backends:
            continue
        try:
            instance = cls()
            if instance.is_available():
                backends[name] = instance
        except Exception as e:
            print(f"第三方后端 '{name}' 实例化失败，已跳过: {e}")

    return backends


def set_active_backend(name: str) -> None:
    """手动切换当前活跃的硬件后端。

    切换完成后，用户可通过 get_active_backend() 获取新后端，
    并使用其 to_device() 方法将模型/张量迁移到新设备。

    参数:
        name: 目标后端名称（如 'cuda', 'ascend', 'cpu'）。

    异常:
        ValueError: 若指定后端不可用或未注册。
    """
    backends = get_available_backends()
    if name not in backends:
        raise ValueError(
            f"后端 '{name}' 不可用或未注册。当前可用: {list(backends.keys())}"
        )
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backends[name]
    print(f"[Genesis HAL] 活跃后端已切换至: {name}")


def get_active_backend() -> GenesisBackend:
    """获取当前活跃的后端实例。

    若未通过 set_active_backend 设置，默认返回 CPU 后端。
    """
    global _ACTIVE_BACKEND
    if _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = CPUBackend()
    return _ACTIVE_BACKEND


# ============================================================================
# 第三方硬件适配指南
# ============================================================================
"""
============================================================
Genesis 框架第三方硬件适配指南
============================================================

1. 设计理念
-----------
Genesis 的硬件抽象层（HAL）为“write once, run anywhere”提供基础。
所有计算后端（GPU、NPU、FPGA 等）都通过统一的 GenesisBackend 接口暴露能力。
框架内部的神经元、突触、胶质细胞等组件不直接依赖特定硬件 API，
而是通过当前活跃后端的 to_device() 方法迁移模型和张量，
并通过后端提供的同步原语保证时序。

2. 适配步骤
-----------
a) 编写后端类：继承 ThirdPartyBackend，实现所有抽象方法。
   - __init__ 中设置 self._torch_device_name 为实际 PyTorch 设备名
     （如 'privateuse1'，或自行注册的新名称）。
   - is_available() 必须安全返回硬件是否可用，不应抛出异常。
   - device_count() 返回可用设备个数。
   - synchronize() 调用对应驱动的同步 API。
   - set_device() 切换当前设备。
   - get_device_name() 返回后端标识名（英文小写，如 'example'）。

b) 注册后端：调用 register_third_party_backend('example', ExampleNPUBackend)
   （建议在框架初始化时调用，或由厂商提供自动注册入口）。

c) 使用后端：
   - 通过 get_available_backends() 查看所有可用后端。
   - set_active_backend('example') 切换活跃后端。
   - 使用 get_active_backend().to_device(model) 迁移模型。
   - 模型内部会自动使用新设备执行运算。

3. 最小化适配示例（ExampleNPU）
--------------------------------
假设某厂商提供了 example_npu 驱动包，其中暴露了以下接口：
   - example_npu.is_available() -> bool
   - example_npu.synchronize()
   - example_npu.device_count() -> int
   - example_npu.set_device(device_id)
且驱动已经通过 PyTorch 的 PrivateUse1 机制注册了设备名 'example'。

则适配代码如下：

    import torch
    from genesis.hal import ThirdPartyBackend, register_third_party_backend

    class ExampleNPUBackend(ThirdPartyBackend):
        def __init__(self, device_id=0):
            super().__init__(device_id)
            self._torch_device_name = 'example'  # 驱动注册的设备名
            self._available = False
            try:
                import example_npu
                self._driver = example_npu
                self._available = example_npu.is_available()
            except ImportError:
                self._driver = None
                self._available = False

        def get_device_name(self):
            return 'example'

        def is_available(self):
            return self._available

        def device_count(self):
            if self._available and self._driver is not None:
                return self._driver.device_count()
            return 0

        def set_device(self, device_id):
            if self._available and self._driver is not None:
                self._driver.set_device(device_id)
                self._device_id = device_id

        def synchronize(self):
            if self._available and self._driver is not None:
                self._driver.synchronize()

    # 注册到 Genesis
    register_third_party_backend('example', ExampleNPUBackend)

    # 使用
    backends = get_available_backends()
    if 'example' in backends:
        set_active_backend('example')
        model = MyGenesisModel()
        backend = get_active_backend()
        backend.to_device(model)
        ... # 正常运行

4. 注意事项
-----------
- 后端类的 __init__ 不应在导入时执行重量级操作（如分配大块显存），
  以防在无硬件环境中造成不必要的资源占用或异常。
- 若驱动未安装，后端类的实例化不应崩溃，is_available() 必须安全返回 False。
- 设备名称应避免与已有内置名称（cpu、cuda、ascend）冲突。
- 对于不提供原生 PyTorch 后端的设备，可使用 PyTorch 的 PrivateUse1 机制
  注册自定义设备（参见 PyTorch 文档），并在后端类中设置对应的 _torch_device_name。
"""