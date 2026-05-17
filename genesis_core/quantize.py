# quantize.py — Genesis 框架量化与模型压缩工具
# 负责将全精度模型压缩为低比特表示，适配边缘设备和神经拟态芯片部署。
# 遵循白皮书第九章及核心公理，提供 PTQ、QAT、阈值中心化量化等完整管线。

import copy
import io
import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Union, Any
from dataclasses import dataclass, field

# 导入核心基类
from .genesis_core import MorphoNeuron, ResonanceSynapse

# 定义精度映射常量（原 _PRECISION_MAP 内容，现公开）
PRECISION_MAP: Dict[str, torch.dtype] = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


# ============================================================================
# 1. PrecisionConfig（混合精度配置类）
# ============================================================================

@dataclass
class PrecisionConfig:
    """混合精度配置。

    允许按变量名（u, h, r, strength 等）指定计算精度，并与 `PRECISION_MAP` 对齐。
    支持 grad_scaler 梯度缩放配置。

    参数:
        var_dtype_map: 字典，键为变量名（如 'u', 'h', 'r', 'strength'），
                       值为精度字符串（'fp32', 'fp16', 'bf16'）。
                       默认所有变量使用 fp32。
        grad_scaler: 是否启用 GradScaler 动态缩放损失，避免梯度下溢。
        scaler_init_scale: GradScaler 的初始缩放因子。
    """
    var_dtype_map: Dict[str, str] = field(default_factory=lambda: {
        'u': 'fp32',
        'h': 'fp32',
        'r': 'fp32',
        'strength': 'fp32',
        'spine_ca': 'fp32',
        'spine_rho': 'fp32',
        'spine_g': 'fp32',
    })
    grad_scaler: bool = True
    scaler_init_scale: float = 2.0 ** 16

    def get_dtype(self, var_name: str) -> torch.dtype:
        """返回变量名对应的 PyTorch dtype。"""
        prec_str = self.var_dtype_map.get(var_name, 'fp32')
        if prec_str not in PRECISION_MAP:
            raise ValueError(f"不支持的精度字符串: {prec_str}。可选值: {list(PRECISION_MAP.keys())}")
        return PRECISION_MAP[prec_str]

    def set_default(self, dtype_str: str) -> None:
        """将所有变量的精度设置为同一个值。"""
        for key in self.var_dtype_map:
            self.var_dtype_map[key] = dtype_str


# ============================================================================
# 2. QuantConfig（量化配置类）
# ============================================================================

@dataclass
class QuantConfig:
    """量化配置。

    支持按变量类型分别指定量化位宽（int8/int4/fp16等），以及对称/非对称量化模式。
    包含阈值中心化量化和权重重整的开关。

    参数:
        default_weight_dtype: 权重默认量化位宽，如 'int8'。
        default_activation_dtype: 激活（状态变量）默认量化位宽。
        symmetric: 是否使用对称量化（默认 True）。
        per_channel: 是否按通道量化（对权重组）。
        weight_rescale: 是否在量化前对权重进行重整缩放。
        threshold_centered: 是否对膜电位使用阈值中心化量化。
        var_dtype_overrides: 按变量名覆盖的量化位宽，例如 {'u': 'int4', 'h': 'int8'}。
        observer_momentum: 观察者（observer）更新统计时的 EMA 动量。
        qconfig_dict: 额外的自定义量化配置字典。
    """
    default_weight_dtype: str = 'int8'
    default_activation_dtype: str = 'int8'
    symmetric: bool = True
    per_channel: bool = False
    weight_rescale: bool = True
    threshold_centered: bool = True
    var_dtype_overrides: Dict[str, str] = field(default_factory=dict)
    observer_momentum: float = 0.1
    qconfig_dict: Dict[str, Any] = field(default_factory=dict)

    def get_var_dtype(self, var_name: str) -> str:
        """返回指定变量的量化位宽字符串。"""
        return self.var_dtype_overrides.get(var_name, self.default_activation_dtype)


# ============================================================================
# 3. 通用量化工具函数
# ============================================================================

def _get_qmin_qmax(dtype_str: str, symmetric: bool = True) -> Tuple[int, int]:
    """根据量化位宽字符串返回量化范围。

    参数:
        dtype_str: 量化类型，如 'int8', 'int4', 'fp16'。
        symmetric: 是否对称量化（对 int 类型有效）。
    返回:
        (qmin, qmax): 量化最小/最大值。
    异常:
        ValueError: 不支持的 dtype_str。
    """
    if dtype_str == 'fp16':
        return -1, -1  # 无需量化范围，使用半精度浮点
    elif dtype_str == 'int8':
        if symmetric:
            return -127, 127
        else:
            return 0, 255
    elif dtype_str == 'int4':
        if symmetric:
            return -8, 7
        else:
            return 0, 15
    else:
        raise ValueError(f"不支持的量化位宽: {dtype_str}")


def compute_quant_params(
        min_val: Union[float, torch.Tensor],
        max_val: Union[float, torch.Tensor],
        qmin: int,
        qmax: int,
        symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算量化的 scale 和 zero_point。

    参数:
        min_val: 数值范围最小值（标量或张量）。
        max_val: 数值范围最大值（标量或张量）。
        qmin: 量化域最小值。
        qmax: 量化域最大值。
        symmetric: 是否对称量化。
    返回:
        scale (Tensor), zero_point (Tensor)。
    """
    # 确保是 tensor
    if not isinstance(min_val, torch.Tensor):
        min_val = torch.tensor(min_val, dtype=torch.float32)
    if not isinstance(max_val, torch.Tensor):
        max_val = torch.tensor(max_val, dtype=torch.float32)

    if symmetric:
        # 对称量化：以0为中心
        abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
        # 避免除以零
        scale = (2 * abs_max) / (qmax - qmin)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero_point = torch.zeros_like(scale, dtype=torch.float32)
    else:
        # 非对称量化
        range_val = max_val - min_val
        scale = range_val / (qmax - qmin)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        zero_point = qmin - min_val / scale
        zero_point = torch.round(zero_point)
        zero_point = torch.clamp(zero_point, qmin, qmax)
    return scale, zero_point


def quantize_tensor(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        qmin: int,
        qmax: int
) -> torch.Tensor:
    """将浮点张量量化为整数表示。

    参数:
        x: 输入浮点张量。
        scale: 量化步长。
        zero_point: 零点偏移。
        qmin: 量化下界。
        qmax: 量化上界。
    返回:
        量化后的整数张量（类型为浮点以便梯度，但值为整数）。
    """
    x_q = torch.round(x / scale + zero_point)
    x_q = torch.clamp(x_q, qmin, qmax)
    return x_q


def dequantize_tensor(
        x_q: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
) -> torch.Tensor:
    """将量化整数张量反量化为浮点近似。

    参数:
        x_q: 量化后的整数张量。
        scale: 量化步长。
        zero_point: 零点偏移。
    返回:
        反量化后的浮点张量。
    """
    return (x_q - zero_point) * scale


def fake_quantize(
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        qmin: int,
        qmax: int
) -> torch.Tensor:
    """伪量化函数（直通估计器 STE）。

    前向传播应用量化-反量化，反向传播将梯度直通传递，模拟量化噪声并使网络可训练。

    参数:
        x: 输入浮点张量。
        scale: 量化步长。
        zero_point: 零点偏移。
        qmin: 量化下界。
        qmax: 量化上界。
    返回:
        伪量化后的浮点张量，值域被离散化。
    """
    x_q = quantize_tensor(x, scale, zero_point, qmin, qmax)
    x_dq = dequantize_tensor(x_q, scale, zero_point)
    # STE: 梯度通过，不经过 clamp/round
    return x + (x_dq - x).detach()


def weight_rescale(
        weight: torch.Tensor,
        method: str = 'max',
        dim: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """权重重整策略：缩放权重以充分占用量化比特。

    参数:
        weight: 待量化的权重张量。
        method: 缩放方法，目前支持 'max'（按指定维度最大绝对值归一化）。
        dim: 沿哪个维度缩放，None 表示全局缩放。
    返回:
        (rescaled_weight, scale_factor): 重整后的权重和缩放因子，
        使得 ``weight ≈ rescaled_weight * scale_factor``。
    """
    if method == 'max':
        if dim is not None:
            # 按通道缩放
            max_vals, _ = weight.abs().max(dim=dim, keepdim=True)
        else:
            max_vals = weight.abs().max()
        scale_factor = max_vals
        eps = 1e-8
        rescaled_weight = weight / (scale_factor + eps)
        return rescaled_weight, scale_factor
    else:
        raise ValueError(f"未知的权重重整方法: {method}")


# ============================================================================
# 4. ThresholdCenteredQuantize（阈值中心化量化模块）
# ============================================================================

class ThresholdCenteredQuantize(nn.Module):
    """针对膜电位 u 的阈值中心化量化模块。

    依据白皮书 9.2 节，在输出阈值附近以指数方式加密量化格点，远离阈值处放宽精度。
    通过 sigmoid 映射将膜电位值域压缩到 [0,1] 并均匀量化，再逆映射恢复。

    参数:
        Vth: 输出阈值。可以是标量（所有神经元共用）或形状为 (num_neurons,) 的张量（逐神经元独立阈值）。默认 1.0。
        k: sigmoid 陡度系数，控制阈值附近的加密程度（默认 5.0）。
        qmin: 均匀量化域下限（默认 -127）。
        qmax: 均匀量化域上限（默认 127）。
        symmetric: 均匀量化是否对称（默认 True，但映射后值域[0,1]通常非对称用更好，此处可按需设置）。
        observer_momentum: observer 更新动量（默认 0.1）。
        learn_alpha: 是否学习陡度系数 k（默认 True）。
        fixed_range: 是否使用固定的映射值域 [0,1] 计算量化参数，跳过动态 observer 更新。默认 False。
    """

    def __init__(
            self,
            Vth: Union[float, torch.Tensor] = 1.0,
            k: float = 5.0,
            qmin: int = -127,
            qmax: int = 127,
            symmetric: bool = False,
            observer_momentum: float = 0.1,
            learn_alpha: bool = False,
            fixed_range: bool = False
    ):
        super().__init__()
        if isinstance(Vth, torch.Tensor):
            self.register_buffer('Vth', Vth)
        else:
            self.Vth = Vth  # 标量 float
        self.k = nn.Parameter(torch.tensor(k), requires_grad=learn_alpha)
        self.qmin = qmin
        self.qmax = qmax
        self.symmetric = symmetric
        self.momentum = observer_momentum
        self.fixed_range = fixed_range

        # 均匀量化器 observer（针对映射后的 y ∈ [0,1]）
        self.register_buffer('y_min', torch.tensor(0.0))
        self.register_buffer('y_max', torch.tensor(1.0))
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0, dtype=torch.float32))
        self.fake_quant_enabled: bool = True
        self._observer_enabled: bool = True

        if self.fixed_range:
            self._compute_y_params()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """对膜电位张量 u 执行阈值中心化伪量化。

        参数:
            u: 膜电位张量，任意形状，值应大致在 [0, 2*Vth] 范围。
        返回:
            伪量化后的膜电位张量，形状同 u。
        """
        if not self.fake_quant_enabled:
            return u

        # 准备可广播的 Vth
        vth = self.Vth
        if isinstance(vth, torch.Tensor):
            if vth.dim() > 0:
                # 重塑 vth 以匹配 u 的最后一维，前置维度为 1 以实现广播
                vth = vth.view( *([1] * (u.dim() - 1)), -1 )
        # 若 vth 是 float 或 0 维张量，无需额外处理

        # 1. sigmoid 映射：将 u 压缩到 [0,1]，Vth 附近导数最大
        y = torch.sigmoid(self.k * (u - vth))

        if self.training and self._observer_enabled:
            if self.fixed_range:
                with torch.no_grad():
                    self.y_min = torch.tensor(0.0)
                    self.y_max = torch.tensor(1.0)
                    self._compute_y_params()
            else:
                # 更新 observer 统计（用于计算 scale/zero_point）
                with torch.no_grad():
                    # 使用 EMA 更新 y 的最小/最大值
                    batch_y_min = y.min()
                    batch_y_max = y.max()
                    self.y_min = (1 - self.momentum) * self.y_min + self.momentum * batch_y_min
                    self.y_max = (1 - self.momentum) * self.y_max + self.momentum * batch_y_max
                    self._compute_y_params()

        # 2. 对 y 进行均匀伪量化
        y_q = fake_quantize(y, self.scale, self.zero_point, self.qmin, self.qmax)

        # 3. 逆 sigmoid 恢复膜电位值
        # 限制 y_q 避免 logit 数值溢出
        y_clamped = torch.clamp(y_q, 0.01, 0.99)
        u_inv = vth + torch.log(y_clamped / (1 - y_clamped)) / self.k

        return u_inv

    def _compute_y_params(self) -> None:
        """根据 observer 的 y_min/y_max 计算 scale 和 zero_point。"""
        # 确保 y_max > y_min
        if self.y_max <= self.y_min:
            self.scale = torch.tensor(1.0)
            self.zero_point = torch.tensor(0.0, dtype=torch.float32)
            return
        # 非对称量化 scale/zero_point（因为 y 范围 [0,1] 非对称）
        self.scale, self.zero_point = compute_quant_params(
            self.y_min, self.y_max, self.qmin, self.qmax, symmetric=self.symmetric
        )

    def set_quant_params(self, scale: float, zero_point: float) -> None:
        """手动设置 scale 和 zero_point，并冻结 observer 更新。"""
        self.scale = torch.tensor(scale)
        self.zero_point = torch.tensor(zero_point, dtype=torch.float32)
        self.fake_quant_enabled = True  # 保持量化

    def disable_observer(self) -> None:
        """禁用 observer 更新，固定当前 scale/zero_point。"""
        self._observer_enabled = False


# ============================================================================
# 5. 通用伪量化模块 FakeQuantize
# ============================================================================

class FakeQuantize(nn.Module):
    """标准伪量化模块（支持对称/非对称，per-tensor/per-channel）。

    包含 observer 自动更新 scale/zero_point，前向应用 STE。

    参数:
        qmin: 量化域下限。
        qmax: 量化域上限。
        symmetric: 是否对称量化。
        per_channel: 是否按通道量化（对权重组生效）。
        channel_dim: 通道维度的轴（当 per_channel=True 时使用）。
        observer_momentum: observer EMA 动量。
    """

    def __init__(
            self,
            qmin: int,
            qmax: int,
            symmetric: bool = True,
            per_channel: bool = False,
            channel_dim: int = 0,
            observer_momentum: float = 0.1
    ):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_dim = channel_dim
        self.momentum = observer_momentum

        # observer 统计数据
        self.register_buffer('min_val', torch.tensor(0.0))
        self.register_buffer('max_val', torch.tensor(0.0))
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0, dtype=torch.float32))
        self.fake_quant_enabled: bool = True
        self._observer_enabled: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入 x 进行伪量化。

        参数:
            x: 浮点输入张量。
        返回:
            伪量化后的张量。
        """
        if not self.fake_quant_enabled:
            return x

        if self.training and self._observer_enabled:
            self._update_observer(x)

        x_q = fake_quantize(x, self.scale, self.zero_point, self.qmin, self.qmax)
        return x_q

    def _update_observer(self, x: torch.Tensor) -> None:
        """使用 EMA 更新 observer 的 min/max 统计。"""
        with torch.no_grad():
            if self.per_channel:
                # 按通道统计（假设除 channel_dim 外的维度归约）
                dims = [d for d in range(x.ndim) if d != self.channel_dim]
                if dims:
                    batch_min_val = x.amin(dim=dims)
                    batch_max_val = x.amax(dim=dims)
                else:
                    batch_min_val = x
                    batch_max_val = x
                # 初始化 observer 形状
                if self.min_val.numel() == 1 and self.min_val == 0.0:
                    self.min_val = torch.zeros_like(batch_min_val)
                    self.max_val = torch.zeros_like(batch_max_val)
                self.min_val = (1 - self.momentum) * self.min_val + self.momentum * batch_min_val
                self.max_val = (1 - self.momentum) * self.max_val + self.momentum * batch_max_val
            else:
                batch_min_val = x.min()
                batch_max_val = x.max()
                self.min_val = (1 - self.momentum) * self.min_val + self.momentum * batch_min_val
                self.max_val = (1 - self.momentum) * self.max_val + self.momentum * batch_max_val
            self._compute_params()

    def _compute_params(self) -> None:
        """根据 observer 的 min/max 重新计算 scale 和 zero_point。"""
        self.scale, self.zero_point = compute_quant_params(
            self.min_val, self.max_val, self.qmin, self.qmax, self.symmetric
        )

    def set_quant_params(self, scale: Union[float, torch.Tensor], zero_point: Union[float, torch.Tensor]) -> None:
        """手动设置 scale 和 zero_point（如 PTQ 后固定）。"""
        self.scale = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale)
        self.zero_point = zero_point if isinstance(zero_point, torch.Tensor) else torch.tensor(zero_point,
                                                                                               dtype=torch.float32)

    def disable_observer(self) -> None:
        """冻结 observer，不再更新 scale/zero_point。"""
        self._observer_enabled = False


# ============================================================================
# 6. 量化版本的形态神经元与共鸣突触
# ============================================================================

class QuantMorphoNeuron(MorphoNeuron):
    """支持量化的形态神经元。

    继承 MorphoNeuron 并在前向过程中对关键内部变量 (u, h, r 等) 进行伪量化。
    量化策略由 QuantConfig 控制，膜电位 u 可使用阈值中心化量化。

    参数:
        其余参数与 MorphoNeuron 一致，额外接收 quant_config。
    """

    def __init__(self, quant_config: QuantConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_config = quant_config
        self._build_quantizers()

    def _build_quantizers(self) -> None:
        """根据 QuantConfig 构建内部量化器。"""
        # 膜电位 u: 根据配置选择普通或阈值中心化量化器
        if self.quant_config.threshold_centered:
            # 使用阈值中心化量化器，u 的位宽由 config 决定
            u_dtype = self.quant_config.get_var_dtype('u')
            qmin_u, qmax_u = _get_qmin_qmax(u_dtype, symmetric=False)  # 映射后非对称
            self.quantizer_u = ThresholdCenteredQuantize(
                Vth=self.threshold.data.clone(),
                k=5.0,
                qmin=qmin_u,
                qmax=qmax_u,
                symmetric=False,
                learn_alpha=False
            )
        else:
            u_dtype = self.quant_config.get_var_dtype('u')
            qmin_u, qmax_u = _get_qmin_qmax(u_dtype, self.quant_config.symmetric)
            self.quantizer_u = FakeQuantize(
                qmin_u, qmax_u, symmetric=self.quant_config.symmetric,
                per_channel=False
            )

        # 健康度 h (范围 [0,1])
        h_dtype = self.quant_config.get_var_dtype('h')
        qmin_h, qmax_h = _get_qmin_qmax(h_dtype, self.quant_config.symmetric)
        self.quantizer_h = FakeQuantize(
            qmin_h, qmax_h, symmetric=self.quant_config.symmetric, per_channel=False
        )

        # 谐振敏感度 r (范围可正可负)
        r_dtype = self.quant_config.get_var_dtype('r')
        qmin_r, qmax_r = _get_qmin_qmax(r_dtype, self.quant_config.symmetric)
        self.quantizer_r = FakeQuantize(
            qmin_r, qmax_r, symmetric=self.quant_config.symmetric, per_channel=False
        )

        # 树突棘状态等其他变量可选择性量化，此处按需扩展
        # 将量化器注册为子模块
        self._quant_modules = nn.ModuleDict({
            'u': self.quantizer_u,
            'h': self.quantizer_h,
            'r': self.quantizer_r,
        })

    def forward(self,
                x: Optional[torch.Tensor] = None,
                modulator_concentrations: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """执行一个时间步的量化前向。

        在位宽量化时，伪量化被插入到状态读取和写入之间，模拟低精度计算。
        """
        # 量化当前状态（开始前）
        self._u = self.quantizer_u(self._u)
        self._h = self.quantizer_h(self._h)
        self._r = self.quantizer_r(self._r)

        # 调用父类完整逻辑（父类 forward 会读取这些量化后的状态）
        out = super().forward(x, modulator_concentrations)

        # 量化更新后的状态（结束前），确保下个时间步使用量化值
        self._u = self.quantizer_u(self._u)
        self._h = self.quantizer_h(self._h)
        self._r = self.quantizer_r(self._r)

        return out


class QuantResonanceSynapse(ResonanceSynapse):
    """支持量化的共鸣突触。

    对耦合强度 strength 等关键变量进行量化，并可根据配置进行权重重整。

    参数:
        其余参数与 ResonanceSynapse 一致，额外接收 quant_config。
    """

    def __init__(self, quant_config: QuantConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_config = quant_config
        self._build_quantizers()

    def _build_quantizers(self) -> None:
        """构建突触强度量化器，并可能应用权重重整。"""
        s_dtype = self.quant_config.get_var_dtype('strength')
        qmin_s, qmax_s = _get_qmin_qmax(s_dtype, self.quant_config.symmetric)
        self.quantizer_strength = FakeQuantize(
            qmin_s, qmax_s, symmetric=self.quant_config.symmetric,
            per_channel=self.quant_config.per_channel
        )

        # 权重重整因子（仅用于强度 static 备份，动态时在 forward 中处理）
        self._rescale_factor: Optional[torch.Tensor] = None

        # 将其注册为子模块
        self._quant_modules = nn.ModuleDict({
            'strength': self.quantizer_strength,
        })

    def forward(self,
                pre_spike: torch.Tensor,
                post_spike: torch.Tensor,
                pre_r: torch.Tensor,
                post_r: torch.Tensor,
                modulator_concentrations: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """执行一个时间步的量化突触计算。

        耦合强度 strength 被伪量化，以模拟低精度存储与计算。
        """
        # 量化 strength 当前值
        self.strength = self.quantizer_strength(self.strength)

        # 调用父类逻辑
        out = super().forward(pre_spike, post_spike, pre_r, post_r, modulator_concentrations)

        # 再次量化更新后的 strength
        self.strength = self.quantizer_strength(self.strength)

        return out

    def apply_weight_rescale(self) -> None:
        """对 strength 执行权重重整，使其分布更适合量化。"""
        if self.quant_config.weight_rescale:
            rescaled, factor = weight_rescale(self.strength.data, method='max')
            self.strength.data = rescaled
            self._rescale_factor = factor
            # 如果量化器 observer 重新统计，需注意，此处简单处理
            if hasattr(self.quantizer_strength, '_update_observer'):
                self.quantizer_strength._update_observer(self.strength)


# ============================================================================
# 7. 训练后量化（PTQ）
# ============================================================================

def quantize_model_ptq(
        model: nn.Module,
        config: QuantConfig,
        calib_data: Optional[Any] = None,
        calib_steps: int = 10
) -> nn.Module:
    """对已训练模型执行一次性权重量化（PTQ）。

    方法：
        1. 使用校准数据（若提供）运行少量时间步，统计各状态变量的观测范围。
        2. 根据统计值计算量化参数并固定。
        3. 将模型转换为量化版本（替换子模块），并开关 weight_rescale。

    参数:
        model: 已训练的 Genesis 模型（nn.Module）。
        config: 量化配置。
        calib_data: 校准数据加载器或输入张量，用于统计激活范围。若为 None，则仅基于权重静态范围量化。
        calib_steps: 校准步数。
    返回:
        量化后的模型（内部模块已替换为量化版本，量化参数固定）。
    """
    # 使用序列化复制模型，避免 deepcopy 对非叶节点张量的潜在问题
    # 同时保持对新实例的修改不影响原模型
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    qmodel = torch.load(buffer, weights_only=False)

    # 第一步：替换模块为量化版本（但量化器尚未固定）
    qmodel = _replace_modules_for_qat(qmodel, config)
    qmodel.train()  # 启用 observer 更新

    # 第二步：如果有校准数据，运行模型收集统计
    if calib_data is not None:
        with torch.no_grad():
            # 重置状态
            if hasattr(qmodel, 'reset_state'):
                qmodel.reset_state(1)
            for i, data in enumerate(calib_data):
                if i >= calib_steps:
                    break
                # 假设 calib_data 产生 (x, ...) 输入
                if isinstance(data, (tuple, list)):
                    x = data[0]
                else:
                    x = data
                _ = qmodel(x)
        # 校准结束后，固定量化参数并切换到 eval
        _fix_quantizer_params(qmodel)

    qmodel.eval()
    return qmodel


def _fix_quantizer_params(model: nn.Module) -> None:
    """遍历模型，将所有 FakeQuantize / ThresholdCenteredQuantize 的 observer 禁用，固定 scale/zero_point。"""
    for module in model.modules():
        if isinstance(module, (FakeQuantize, ThresholdCenteredQuantize)):
            module.disable_observer()
            module.fake_quant_enabled = True


# ============================================================================
# 8. 量化感知训练（QAT）接口
# ============================================================================

class QATModel(nn.Module):
    """量化感知训练包装器。

    内部持有训练模型，并在前向传播中应用伪量化。
    该包装器主要负责状态管理，实际的量化替换在 `prepare_qat` 中完成。

    参数:
        model: 已转换为量化子模块的模型（通常由 prepare_qat 返回）。
        config: 量化配置。
    """

    def __init__(self, model: nn.Module, config: QuantConfig):
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self


def prepare_qat(model: nn.Module, config: QuantConfig) -> QATModel:
    """准备模型进行量化感知训练（QAT）。

    递归遍历模型，将 MorphoNeuron 替换为 QuantMorphoNeuron，
    将 ResonanceSynapse 替换为 QuantResonanceSynapse，并插入量化器。

    参数:
        model: 原始全精度 Genesis 模型。
        config: 量化配置。
    返回:
        QATModel 实例，其内部已替换为量化子模块且处于训练模式。
    """
    qmodel = _replace_modules_for_qat(model, config)
    qmodel.train()  # 启用 observer
    return QATModel(qmodel, config)


def _replace_modules_for_qat(model: nn.Module, config: QuantConfig) -> nn.Module:
    """递归替换子模块为量化版本。"""
    for name, child in model.named_children():
        if isinstance(child, MorphoNeuron) and not isinstance(child, QuantMorphoNeuron):
            # 辅助函数：将任意值转换为 Python 标量
            def _to_scalar(t):
                if isinstance(t, torch.Tensor):
                    return t.item() if t.numel() == 1 else t.reshape(-1)[0].item()
                return t

            # 提取构造参数（全部转换为标量，以防多元素张量导致构造器错误）
            nrn_params = {
                'num_neurons': _to_scalar(child.num_neurons),
                'num_dendrites': _to_scalar(child.num_dendrites),
                'tau_u': _to_scalar(child.tau_u),
                'rest_h': _to_scalar(child.rest_h),
                'threshold': _to_scalar(child.threshold),
                'refractory_period': _to_scalar(child.refractory_period),
                'axon_delay': _to_scalar(child.axon_delay),
                'bias_dim': _to_scalar(child.r_dim),
                'noise_scale': _to_scalar(child.noise_scale),
            }
            # 创建量化版本
            new_child = QuantMorphoNeuron(quant_config=config, **nrn_params)
            # 加载原状态（忽略多余或缺少的键）
            _copy_state(child, new_child)
            setattr(model, name, new_child)
        elif isinstance(child, ResonanceSynapse) and not isinstance(child, QuantResonanceSynapse):
            syn_params = {
                'num_synapses': child.num_synapses,
                'pool_total': child.pool_total,
                'k_cyc': child.k_cyc,
                'k_rp': child.k_rp,
                'p_base': child.p_base,
                'ca_decay': child.ca_decay,
                'facil_decay': child.facil_decay,
                'facil_factor': child.facil_factor,
                'memory_size': child.memory_size,
                'A_plus': child.A_plus,
                'A_minus': child.A_minus,
                'trace_decay': child.trace_decay,
            }
            new_child = QuantResonanceSynapse(quant_config=config, **syn_params)
            _copy_state(child, new_child)
            setattr(model, name, new_child)
        else:
            # 递归进入子模块，并将修改后的子模块赋值回模型
            setattr(model, name, _replace_modules_for_qat(child, config))
    return model


def _copy_state(src: nn.Module, dst: nn.Module) -> None:
    """将 src 的 state_dict 尽可能复制到 dst，忽略形状不匹配或缺失的键。"""
    src_state = src.state_dict()
    dst_state = dst.state_dict()
    for key in dst_state:
        if key in src_state and dst_state[key].shape == src_state[key].shape:
            dst_state[key].copy_(src_state[key])
    # 可选：重新加载 dst 的 state_dict
    # 由于直接修改了 dst_state 的 tensor，无需再次 load，但需确保 buffer 已更新。
    # 简单起见，我们使用 load_state_dict 并 strict=False
    dst.load_state_dict(dst_state, strict=False)


def convert_qat(qat_model: QATModel) -> nn.Module:
    """将经过 QAT 训练的模型转换为推理模型。

    该过程会冻结所有量化参数（scale, zero_point），禁用 observer，
    并可选择将权重转换为静态量化整数（实际仍用浮点模拟，但量化器固定）。

    参数:
        qat_model: QAT 训练完成的 QATModel 实例。
    返回:
        推理用的量化模型（nn.Module），可直接用于评估或部署。
    """
    model = qat_model.model
    model.eval()  # 停止 observer 更新
    # 固定所有量化器的参数
    _fix_quantizer_params(model)
    # 应用权重重整（如果启用）
    for module in model.modules():
        if isinstance(module, QuantResonanceSynapse):
            module.apply_weight_rescale()
    return model


# ============================================================================
# 9. 便捷接口
# ============================================================================

def quantize_model(
        model: nn.Module,
        config: QuantConfig,
        method: str = 'ptq',
        calib_data: Optional[Any] = None,
        calib_steps: int = 10
) -> nn.Module:
    """一键量化入口，自动选择 PTQ 或 QAT。

    参数:
        model: 待量化模型。
        config: 量化配置。
        method: 量化方法，'ptq' 或 'qat'。若为 'qat'，返回 QAT ready 的包装器，用户需自行训练。
        calib_data: PTQ 校准数据（仅 'ptq' 方法使用）。
        calib_steps: PTQ 校准步数。
    返回:
        量化后的模型（PTQ）或 QATModel 实例（QAT）。
    """
    if method == 'ptq':
        return quantize_model_ptq(model, config, calib_data, calib_steps)
    elif method == 'qat':
        return prepare_qat(model, config)
    else:
        raise ValueError(f"未知量化方法: {method}，请使用 'ptq' 或 'qat'")


def save_quantized_model(model: nn.Module, path: str, config: Optional[QuantConfig] = None) -> None:
    """保存量化模型，包括 state_dict 和量化配置。

    参数:
        model: 量化后的模型（nn.Module）。
        path: 保存路径（建议后缀 .pth 或 .life.q）。
        config: 可选，保存时附带的 QuantConfig。
    """
    save_dict = {
        'state_dict': model.state_dict(),
        'quant_config': config,
    }
    torch.save(save_dict, path)


def load_quantized_model(
        path: str,
        model_class: Callable[..., nn.Module],
        config: Optional[QuantConfig] = None,
        **model_kwargs
) -> nn.Module:
    """从文件加载量化模型。

    参数:
        path: 模型文件路径。
        model_class: 用于实例化原始模型的可调用对象（如搭建好的模型类）。
        config: 若文件内未包含 QuantConfig，则需要外部提供。
        **model_kwargs: 传递给 model_class 的构造参数。
    返回:
        加载并恢复的量化模型（若包含量化器，则一并恢复）。
    """
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    saved_config = checkpoint.get('quant_config', config)

    # 实例化模型（假设 model_class 接受 config 参数，若没有则忽略）
    try:
        model = model_class(quant_config=saved_config, **model_kwargs)
    except TypeError:
        model = model_class(**model_kwargs)

    model.load_state_dict(state_dict, strict=False)
    return model


# ============================================================================
# 10. 辅助函数：手动预处理（可选）
# ============================================================================

def apply_precision(model: nn.Module, precision: Union[str, PrecisionConfig]) -> nn.Module:
    """快速切换模型的精度（混合精度控制）。

    参数:
        model: Genesis 模型。
        precision: 若为字符串，仅支持 'fp32', 'fp16', 'amp'；若为 PrecisionConfig，则按变量切换。
    返回:
        转换后的模型（原地修改）。
    """
    if isinstance(precision, str):
        if precision in ('fp32', 'fp16', 'bf16'):
            dtype = PRECISION_MAP[precision]
            model.to(dtype=dtype)
        elif precision == 'amp':
            # 启用自动混合精度（需配合 torch.cuda.amp）
            # 此处不直接修改参数，仅记录配置供外部使用
            pass
        else:
            raise ValueError(f"未知精度命令: {precision}")
    elif isinstance(precision, PrecisionConfig):
        # 按变量名逐一转换 dtype
        for var_name, dtype_str in precision.var_dtype_map.items():
            dtype = PRECISION_MAP[dtype_str]
            # 转换模型中的 buffer 和 parameter（如果名称匹配）
            for name, buf in model.named_buffers():
                if var_name in name:  # 粗略匹配
                    buf.data = buf.data.to(dtype=dtype)
            for name, param in model.named_parameters():
                if var_name in name:
                    param.data = param.data.to(dtype=dtype)
    return model