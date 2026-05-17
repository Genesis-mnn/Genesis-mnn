# utils.py — Genesis 框架辅助工具箱
# 提供网络拓扑构建、集群管理、稀疏连接生成以及数据加载等通用工具。
# 对应白皮书 10.8 节（集群管理）、稀疏连接生成及 10.7 节（神经拟态数据支持）。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional, Callable, Union, Tuple, Type, Any
from dataclasses import dataclass, field
import random
import math
import warnings
import inspect

# 导入核心组件
from .genesis_core import MorphoNeuron, ResonanceSynapse
from . import genesis_core
from .neuron import (
    IFNeuron, LIFNeuron, PLIFNeuron, QIFNeuron, EIFNeuron, IzhikevichNeuron,
)
from .synapse import STDPSynapse, PlasticityRule

# ============================================================================
# 集群数据结构 —— 对应白皮书 10.8 节
# ============================================================================

@dataclass
class Layer:
    """一层同质神经元，对应白皮书 create_layer。"""
    name: str
    neuron_group: MorphoNeuron   # 包含该层所有神经元的组对象（一个实例管理多个神经元）
    size: int                    # 神经元数量
    neuron_type: str             # 神经元类型名称

    def get_indices(self) -> List[int]:
        """返回层内所有神经元的局部索引列表。"""
        return list(range(self.size))

    def get_neuron_group(self) -> MorphoNeuron:
        """返回管理该层神经元的组对象。"""
        return self.neuron_group


@dataclass
class Assembly:
    """神经元集群（任意编组），对应白皮书 create_assembly。"""
    name: str
    parent: Layer                # 所属的层（目前每个 Assembly 必须依附于一个 Layer）
    indices: List[int]           # 在 parent 层内的神经元局部索引列表

    def __post_init__(self):
        """验证索引的有效性。"""
        max_idx = self.parent.size - 1
        for idx in self.indices:
            if idx < 0 or idx > max_idx:
                raise ValueError(
                    f"Assembly 索引 {idx} 超出层 '{self.parent.name}' 的范围 [0, {max_idx}]"
                )

    def get_indices(self) -> List[int]:
        return self.indices

    def get_neuron_group(self) -> MorphoNeuron:
        return self.parent.get_neuron_group()


@dataclass
class Column:
    """功能柱，对应白皮书 create_column。"""
    name: str
    layers: List[Layer]                         # 组成该柱的层列表
    connections: List['Connection'] = field(default_factory=list)  # 层间连接

    def get_layers(self) -> List[Layer]:
        """返回柱内所有层。"""
        return self.layers


@dataclass
class BrainRegion:
    """宏观脑区，对应白皮书 create_brain_region。"""
    name: str
    columns: List[Column]                       # 组成该脑区的功能柱列表
    connections: List['Connection'] = field(default_factory=list)  # 柱间连接

    def get_columns(self) -> List[Column]:
        """返回脑区内所有柱。"""
        return self.columns


@dataclass
class Connection:
    """集群间连接，对应白皮书 connect。"""
    name: str
    synapse: ResonanceSynapse                   # 共鸣突触实例（公理二）
    pre_indices: torch.Tensor                    # (num_synapses,) 源集群内的局部神经元索引
    post_indices: torch.Tensor                   # (num_synapses,) 目标集群内的局部神经元索引
    src: Union[Layer, Assembly]                  # 源集群
    dst: Union[Layer, Assembly]                  # 目标集群
    rule: str                                    # 使用的连接规则名称

    def get_pre_group(self) -> MorphoNeuron:
        """返回源神经元组对象。"""
        return self.src.get_neuron_group()

    def get_post_group(self) -> MorphoNeuron:
        """返回目标神经元组对象。"""
        return self.dst.get_neuron_group()

    def reset_state(self, batch_size: int, device: Optional[torch.device] = None):
        """重置突触的内部状态，以便从一个干净的批次开始运行（公理二）。"""
        self.synapse.reset_state(batch_size, device)


# ============================================================================
# 集群管理接口 —— 对应白皮书 10.8 节
# ============================================================================

def create_assembly(
    neuron_ids: List[int],
    name: str,
    parent_layer: Layer,
) -> Assembly:
    """将任意神经元编组为一个集群。对应白皮书 10.8 节。

    公理四（存在先于任务）：集群编组仅为拓扑管理，不引入外部任务信号。

    参数:
        neuron_ids: 要在集群中包含的神经元局部索引（相对于 parent_layer）。
        name: 集群名称。
        parent_layer: 神经元所在的层对象（必需）。

    返回:
        Assembly 实例。
    """
    return Assembly(name=name, parent=parent_layer, indices=list(neuron_ids))


def create_layer(
    num_neurons: int,
    neuron_type: Union[str, Type[MorphoNeuron]],
    name: Optional[str] = None,
    **neuron_kwargs,
) -> Layer:
    """创建一层同质神经元。对应白皮书 10.8 节。

    公理一（内在性优先）：创建的神经元内置健康度、敏感度等内在变量，
    不依赖外部损失函数。公理四：神经元初始状态即可实现自持活动。

    参数:
        num_neurons: 层中神经元的数量。
        neuron_type: 神经元类型，可以是字符串（如 'LIF'）或 MorphoNeuron 子类。
        name: 层的名称，若未提供则自动生成。
        **neuron_kwargs: 传递给神经元构造函数的额外参数。
            例如 tau=10.0, v_threshold=1.0, R=1.0 等。

    返回:
        Layer 实例，包含创建好的 MorphoNeuron 组对象。
    """
    # 获取神经元类
    if isinstance(neuron_type, str):
        neuron_type_map = genesis_core.get_neuron_type_map()
        if neuron_type not in neuron_type_map:
            raise ValueError(
                f"未知的神经元类型字符串: '{neuron_type}'。可用: {list(neuron_type_map.keys())}"
            )
        neuron_cls = neuron_type_map[neuron_type]
    elif isinstance(neuron_type, type) and issubclass(neuron_type, MorphoNeuron):
        neuron_cls = neuron_type
    else:
        raise TypeError("neuron_type 必须为字符串或 MorphoNeuron 的子类。")

    # 自动生成名称
    if name is None:
        name = f"layer_{neuron_cls.__name__}_{num_neurons}"

    # 提取阈值参数（兼容两种参数名，优先 'v_threshold'）
    v_thresh_val = neuron_kwargs.pop('v_threshold', None)
    thresh_val = neuron_kwargs.pop('threshold', None)
    if v_thresh_val is not None:
        threshold_val = v_thresh_val
    elif thresh_val is not None:
        threshold_val = thresh_val
    else:
        threshold_val = 1.0

    # 确定神经元类使用的阈值参数名
    try:
        init_sig = inspect.signature(neuron_cls.__init__)
        params = init_sig.parameters
        if 'v_threshold' in params:
            threshold_key = 'v_threshold'
        elif 'threshold' in params:
            threshold_key = 'threshold'
        else:
            threshold_key = 'v_threshold'
    except (ValueError, TypeError):
        # 无法检查时，默认使用 'v_threshold'（预置神经元标准）
        threshold_key = 'v_threshold'

    # 构建构造参数（允许用户覆盖）
    kwargs = {
        'num_neurons': num_neurons,
        'num_dendrites': neuron_kwargs.pop('num_dendrites', 1),
        'rest_h': neuron_kwargs.pop('rest_h', 0.7),
        threshold_key: threshold_val,
    }
    # 合并用户自定义参数
    kwargs.update(neuron_kwargs)

    try:
        neuron_group = neuron_cls(**kwargs)
    except Exception as e:
        raise RuntimeError(f"无法创建神经元层: {e}")

    return Layer(
        name=name,
        neuron_group=neuron_group,
        size=num_neurons,
        neuron_type=neuron_type if isinstance(neuron_type, str) else neuron_type.__name__,
    )


def create_column(
    layers: List[Layer],
    connections: Optional[List[Connection]] = None,
    name: Optional[str] = None,
) -> Column:
    """将多层组织为功能柱。对应白皮书 10.8 节。

    公理一（内在性优先）：柱内各层协同维持整体稳态。

    参数:
        layers: 组成该柱的层列表。
        connections: 已建立的层间 Connection 对象列表（可选）。
        name: 柱名称，若未提供则自动生成。

    返回:
        Column 实例。
    """
    if name is None:
        name = f"column_{len(layers)}"
    if connections is None:
        connections = []
    return Column(name=name, layers=layers, connections=connections)


def create_brain_region(
    columns: List[Column],
    connections: Optional[List[Connection]] = None,
    name: Optional[str] = None,
) -> BrainRegion:
    """定义由多个功能柱构成的宏观脑区。对应白皮书 10.8 节。

    公理四（存在先于任务）：脑区提供自足的宏观拓扑，不依赖外部任务信号。

    参数:
        columns: 组成该脑区的功能柱列表。
        connections: 已建立的柱间 Connection 对象列表（可选）。
        name: 脑区名称，若未提供则自动生成。

    返回:
        BrainRegion 实例。
    """
    if name is None:
        name = f"region_{len(columns)}"
    if connections is None:
        connections = []
    return BrainRegion(name=name, columns=columns, connections=connections)


def connect(
    src: Union[Layer, Assembly],
    dst: Union[Layer, Assembly],
    rule: Union[str, Callable[..., Tuple[List[int], List[int]]]] = 'full',
    synapse_class: Type[ResonanceSynapse] = ResonanceSynapse,
    synapse_kwargs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    **rule_kwargs,
) -> Connection:
    """基于概率或规则的集群间连接生成器。对应白皮书 10.8 节。

    公理二（共鸣取代权重）：突触使用 ResonanceSynapse，其通透性由节律差决定。
    公理三（演化优于训练）：连接建立后可通过局部可塑性（如 STDP）自发演化。

    支持多种规则（全连接、随机、小世界等），在源和目标集群之间创建突触连接。
    若 src 或 dst 为 Assembly，则仅在该集群的指定神经元子集上建立连接。

    参数:
        src: 源集群，可以是 Layer 或 Assembly。
        dst: 目标集群，可以是 Layer 或 Assembly。
        rule: 连接规则字符串或自定义函数，传递给 generate_sparse_connectivity。
            'full'：全连接；'random'：随机连接（需提供概率 p）；
            'small_world'：小世界网络（需 src 与 dst 尺寸相同，如自连接）。
        synapse_class: 使用的突触类，默认 ResonanceSynapse（可替换为 STDPSynapse 等）。
        synapse_kwargs: 传递给突触构造函数的字典（如 pool_total=1.0 等），
            num_synapses 由框架自动设置。
        name: 连接名称，若未提供则自动生成。
        **rule_kwargs: 连接规则参数，如 p, k 等。

    返回:
        Connection 对象，包含突触实例和索引映射。
    """
    # 获取源和目标集群的局部索引列表
    src_indices = src.get_indices()      # List[int]
    dst_indices = dst.get_indices()

    num_src = len(src_indices)
    num_dst = len(dst_indices)

    # 生成稀疏连接模式（局部索引，范围 0..num_src-1, 0..num_dst-1）
    pre_local, post_local = generate_sparse_connectivity(num_src, num_dst, rule, **rule_kwargs)

    if len(pre_local) == 0:
        raise ValueError("生成的连接数为零，请检查规则参数或集群大小。")

    # 将局部索引映射回原始集群内的全局局部索引
    pre_indices_mapped = [src_indices[i] for i in pre_local]
    post_indices_mapped = [dst_indices[j] for j in post_local]

    num_synapses = len(pre_indices_mapped)

    # 创建突触实例 —— 公理二、公理三
    syn_kwargs = synapse_kwargs.copy() if synapse_kwargs else {}
    syn_kwargs['num_synapses'] = num_synapses
    synapse = synapse_class(**syn_kwargs)

    # 自动生成名称
    conn_name = name or f"conn_{rule if isinstance(rule, str) else 'custom'}_{num_synapses}"

    return Connection(
        name=conn_name,
        synapse=synapse,
        pre_indices=torch.tensor(pre_indices_mapped, dtype=torch.long),
        post_indices=torch.tensor(post_indices_mapped, dtype=torch.long),
        src=src,
        dst=dst,
        rule=rule if isinstance(rule, str) else 'custom',
    )


# ============================================================================
# 稀疏连接生成工具 —— 对应白皮书 10.8 节稀疏连接部分
# ============================================================================

def generate_sparse_connectivity(
    num_src: int,
    num_dst: int,
    rule: Union[str, Callable[..., Tuple[List[int], List[int]]]] = 'random',
    **kwargs,
) -> Tuple[List[int], List[int]]:
    """生成稀疏连接索引（COO 格式）。对应白皮书 10.8 节稀疏连接工具。

    返回源和目标神经元在各自集群内的局部索引列表，可用于构建突触。
    所有规则产生的连接均为单向（源 → 目标）。

    参数:
        num_src: 源集群神经元数量。
        num_dst: 目标集群神经元数量。
        rule: 连接规则。支持以下字符串或自定义函数：
            - 'full': 全连接（每个源到每个目标）。
            - 'random': 随机连接，需提供概率 p (float, 0-1)。
            - 'small_world': 小世界网络（Watts-Strogatz算法）。
                适用于 src == dst 或 num_src == num_dst。需提供 k (近邻数) 和 p (重连概率)。
            - 可调用对象: 自定义函数，接收 (num_src, num_dst, **kwargs) 并返回 (pre, post)。
        **kwargs: 传递给规则的额外参数。

    返回:
        pre_indices: 源神经元局部索引列表。
        post_indices: 目标神经元局部索引列表。两者等长。
    """
    if isinstance(rule, str):
        if rule == 'full':
            # 全连接：所有源→所有目标
            pre = []
            post = []
            for i in range(num_src):
                for j in range(num_dst):
                    pre.append(i)
                    post.append(j)
            return pre, post

        elif rule == 'random':
            # 随机连接：每对可能的连接以概率 p 独立存在
            p = kwargs.get('p', 0.1)
            if not (0 <= p <= 1):
                raise ValueError(f"随机连接概率 p 需在 [0,1] 内，收到 {p}")
            pre = []
            post = []
            for i in range(num_src):
                for j in range(num_dst):
                    if random.random() < p:
                        pre.append(i)
                        post.append(j)
            return pre, post

        elif rule == 'small_world':
            # 小世界网络（Watts-Strogatz 算法），要求源和目标尺寸相同
            if num_src != num_dst:
                raise ValueError(
                    "小世界网络规则要求源和目标神经元数量相同，通常用于自连接。"
                    f" 当前 num_src={num_src}, num_dst={num_dst}"
                )
            k = kwargs.get('k', 4)
            p = kwargs.get('p', 0.1)
            return _watts_strogatz(num_src, k, p)

        else:
            raise ValueError(f"未知的连接规则: '{rule}'")
    elif callable(rule):
        # 用户自定义规则
        return rule(num_src, num_dst, **kwargs)
    else:
        raise TypeError("rule 必须为字符串或可调用对象")


def _watts_strogatz(n: int, k: int, p: float) -> Tuple[List[int], List[int]]:
    """生成 Watts-Strogatz 小世界网络（有向，无自环）。"""
    if k >= n or k < 2:
        raise ValueError("k 必须是偶数且小于 n")
    # 初始环形正则网络：每个节点向 k/2 个右侧邻居发出连接
    half_k = k // 2
    edges = set()
    for i in range(n):
        for j in range(1, half_k + 1):
            target = (i + j) % n
            edges.add((i, target))
    # 重连：以概率 p 替换目标为随机节点（避免自环和重复）
    new_edges = set()
    for (u, v) in edges:
        if random.random() < p:
            # 随机选择一个新目标（不等于 u）
            candidates = list(set(range(n)) - {u})
            if candidates:
                v_new = random.choice(candidates)
                new_edges.add((u, v_new))
            else:
                new_edges.add((u, v))  # 保留原边
        else:
            new_edges.add((u, v))
    pre = [e[0] for e in new_edges]
    post = [e[1] for e in new_edges]
    return pre, post


# ============================================================================
# 数据加载工具 —— 对应白皮书 10.7 节
# ============================================================================

def load_neuromorphic_dataset(
    name: str,
    root: str = './data',
    batch_size: int = 32,
    train: bool = True,
    download: bool = True,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    **kwargs,
) -> DataLoader:
    """加载标准神经拟态数据集。对应白皮书 10.7 节。

    支持 DVS128 Gesture、N-Caltech101、CIFAR10-DVS 等标准数据集。
    内部优先使用 `tonic` 库，若不可用则尝试 `spikingjelly`。
    如果两者均未安装，将抛出 ImportError 并提示安装。

    公理四（存在先于任务）：数据集加载仅为提供初始外部脉冲输入，不影响网络的内在自持活动。

    参数:
        name: 数据集名称标识符。例如 'dvs-gesture', 'n-caltech101', 'cifar10-dvs'。
        root: 数据集存储根目录。
        batch_size: 批大小。
        train: 是否为训练集，否则为测试集。
        download: 是否下载数据集（如果尚未存在）。
        num_workers: DataLoader 的工作线程数。
        transform: 应用于数据样本的变换（建议提供，否则将返回原始事件/帧）。
        target_transform: 应用于标签的变换。
        **kwargs: 传递给 DataLoader 的额外参数。

    返回:
        torch.utils.data.DataLoader 实例。
    """
    # 统一转为小写
    name_lower = name.lower().strip()

    # ---------- DVS Gesture ----------
    if name_lower in ('dvs-gesture', 'dvs_gesture', 'dvsgesture'):
        # 尝试 tonic
        try:
            import tonic
            from tonic.datasets import DVSGesture
            sensor_size = DVSGesture.sensor_size
            # 若未提供 transform，使用基本的帧转换（需要 tonic.transforms）
            if transform is None:
                try:
                    from tonic.transforms import Compose, ToFrame, Downsample
                    transform = Compose([
                        Downsample(spatial_factor=0.5),
                        ToFrame(sensor_size=sensor_size, time_window=30000),
                    ])
                except ImportError:
                    warnings.warn("无法导入 tonic.transforms，将使用原始事件加载器。请手动指定 transform。")
            dataset = DVSGesture(
                save_to=root, train=train,
                transform=transform, target_transform=target_transform,
                download=download
            )
            return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                              num_workers=num_workers, **kwargs)
        except ImportError:
            # 尝试 spikingjelly
            try:
                # spikingjelly 暂时无内置 DVS Gesture 直接加载，抛错
                raise ImportError("spikingjelly 暂不支持直接加载 DVS Gesture，请安装 tonic。")
            except ImportError:
                raise ImportError(
                    "无法加载 DVS Gesture 数据集。请安装 tonic 库：pip install tonic"
                )

    # ---------- N-Caltech101 ----------
    elif name_lower in ('n-caltech101', 'ncaltech101', 'ncaltech'):
        try:
            import tonic
            from tonic.datasets import NCaltech101
            if transform is None:
                from tonic.transforms import Compose, ToFrame, Downsample
                sensor_size = NCaltech101.sensor_size
                transform = Compose([
                    Downsample(spatial_factor=0.5),
                    ToFrame(sensor_size=sensor_size, time_window=10000),
                ])
            dataset = NCaltech101(
                save_to=root, train=train, transform=transform,
                target_transform=target_transform, download=download
            )
            return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                              num_workers=num_workers, **kwargs)
        except ImportError:
            try:
                import spikingjelly.datasets as sjd
                # spikingjelly 中为 NCaltech101
                dataset = sjd.NCaltech101(
                    root=root, train=train, transform=transform,
                    target_transform=target_transform, download=download
                )
                return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                  num_workers=num_workers, **kwargs)
            except ImportError:
                raise ImportError(
                    "无法加载 N-Caltech101 数据集。请安装 tonic 或 spikingjelly。"
                )

    # ---------- CIFAR10-DVS ----------
    elif name_lower in ('cifar10-dvs', 'cifar10dvs', 'cifar-dvs'):
        try:
            import tonic
            from tonic.datasets import CIFAR10DVS
            if transform is None:
                from tonic.transforms import Compose, ToFrame, Downsample
                sensor_size = CIFAR10DVS.sensor_size
                transform = Compose([
                    Downsample(spatial_factor=0.5),
                    ToFrame(sensor_size=sensor_size, time_window=5000),
                ])
            dataset = CIFAR10DVS(
                save_to=root, train=train, transform=transform,
                target_transform=target_transform, download=download
            )
            return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                              num_workers=num_workers, **kwargs)
        except ImportError:
            try:
                import spikingjelly.datasets as sjd
                dataset = sjd.CIFAR10DVS(
                    root=root, train=train, transform=transform,
                    target_transform=target_transform, download=download
                )
                return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                  num_workers=num_workers, **kwargs)
            except ImportError:
                raise ImportError(
                    "无法加载 CIFAR10-DVS 数据集。请安装 tonic 或 spikingjelly。"
                )

    else:
        raise ValueError(
            f"未知的数据集名称: '{name}'。目前支持: dvs-gesture, n-caltech101, cifar10-dvs。"
            " 如需加载自定义数据集，请使用 create_custom_dataloader 函数。"
        )


def create_custom_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """从用户自定义的 PyTorch Dataset 创建 DataLoader。对应白皮书 10.7 节用户自定义接口。

    公理四（存在先于任务）：该接口仅创建数据流管道，网络运行仍由其内部稳态驱动。

    参数:
        dataset: torch.utils.data.Dataset 实例（通常为事件流或脉冲帧数据集）。
        batch_size: 批大小。
        shuffle: 是否打乱数据。
        num_workers: 工作线程数。
        **kwargs: 传递给 DataLoader 的其他参数（如 drop_last, pin_memory 等）。

    返回:
        torch.utils.data.DataLoader 实例。
    """
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, **kwargs
    )


# ============================================================================
# 模块公开接口
# ============================================================================
__all__ = [
    'Layer', 'Assembly', 'Column', 'BrainRegion', 'Connection',
    'create_assembly', 'create_layer', 'create_column', 'create_brain_region',
    'connect', 'generate_sparse_connectivity',
    'load_neuromorphic_dataset', 'create_custom_dataloader',
]