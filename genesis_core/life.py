#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genesis 数字生命持久化模块 (life.py)

实现 .life 文件的保存、加载、演化历史哈希与防克隆唯一性机制。
严格遵循《创世纪框架白皮书-v2.6》第 10.11 节定义的 .life 文件标准。

核心组件：
    - EvolutionHasher: 实时累积网络状态，生成不可逆的演化历史哈希。
    - LifeSaver: 将网络完整状态保存为 .life 文件（HDF5 容器）。
    - LifeLoader: 从 .life 文件重建网络，包含完整性校验与唯一性检查。
    - GenesisNetwork: 网络组件的持久化容器。
    - LifeRegistry: 可选的生命文件哈希注册表，用于生态防克隆。

所有术语严格遵守“术语中立宪章”，不引入任何违禁词。

【参数还原策略说明】
本模块在保存时尽力记录所有影响动力学行为的构造参数（config），加载时：
1. 优先使用保存的 config 构造实例，确保构造参数与原始网络一致。
2. 若 config 中缺失某些关键参数（例如旧版本 .life 文件未保存该参数，
   或参数不属于预定义的配置键列表），则从 state_dict 中寻找同名的标量张量
   补充到 config 中，再调用构造函数。
3. 最后通过 load_state_dict 恢复完整的参数与缓冲区状态。
此策略保证即使构造参数列表发生变化，仍能最大限度地还原保存瞬间的动力学。
"""

import h5py
import torch
import torch.nn as nn
import numpy as np
import hashlib
import datetime
import json
import uuid
import warnings
import inspect
from typing import Dict, List, Optional, Any, Tuple, Union, Type, Callable
from pathlib import Path
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 导入 Genesis 核心模块（确保与 life.py 位于同一目录，或正确配置PYTHONPATH）
# ---------------------------------------------------------------------------
from .genesis_core import MorphoNeuron, ResonanceSynapse
from . import genesis_core
from .neuron import (
    IFNeuron, LIFNeuron, PLIFNeuron, QIFNeuron, EIFNeuron, IzhikevichNeuron,
)
from .synapse import STDPSynapse, PlasticityRule
from .glia import Astrocyte, Microglia, GlialNetwork
from .modulator import (
    Modulator, ModulatorSystem,
    RewardSignal, SatietyFactor, NoveltyResponder,
    StressActivator, FatigueAccumulator,
)


# ============================================================================
# 自定义异常
# ============================================================================

class IntegrityError(Exception):
    """当 .life 文件的快照完整性哈希与当前计算不匹配时抛出。"""
    pass


class LifeFormatError(Exception):
    """当 .life 文件格式不正确、缺失关键数据或不完整时抛出。"""
    pass


# ============================================================================
# 辅助数据结构
# ============================================================================

@dataclass
class SynapticConnection:
    """突触连接，包含突触实例及其前后神经元的局部索引（COO格式）。"""
    synapse: ResonanceSynapse
    pre_indices: torch.Tensor  # shape (num_synapses,), dtype int64
    post_indices: torch.Tensor  # shape (num_synapses,), dtype int64
    name: Optional[str] = None


# ============================================================================
# 网络容器（用于持久化操作）
# ============================================================================

class GenesisNetwork:
    """创世纪网络容器，用于统一保存/加载所有组件。

    属性:
        neurons: 形态神经元列表。
        connections: 突触连接列表。
        glia: GlialNetwork 组件（可选）。
        modulators: 全局调制场系统（可选）。
        architecture: 网络拓扑定义（字典）。
        meta: 元数据字典（创建者、时间戳等）。
    """

    def __init__(self):
        self.neurons: List[MorphoNeuron] = []
        self.connections: List[SynapticConnection] = []
        self.glia: Optional[GlialNetwork] = None
        self.modulators: Optional[ModulatorSystem] = None
        self.architecture: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {}

    def to(self, device: torch.device) -> 'GenesisNetwork':
        """将所有内部张量迁移到指定设备。"""
        for nrn in self.neurons:
            nrn.to(device)
        for conn in self.connections:
            conn.synapse.to(device)
            if conn.pre_indices.device != device:
                conn.pre_indices = conn.pre_indices.to(device)
                conn.post_indices = conn.post_indices.to(device)
        # Glial 网络和调制系统主要包含 Python 对象，如有张量可在此扩展
        return self


# ============================================================================
# 演化历史哈希器
# ============================================================================

class EvolutionHasher:
    """实时演化历史哈希器。

    通过持续吸收神经元与突触的状态更新，生成不可逆的链式哈希。
    内部使用 SHA-256 链式哈希：每次 update 将当前哈希与新的状态字节拼接后重新哈希。
    用于在运行生命周期中逐步累积唯一的演化历史。
    """

    def __init__(self):
        """初始化空的演化历史哈希器。"""
        self._hasher = hashlib.sha256()
        self._counter = 0

    def update(self, state_bytes: bytes) -> None:
        """吸收一次状态更新事件，累积更新哈希值。

        参数:
            state_bytes: 序列化后的状态字节（例如神经元 u, h, r 和突触 strength 的拼接）。
        """
        # 链式更新：当前哈希拼接新数据再哈希
        current_digest = self._hasher.digest()
        self._hasher = hashlib.sha256(current_digest + state_bytes)
        self._counter += 1

    def get_hash(self) -> str:
        """返回当前累积哈希的十六进制字符串。"""
        return self._hasher.hexdigest()

    def reset(self) -> None:
        """重置哈希器到初始状态。"""
        self._hasher = hashlib.sha256()
        self._counter = 0


# ============================================================================
# 序列化辅助函数
# ============================================================================

def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """将张量转换为连续的字节表示（用于哈希计算）。"""
    return tensor.detach().cpu().numpy().tobytes()


def _sanitize_key(key: str) -> str:
    """将可能包含路径分隔符的键转换为安全名称。"""
    return key.replace('.', '_').replace('/', '_')


# ---------- 配置键映射（保存哪些构造参数） ----------
# 扩展神经元配置键，加入所有影响动力学行为的关键参数（dt, tau, R, v_reset, threshold, rest_h 等）
# 注意：某些参数在子类中可能不存在，hasattr 检查会自动跳过；非标量张量会被安全跳过。
_NEURON_BASE_KEYS = [
    'num_neurons', 'num_dendrites', 'refractory_period',
    'axon_delay', 'bias_dim', 'noise_scale',
    'dt', 'tau', 'R', 'v_reset', 'threshold', 'rest_h',
]

NEURON_CONFIG_KEYS = {
    'MorphoNeuron': _NEURON_BASE_KEYS,
    'IFNeuron': _NEURON_BASE_KEYS,
    'LIFNeuron': _NEURON_BASE_KEYS,
    'PLIFNeuron': _NEURON_BASE_KEYS,
    'QIFNeuron': _NEURON_BASE_KEYS,
    'EIFNeuron': _NEURON_BASE_KEYS,
    'IzhikevichNeuron': _NEURON_BASE_KEYS + ['homeo_gain', 'noise_std'],
}

# 扩展突触配置键，补充通用动力学参数（dt, tau 等），确保所有关键参数被记录
_SYNAPSE_BASE_KEYS = [
    'num_synapses', 'pool_total', 'k_cyc', 'k_rp',
    'p_base', 'ca_decay', 'facil_decay', 'facil_factor',
    'memory_size', 'A_plus', 'A_minus', 'trace_decay',
    'dt', 'tau',
]

SYNAPSE_CONFIG_KEYS = {
    'ResonanceSynapse': _SYNAPSE_BASE_KEYS,
    'PlasticityRule': _SYNAPSE_BASE_KEYS,
    'STDPSynapse': _SYNAPSE_BASE_KEYS + [
        'tau_pre', 'tau_post', 'lr', 'reward_scale',
    ],
}

MODULATOR_CLASS_MAP: Dict[str, Type[Modulator]] = {
    'RewardSignal': RewardSignal,
    'SatietyFactor': SatietyFactor,
    'NoveltyResponder': NoveltyResponder,
    'StressActivator': StressActivator,
    'FatigueAccumulator': FatigueAccumulator,
}


def _get_object_config(obj: Any, config_keys: List[str]) -> Dict[str, Any]:
    """从对象中提取指定键的配置值（通常为非张量的简单属性）。

    若属性为标量张量（numel()==1），自动提取其 Python 标量；
    若为非标量张量，则跳过并发出警告（这类参数将通过 state_dict 持久化）。
    """
    config = {}
    for key in config_keys:
        if hasattr(obj, key):
            val = getattr(obj, key)
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    config[key] = val.item()
                else:
                    warnings.warn(
                        f"配置键 '{key}' 对应的张量形状为 {val.shape}，非标量，"
                        "不会保存为 config 参数。该参数将通过 state_dict 持久化。"
                    )
            else:
                config[key] = val
    return config


def _compute_snapshot_integrity_hash(neurons: List[MorphoNeuron],
                                     connections: List[SynapticConnection]) -> str:
    """计算当前网络快照的完整性哈希。

    独立于演化历史累积，仅基于当前状态值，按固定顺序序列化后计算 SHA-256。
    遍历所有神经元与突触的内部状态，确保快照内容不可篡改。
    """
    hasher = hashlib.sha256()
    for nrn in neurons:
        state = nrn.get_internal_state()
        for key in sorted(state.keys()):
            tensor = state[key]
            hasher.update(_tensor_to_bytes(tensor.contiguous()))
    for conn in connections:
        syn_state = conn.synapse.get_synaptic_state()
        for key in sorted(syn_state.keys()):
            tensor = syn_state[key]
            hasher.update(_tensor_to_bytes(tensor.contiguous()))
    return hasher.hexdigest()


# ============================================================================
# 保存器
# ============================================================================

class LifeSaver:
    """.life 文件保存器。

    将整个网络状态（神经元、突触、Glial 组件、调制场、架构）保存为
    符合白皮书 v2.6 10.11 节的 HDF5 文件，并自动计算演化历史哈希。
    """

    @staticmethod
    def save(
            network: GenesisNetwork,
            filepath: Union[str, Path],
            creator: Optional[str] = None,
            framework_version: str = 'v2.6'
    ) -> None:
        """
        保存网络上所有组件到 .life 文件。

        参数:
            network: GenesisNetwork 实例，包含所有待持久化的组件。
            filepath: 输出文件路径（建议以 .life 结尾）。
            creator: 创建者标识符，若为 None 则使用 'anonymous'。
            framework_version: 创世纪框架版本号。
        """
        filepath = Path(filepath)
        with h5py.File(filepath, 'w') as h5f:
            # ---- 元数据 ----
            meta_grp = h5f.create_group('/meta')
            # 自动生成时间戳和随机种子
            now = datetime.datetime.now(datetime.timezone.utc)
            creation_timestamp = now.isoformat(timespec='milliseconds')
            random_seed = np.random.randint(0, 2 ** 63, dtype=np.int64)

            meta = network.meta or {}
            meta['creator'] = meta.get('creator', creator or 'anonymous')
            meta['creation_timestamp'] = meta.get('creation_timestamp', creation_timestamp)
            meta['random_seed'] = meta.get('random_seed', int(random_seed))

            # 尝试从神经元或突触获取已注入的演化历史哈希器
            evolution_hasher = None
            for nrn in network.neurons:
                if hasattr(nrn, '_evolution_hasher') and nrn._evolution_hasher is not None:
                    evolution_hasher = nrn._evolution_hasher
                    break
            if evolution_hasher is None:
                for conn in network.connections:
                    if hasattr(conn.synapse, '_evolution_hasher') and conn.synapse._evolution_hasher is not None:
                        evolution_hasher = conn.synapse._evolution_hasher
                        break
            if evolution_hasher is not None:
                evolution_hash = evolution_hasher.get_hash()
                meta['hash_source'] = 'cumulative'
            else:
                warnings.warn("网络上未注入EvolutionHasher，将使用临时演化标识。")
                temp_hash_input = f"{creation_timestamp}_{random_seed}".encode('utf-8')
                evolution_hash = hashlib.sha256(temp_hash_input).hexdigest()
                meta['hash_source'] = 'temporary'
            meta['evolution_hash'] = evolution_hash
            # 计算当前状态快照的完整性哈希
            integrity_hash = _compute_snapshot_integrity_hash(network.neurons, network.connections)
            meta['integrity_hash'] = integrity_hash
            meta['framework_version'] = framework_version
            meta['protocol_id'] = 'GenesisLife/1.0'

            # 写入属性（简单类型）或作为 dataset
            for key, value in meta.items():
                if isinstance(value, (int, float, str)):
                    meta_grp.attrs[key] = value
                else:
                    # 避免复杂类型
                    meta_grp.attrs[key] = str(value)

            # ---- 架构 ----
            arch_grp = h5f.create_group('/architecture')
            arch_json = json.dumps(network.architecture if network.architecture else {})
            arch_grp.create_dataset('definition', data=arch_json.encode('utf-8'))

            # ---- 神经元 ----
            neurons_grp = h5f.create_group('/neurons')
            # 记录每个神经元的类型、配置、状态字典
            for i, nrn in enumerate(network.neurons):
                nrn_grp = neurons_grp.create_group(f'neuron_{i:04d}')
                cls_name = type(nrn).__name__
                nrn_grp.attrs['type'] = cls_name
                # 提取配置键
                config_keys = NEURON_CONFIG_KEYS.get(cls_name)
                if config_keys is None:
                    # 回退：尝试提取非张量简单属性
                    config_keys = [k for k, v in vars(nrn).items()
                                   if not isinstance(v, torch.Tensor) and not callable(v) and not k.startswith('_')]
                config = _get_object_config(nrn, config_keys)
                nrn_grp.create_dataset('config', data=json.dumps(config).encode('utf-8'))
                # 保存 state_dict（参数+缓冲区）
                state_grp = nrn_grp.create_group('state')
                state_dict = nrn.state_dict()
                for key, tensor in state_dict.items():
                    # 跳过空张量
                    if tensor.numel() == 0:
                        continue
                    safe_key = _sanitize_key(key)
                    state_grp.create_dataset(safe_key, data=tensor.cpu().numpy())

            # ---- 突触连接 ----
            syns_grp = h5f.create_group('/synapses')
            for i, conn in enumerate(network.connections):
                conn_grp = syns_grp.create_group(f'conn_{i:04d}')
                syn = conn.synapse
                cls_name = type(syn).__name__
                conn_grp.attrs['type'] = cls_name
                # 配置
                config_keys = SYNAPSE_CONFIG_KEYS.get(cls_name)
                if config_keys is None:
                    config_keys = [k for k, v in vars(syn).items()
                                   if not isinstance(v, torch.Tensor) and not callable(v) and not k.startswith('_')]
                config = _get_object_config(syn, config_keys)
                conn_grp.create_dataset('config', data=json.dumps(config).encode('utf-8'))
                # state_dict
                state_grp = conn_grp.create_group('state')
                state_dict = syn.state_dict()
                for key, tensor in state_dict.items():
                    if tensor.numel() == 0:
                        continue
                    safe_key = _sanitize_key(key)
                    state_grp.create_dataset(safe_key, data=tensor.cpu().numpy())
                # 连接索引
                conn_grp.create_dataset('pre_indices', data=conn.pre_indices.cpu().numpy())
                conn_grp.create_dataset('post_indices', data=conn.post_indices.cpu().numpy())
                if conn.name:
                    conn_grp.attrs['name'] = conn.name

            # ---- Glial 网络 ----
            if network.glia is not None:
                glia_grp = h5f.create_group('/glia')
                glial = network.glia
                # 全局配置
                glia_grp.attrs['gap_junction_radius'] = glial.gap_junction_radius
                glia_grp.attrs['diffusion_rate'] = glial.diffusion_rate
                glia_grp.attrs['time_scale'] = glial.dt  # dt 存储为 time_scale

                # Astrocyte 组
                astro_grp = glia_grp.create_group('astrocytes')
                # 为快速索引，建立 synapse 对象到 connection 索引的映射
                syn_to_idx = {id(conn.synapse): idx for idx, conn in enumerate(network.connections)}
                for i, astro in enumerate(glial.astrocytes):
                    ast_grp = astro_grp.create_group(f'astro_{i:04d}')
                    ast_grp.attrs['id'] = astro.id
                    # 位置：存储为逗号分隔字符串或列表
                    ast_grp.create_dataset('position', data=np.array(astro.position))
                    ast_grp.attrs['ca'] = astro.ca
                    ast_grp.attrs['ca_tau'] = astro.ca_tau
                    ast_grp.attrs['ca_beta'] = astro.ca_beta
                    ast_grp.attrs['signal_threshold'] = astro.signal_threshold
                    ast_grp.attrs['neuron_mod_gain'] = astro.neuron_mod_gain
                    ast_grp.attrs['synapse_mod_gain'] = astro.synapse_mod_gain
                    # 关联的神经元：保存 (全局神经元索引, 局部索引列表) 对
                    neuron_refs_data = []
                    for ref in astro.neuron_refs:
                        nrn_obj = ref['obj']
                        try:
                            nrn_idx = network.neurons.index(nrn_obj)
                        except ValueError:
                            nrn_idx = -1  # 未找到，容错
                        indices_list = ref['indices'].cpu().tolist()
                        neuron_refs_data.append([nrn_idx, indices_list])
                    ast_grp.create_dataset('neuron_refs', data=json.dumps(neuron_refs_data).encode('utf-8'))
                    # 关联的突触
                    synapse_refs_data = []
                    for ref in astro.synapse_refs:
                        syn_obj = ref['obj']
                        syn_idx = syn_to_idx.get(id(syn_obj), -1)
                        indices_list = ref['indices'].cpu().tolist()
                        synapse_refs_data.append([syn_idx, indices_list])
                    ast_grp.create_dataset('synapse_refs', data=json.dumps(synapse_refs_data).encode('utf-8'))

                # Microglia 组
                micro_grp = glia_grp.create_group('microglias')
                for i, micro in enumerate(glial.microglias):
                    mic_grp = micro_grp.create_group(f'micro_{i:04d}')
                    # 找到关联的突触索引
                    syn_idx = syn_to_idx.get(id(micro.synapse), -1)
                    mic_grp.attrs['synapse_idx'] = syn_idx
                    mic_grp.attrs['smoothing'] = micro.smoothing
                    mic_grp.attrs['pruning_threshold'] = micro.pruning_threshold
                    mic_grp.attrs['protection_threshold'] = micro.protection_threshold
                    mic_grp.attrs['check_interval'] = micro.check_interval
                    mic_grp.attrs['step_counter'] = micro.step_counter
                    # 状态缓冲区 activity_avg
                    state_grp = mic_grp.create_group('state')
                    if micro.activity_avg.numel() > 0:
                        state_grp.create_dataset('activity_avg', data=micro.activity_avg.cpu().numpy())

            # ---- 调制场系统 ----
            if network.modulators is not None:
                mod_grp = h5f.create_group('/modulators')
                mod_sys = network.modulators
                for name, mod in mod_sys.modulators.items():
                    mod_name_grp = mod_grp.create_group(_sanitize_key(name))
                    mod_name_grp.attrs['type'] = type(mod).__name__
                    mod_name_grp.attrs['name'] = name
                    mod_name_grp.attrs['diffusion_radius'] = mod.diffusion_radius
                    mod_name_grp.attrs['half_life'] = mod.half_life
                    # 位置
                    mod_name_grp.create_dataset('release_position', data=np.array(mod.release_position))
                    # 源列表：转换为可序列化的列表
                    sources_list = [(list(pos), strength) for pos, strength in mod.sources]
                    mod_name_grp.create_dataset('sources', data=json.dumps(sources_list).encode('utf-8'))
            else:
                h5f.create_group('/modulators')  # 空组

        print(f"网络状态已保存至 {filepath}")


# ============================================================================
# 加载器
# ============================================================================

class LifeLoader:
    """.life 文件加载器。

    从 .life 文件重建完整的 GenesisNetwork，包含完整性校验（基于快照完整性哈希）
    和唯一的克隆检测接口。

    加载策略：
    1. 优先使用文件中的 config 构造实例。
    2. 若 config 中缺失某些构造参数，则尝试从 state_dict 中提取对应标量值进行补充。
    3. 最后调用 load_state_dict 恢复完整的可学习参数与缓冲区状态。
    该策略确保即使构造参数列表发生变化，仍能精确复现保存瞬间的动力学。
    """

    @staticmethod
    def load(
            filepath: Union[str, Path],
            device: Optional[torch.device] = None,
            unique_checker: Optional[Callable[[str], bool]] = None,
    ) -> GenesisNetwork:
        """
        从 .life 文件加载网络，执行完整性校验并返回 GenesisNetwork。

        参数:
            filepath: 输入 .life 文件的路径。
            device: 恢复张量的目标设备，默认为 CPU。支持跨硬件迁移。
            unique_checker: 可选的回调函数，接收 evolution_hash，若返回 False
                           则视为重复实例并抛出 IntegrityError（实现防克隆）。
        返回:
            重建的 GenesisNetwork 实例，所有组件已加载完毕。
        异常:
            LifeFormatError: 文件缺少必要数据或格式不正确。
            IntegrityError: 演化历史哈希不匹配或唯一性检查失败。
            FileNotFoundError: 文件不存在。
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f".life 文件不存在: {filepath}")

        network = GenesisNetwork()
        device = device or torch.device('cpu')

        with h5py.File(filepath, 'r') as h5f:
            # ---- 读取元数据 ----
            if '/meta' not in h5f:
                raise LifeFormatError("缺少 /meta 组")
            meta_grp = h5f['/meta']
            meta = {}
            for key in meta_grp.attrs.keys():
                meta[key] = meta_grp.attrs[key]
            network.meta = meta
            expected_evolution_hash = meta.get('evolution_hash')
            if not expected_evolution_hash:
                raise LifeFormatError("未找到 evolution_hash")

            # ---- 读取架构 ----
            if '/architecture' in h5f:
                arch_grp = h5f['/architecture']
                if 'definition' in arch_grp:
                    arch_json = bytes(arch_grp['definition'][()]).decode('utf-8')
                    network.architecture = json.loads(arch_json)

            # ---- 读取神经元 ----
            neurons_grp = h5f.get('/neurons')
            if neurons_grp is None:
                raise LifeFormatError("缺少 /neurons 组")
            neuron_list = []
            # 按名称排序以确保顺序
            sorted_nrn_keys = sorted(neurons_grp.keys())
            for nrn_key in sorted_nrn_keys:
                nrn_grp = neurons_grp[nrn_key]
                cls_name = nrn_grp.attrs['type']
                config_buf = bytes(nrn_grp['config'][()]).decode('utf-8')
                config = json.loads(config_buf)
                # 预先加载 state_dict（用于补充构造参数，同时待后续 load_state_dict 使用）
                state_dict = {}
                if 'state' in nrn_grp:
                    state_grp = nrn_grp['state']
                    for key in state_grp.keys():
                        tensor_data = state_grp[key][()]
                        if isinstance(tensor_data, np.ndarray):
                            tensor = torch.from_numpy(tensor_data).to(device)
                        else:
                            tensor = torch.tensor(tensor_data, device=device)
                        state_dict[key] = tensor
                # 创建神经元实例，传入 state_dict 以补充可能缺失的构造参数
                neuron = LifeLoader._create_neuron(cls_name, config, device, state_dict)
                # 应用完整的 state_dict 恢复所有可学习参数和缓冲区
                if state_dict:
                    neuron.load_state_dict(state_dict, strict=False)
                neuron_list.append(neuron)
            network.neurons = neuron_list

            # ---- 读取突触连接 ----
            syns_grp = h5f.get('/synapses')
            if syns_grp is None:
                raise LifeFormatError("缺少 /synapses 组")
            connection_list = []
            sorted_syn_keys = sorted(syns_grp.keys())
            for syn_key in sorted_syn_keys:
                conn_grp = syns_grp[syn_key]
                cls_name = conn_grp.attrs['type']
                config_buf = bytes(conn_grp['config'][()]).decode('utf-8')
                config = json.loads(config_buf)
                # 预先加载 state_dict
                state_dict = {}
                if 'state' in conn_grp:
                    state_grp = conn_grp['state']
                    for key in state_grp.keys():
                        tensor_data = state_grp[key][()]
                        if isinstance(tensor_data, np.ndarray):
                            tensor = torch.from_numpy(tensor_data).to(device)
                        else:
                            tensor = torch.tensor(tensor_data, device=device)
                        state_dict[key] = tensor
                # 创建突触实例，传入 state_dict 以补充可能缺失的构造参数
                synapse = LifeLoader._create_synapse(cls_name, config, device, state_dict)
                # 应用完整的 state_dict
                if state_dict:
                    synapse.load_state_dict(state_dict, strict=False)
                # 读取索引
                pre_indices = torch.from_numpy(conn_grp['pre_indices'][()]).to(device)
                post_indices = torch.from_numpy(conn_grp['post_indices'][()]).to(device)
                conn_name = conn_grp.attrs.get('name', None)
                connection = SynapticConnection(
                    synapse=synapse,
                    pre_indices=pre_indices,
                    post_indices=post_indices,
                    name=conn_name,
                )
                connection_list.append(connection)
            network.connections = connection_list

            # ---- 读取 Glial 网络 ----
            glia_grp = h5f.get('/glia')
            if glia_grp is not None and len(glia_grp.keys()) > 0:
                # 全局配置
                gap_junction_radius = glia_grp.attrs.get('gap_junction_radius', 0.3)
                diffusion_rate = glia_grp.attrs.get('diffusion_rate', 0.01)
                time_scale = glia_grp.attrs.get('time_scale', 1.0)

                # 重建 Astrocyte 实例（先用空关联，稍后填充）
                astro_list = []
                if 'astrocytes' in glia_grp:
                    astro_grp = glia_grp['astrocytes']
                    sorted_astro_keys = sorted(astro_grp.keys())
                    # 首次遍历创建实例，不填充引用，因为可能交叉引用
                    astro_instances = []
                    for akey in sorted_astro_keys:
                        ag = astro_grp[akey]
                        astro = Astrocyte(
                            astrocyte_id=ag.attrs['id'],
                            neuron_groups=[],  # 稍后设置
                            synapse_groups=[],  # 稍后设置
                            position=tuple(ag['position'][()].tolist()),
                            ca_initial=ag.attrs['ca'],
                            ca_tau=ag.attrs['ca_tau'],
                            ca_beta=ag.attrs['ca_beta'],
                            signal_threshold=ag.attrs['signal_threshold'],
                            neuron_mod_gain=ag.attrs['neuron_mod_gain'],
                            synapse_mod_gain=ag.attrs['synapse_mod_gain'],
                        )
                        astro.ca = ag.attrs['ca']  # 确保状态值准确
                        # 存储原始关联数据
                        neuron_refs_raw = json.loads(bytes(ag['neuron_refs'][()]).decode('utf-8'))
                        synapse_refs_raw = json.loads(bytes(ag['synapse_refs'][()]).decode('utf-8'))
                        astro._neuron_refs_raw = neuron_refs_raw
                        astro._synapse_refs_raw = synapse_refs_raw
                        astro_instances.append(astro)
                    # 第二次遍历：根据索引重建引用
                    for astro in astro_instances:
                        # 神经元关联
                        neuron_groups = []
                        for (nrn_idx, indices) in astro._neuron_refs_raw:
                            if 0 <= nrn_idx < len(neuron_list):
                                nrn_obj = neuron_list[nrn_idx]
                                idx_tensor = torch.tensor(indices, dtype=torch.long)
                                # 重新注册 Astrocyte 关联
                                for idx in indices:
                                    nrn_obj.register_astrocyte(astro.id)
                                # 提取原始阈值等（与 Astrocyte.__init__ 中逻辑一致）
                                orig_threshold = nrn_obj.threshold.data[idx_tensor].clone()
                                orig_rest_h = nrn_obj.rest_h.data[idx_tensor].clone()
                                neuron_groups.append({
                                    'obj': nrn_obj,
                                    'indices': idx_tensor,
                                    'orig_threshold': orig_threshold,
                                    'orig_rest_h': orig_rest_h,
                                })
                        astro.neuron_refs = neuron_groups
                        # 突触关联
                        synapse_groups = []
                        for (syn_idx, indices) in astro._synapse_refs_raw:
                            if 0 <= syn_idx < len(connection_list):
                                syn_obj = connection_list[syn_idx].synapse
                                idx_tensor = torch.tensor(indices, dtype=torch.long)
                                synapse_groups.append({
                                    'obj': syn_obj,
                                    'indices': idx_tensor,
                                })
                        astro.synapse_refs = synapse_groups
                        # 清理临时属性
                        del astro._neuron_refs_raw
                        del astro._synapse_refs_raw
                    astro_list = astro_instances

                # 重建 Microglia 实例
                micro_list = []
                if 'microglias' in glia_grp:
                    micro_grp = glia_grp['microglias']
                    sorted_micro_keys = sorted(micro_grp.keys())
                    for mkey in sorted_micro_keys:
                        mg = micro_grp[mkey]
                        syn_idx = mg.attrs['synapse_idx']
                        if 0 <= syn_idx < len(connection_list):
                            target_synapse = connection_list[syn_idx].synapse
                        else:
                            raise LifeFormatError(f"Microglia 引用了无效的突触索引 {syn_idx}")
                        micro = Microglia(
                            synapse=target_synapse,
                            activity_smoothing=mg.attrs['smoothing'],
                            pruning_threshold=mg.attrs['pruning_threshold'],
                            protection_threshold=mg.attrs['protection_threshold'],
                            check_interval=mg.attrs['check_interval'],
                        )
                        micro.step_counter = mg.attrs['step_counter']
                        # 加载状态
                        if 'state' in mg and 'activity_avg' in mg['state']:
                            micro.activity_avg = torch.from_numpy(
                                mg['state']['activity_avg'][()]
                            ).to(device)
                        micro_list.append(micro)

                # 构建 Glial 网络
                glial_net = GlialNetwork(
                    astrocytes=astro_list,
                    microglias=micro_list,
                    gap_junction_radius=gap_junction_radius,
                    diffusion_rate=diffusion_rate,
                    time_scale=time_scale,
                )
                network.glia = glial_net

            # ---- 读取调制场系统 ----
            mod_grp = h5f.get('/modulators')
            if mod_grp is not None and len(mod_grp.keys()) > 0:
                mod_sys = ModulatorSystem()
                for mod_key in mod_grp.keys():
                    mg = mod_grp[mod_key]
                    mod_type = mg.attrs['type']
                    mod_name = mg.attrs['name']
                    half_life = mg.attrs['half_life']
                    diffusion_radius = mg.attrs['diffusion_radius']
                    position = tuple(mg['release_position'][()].tolist())
                    # 创建对应子类实例
                    mod_class = MODULATOR_CLASS_MAP.get(mod_type)
                    if mod_class is None:
                        # 回退：通用 Modulator 基类（不可直接实例化，尝试）
                        # 使用基类创建需要 name，但基类 compute_release 未实现，
                        # 我们仅恢复状态，实际行为可能缺失。
                        # 更安全的做法是跳过未知类型并警告
                        warnings.warn(f"未知的调制原子类 '{mod_type}'，将使用 Modulator 基类并以空源恢复。")
                        modulator = Modulator(name=mod_name, half_life=half_life, diffusion_radius=diffusion_radius)
                    else:
                        modulator = mod_class(half_life=half_life, diffusion_radius=diffusion_radius)
                        # 构造时 name 被自动设置为类常量，但我们需要覆盖为保存的值
                        if hasattr(modulator, 'name'):
                            setattr(modulator, 'name', mod_name)
                    modulator.release_position = position
                    # 恢复源列表
                    sources_raw = json.loads(bytes(mg['sources'][()]).decode('utf-8'))
                    sources = [(tuple(pos), strength) for pos, strength in sources_raw]
                    modulator.sources = sources
                    mod_sys.add_modulator(modulator)
                network.modulators = mod_sys

            # ---- 完整性校验（基于快照完整性哈希） ----
            integrity_hash_loaded = meta.get('integrity_hash')
            if integrity_hash_loaded is None:
                raise LifeFormatError("快照完整性哈希 integrity_hash 缺失")
            computed_integrity = _compute_snapshot_integrity_hash(network.neurons, network.connections)
            if computed_integrity != integrity_hash_loaded:
                raise IntegrityError(
                    f"快照完整性哈希不匹配：文件声明 {integrity_hash_loaded}，实际计算 {computed_integrity}。"
                    "文件可能已被篡改或损坏。"
                )
            # 可选的唯一性检查
            if unique_checker is not None:
                if not unique_checker(expected_evolution_hash):
                    raise IntegrityError(
                        f"演化历史哈希 {expected_evolution_hash} 已被生态中的另一个实例占用，"
                        "拒绝加载以防止克隆。"
                    )

        return network

    @staticmethod
    def _create_neuron(cls_name: str, config: Dict[str, Any], device: torch.device,
                       state_dict: Optional[Dict[str, torch.Tensor]] = None) -> MorphoNeuron:
        """根据类型名和配置字典创建神经元实例。

        优先使用 config 中的参数构造实例；若 config 中缺失某些关键构造参数，
        则尝试从 state_dict 中提取对应的标量值进行补充，以确保构造参数与保存时的动力学参数一致。
        最后通过 load_state_dict 恢复完整状态。
        """
        if 'num_neurons' not in config:
            config['num_neurons'] = 1
        neuron_type_map = genesis_core.get_neuron_type_map()
        if cls_name not in neuron_type_map:
            raise LifeFormatError(f"未知的神经元类型: {cls_name}")
        neuron_cls = neuron_type_map[cls_name]
        init_params = inspect.signature(neuron_cls.__init__).parameters
        # 如果提供了 state_dict，尝试补充 config 中缺失的构造参数
        if state_dict is not None:
            for pname in init_params:
                if pname == 'self':
                    continue
                if pname not in config:
                    # 在 state_dict 中查找同名的键（通常为顶层参数，不包含点号）
                    if pname in state_dict:
                        val = state_dict[pname]
                        if isinstance(val, torch.Tensor) and val.numel() == 1:
                            config[pname] = val.item()
                        elif isinstance(val, torch.Tensor):
                            # 非标量张量不适合作为构造参数，忽略
                            pass
                        else:
                            config[pname] = val
                    # 若 state_dict 中也没有，则使用类的默认值（不修改 config）
        # 过滤掉不能识别的参数，避免 TypeError
        valid_config = {k: v for k, v in config.items() if k in init_params}
        return neuron_cls(**valid_config).to(device)

    @staticmethod
    def _create_synapse(cls_name: str, config: Dict[str, Any], device: torch.device,
                        state_dict: Optional[Dict[str, torch.Tensor]] = None) -> ResonanceSynapse:
        """根据类型名和配置字典创建突触实例。

        优先使用 config 中的参数构造实例；若 config 中缺失某些关键构造参数，
        则尝试从 state_dict 中提取对应的标量值进行补充，以确保构造参数与保存时的动力学参数一致。
        最后通过 load_state_dict 恢复完整状态。
        """
        if cls_name not in SYNAPSE_TYPE_MAP:
            raise LifeFormatError(f"未知的突触类型: {cls_name}")
        syn_cls = SYNAPSE_TYPE_MAP[cls_name]
        init_params = inspect.signature(syn_cls.__init__).parameters
        # 如果提供了 state_dict，尝试补充 config 中缺失的构造参数
        if state_dict is not None:
            for pname in init_params:
                if pname == 'self':
                    continue
                if pname not in config:
                    if pname in state_dict:
                        val = state_dict[pname]
                        if isinstance(val, torch.Tensor) and val.numel() == 1:
                            config[pname] = val.item()
                        elif isinstance(val, torch.Tensor):
                            pass
                        else:
                            config[pname] = val
        valid_config = {k: v for k, v in config.items() if k in init_params}
        return syn_cls(**valid_config).to(device)


# 突触类型映射（用于加载）
SYNAPSE_TYPE_MAP: Dict[str, Type[ResonanceSynapse]] = {
    'ResonanceSynapse': ResonanceSynapse,
    'PlasticityRule': PlasticityRule,
    'STDPSynapse': STDPSynapse,
}


# ============================================================================
# 生命文件注册表（防克隆唯一性辅助）
# ============================================================================

class LifeRegistry:
    """生态内生命文件演化历史哈希注册表。

    提供内存中的哈希集合，用于检查某个演化历史哈希是否已存在，
    以辅助 .life 文件的防克隆机制。
    """

    def __init__(self):
        self._hashes: set = set()

    def register(self, evolution_hash: str) -> bool:
        """注册一个新的演化历史哈希。

        返回:
            True 如果哈希未重复并成功注册；False 如果已存在（潜在克隆）。
        """
        if evolution_hash in self._hashes:
            return False
        self._hashes.add(evolution_hash)
        return True

    def is_registered(self, evolution_hash: str) -> bool:
        """检查哈希是否已注册。"""
        return evolution_hash in self._hashes

    def clear(self) -> None:
        """清空注册表。"""
        self._hashes.clear()


# ============================================================================
# 自包含测试（可选）
# ============================================================================
if __name__ == "__main__":
    # 本测试用于验证 life.py 模块的基本功能，依赖 Genesis 其他模块。
    # 运行前请确保 genesis_core, neuron, synapse, glia, modulator 在同一目录。
    print("Life 模块自包含测试...")
    import os

    save_path = "test.life"
    try:
        # 创建简单网络
        network = GenesisNetwork()
        # 创建一层 LIF 神经元
        lif = LIFNeuron(num_neurons=5, num_dendrites=1)
        network.neurons.append(lif)
        # 创建一个突触连接（自连接示例）
        pre = torch.arange(5, dtype=torch.long)
        post = torch.arange(5, dtype=torch.long)
        syn = STDPSynapse(num_synapses=5)
        conn = SynapticConnection(synapse=syn, pre_indices=pre, post_indices=post, name="test")
        network.connections.append(conn)
        # 创建调制系统
        ms = ModulatorSystem()
        reward = RewardSignal()
        ms.add_modulator(reward)
        network.modulators = ms
        # 创建 Glial 网络占位（无 glia）
        # 保存
        LifeSaver.save(network, save_path, creator="Tester")
        # 加载
        loaded = LifeLoader.load(save_path, device=torch.device('cpu'))
        print("加载成功，神经元数量:", len(loaded.neurons))
        print("加载成功，连接数量:", len(loaded.connections))
        # 验证状态一致性
        orig_state = lif.get_internal_state()
        load_state = loaded.neurons[0].get_internal_state()
        for k, v in orig_state.items():
            diff = (v - load_state[k]).abs().max().item()
            print(f"状态 {k} 最大差异: {diff:.6f}")
        print("测试通过。")
    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)