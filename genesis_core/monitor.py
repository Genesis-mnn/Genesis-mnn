# monitor.py — Genesis 框架监视与可视化模块
# 实现非侵入式监听、观察和理解网络内部状态的监视器。
# 严格遵循白皮书六条核心公理，特别是公理一（内在性优先）和公理四（存在先于任务）。
#
# 核心组件：
#   - BaseMonitor：监视器基类，定义统一数据记录/导出/异常检测接口。
#   - SpikeMonitor：脉冲发放监视器，统计发放频率、ISI、导出栅格图。
#   - MembraneMonitor：膜电位监视器，通过 get_internal_state() 钩子追踪 u。
#   - WeightMonitor：突触权重监视器，通过 get_synaptic_state() 钩子追踪 strength。
#   - PlasticityMonitor：可塑性事件监视器，记录增强/修剪事件。
#   - ModulatorMonitor：调制场监视器，记录调制原浓度场空间分布。
#   - replay_membrane / replay_modulator_field：便捷回放与可视化函数。

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import json
import os

# 从核心模块导入必要的基类和钩子
from .genesis_core import MorphoNeuron, ResonanceSynapse
from .modulator import ModulatorSystem, Modulator


# ============================================================================
# 监视器基类
# ============================================================================

class BaseMonitor:
    """监视器基类，提供统一的数据记录、NaN/Inf 异常检测与数据导出接口。

    所有具体监视器必须继承本类并实现 `record` 方法。
    本类提供了 `check_nan_inf` 静态方法、以及统一的 `to_csv`、`to_h5` 和 `to_tensorboard` 骨架。
    """

    def __init__(self, name: str = "BaseMonitor") -> None:
        """
        参数:
            name: 监视器名称，用于标识和导出文件前缀。
        """
        self.name = name
        self._step: int = 0
        self._data: List[Any] = []          # 原始数据存储（子类按需使用）
        self._timestamps: List[int] = []    # 记录时间步

    @staticmethod
    def check_nan_inf(data: Union[torch.Tensor, np.ndarray]) -> bool:
        """检查数据张量或数组中是否包含 NaN 或 Inf。

        参数:
            data: 待检测的 PyTorch 张量或 NumPy 数组。

        返回:
            bool: 若包含 NaN 或 Inf 返回 True，否则 False。
        """
        if isinstance(data, torch.Tensor):
            has_nan = torch.isnan(data).any().item()
            has_inf = torch.isinf(data).any().item()
            return has_nan or has_inf
        elif isinstance(data, np.ndarray):
            has_nan = np.isnan(data).any()
            has_inf = np.isinf(data).any()
            return has_nan or has_inf
        return False

    def record(self, *args: Any, **kwargs: Any) -> None:
        """记录一步数据（由子类实现）。

        子类必须实现此方法，将当前仿真步的观察数据存入内部数据结构。
        若检测到 NaN/Inf，应发出警告。
        """
        raise NotImplementedError("子类必须实现 record 方法。")

    def to_csv(self, filepath: str) -> None:
        """将监视数据导出为 CSV 文件。子类可重写以自定义格式。"""
        raise NotImplementedError("子类必须实现 to_csv 方法。")

    def to_h5(self, filepath: str) -> None:
        """将监视数据导出为 HDF5 文件。子类可重写以自定义结构。"""
        try:
            import h5py
        except ImportError:
            raise ImportError("导出 HDF5 需要安装 h5py。请执行 pip install h5py")
        with h5py.File(filepath, 'w') as f:
            self._write_h5(f)

    def _write_h5(self, h5file: Any) -> None:
        """子类应重写此方法以写入自定义数据集。"""
        raise NotImplementedError("子类必须实现 _write_h5 方法。")

    def to_tensorboard(self, writer: Any, step: int, tag: Optional[str] = None) -> None:
        """将监视数据记录到 TensorBoard。子类可重写以记录自定义度量。

        参数:
            writer: torch.utils.tensorboard.SummaryWriter 实例。
            step: 全局步数。
            tag: 自定义标签前缀，默认使用监视器名称。
        """
        raise NotImplementedError("子类必须实现 to_tensorboard 方法。")


# ============================================================================
# 脉冲发放监视器
# ============================================================================

class SpikeMonitor(BaseMonitor):
    """脉冲输出监视器：统计神经元输出频率、脉冲间隔分布（ISI），并支持栅格图导出。

    在仿真循环中，每步调用 `record(spike)` 传入当前输出脉冲张量。
    可存储完整脉冲矩阵（默认）或稀疏索引以节省内存。

    公理四（存在先于任务）：即使在没有外部输入的条件下，本监视器也能记录自发的
    脉冲活动，不依赖外部任务信号。
    """

    def __init__(self,
                 num_neurons: int,
                 batch_index: int = 0,
                 name: str = "SpikeMonitor",
                 store_sparse: bool = False) -> None:
        """
        参数:
            num_neurons: 监视的神经元总数。
            batch_index: 从批次中提取哪个样本的数据（默认 0）。
            name: 监视器名称。
            store_sparse: 若为 True，则仅存储活跃神经元的索引（减少内存），
                          但会影响部分统计方法。
        """
        super().__init__(name)
        self.num_neurons = num_neurons
        self.batch_index = batch_index
        self.store_sparse = store_sparse

        # 存储每步的脉冲数据
        self.spike_records: List[Union[torch.Tensor, np.ndarray]] = []
        # 辅助统计：每步平均发放率（标量）
        self.firing_rates: List[float] = []

    def record(self, spike: torch.Tensor) -> None:
        """记录一步的脉冲输出情况。

        参数:
            spike: 形状为 (batch, num_neurons) 或 (num_neurons,) 的脉冲张量，
                   元素为 0 或 1。
        """
        # 提取目标样本
        if spike.dim() == 2:
            spike_sample = spike[self.batch_index].detach().cpu()
        else:
            spike_sample = spike.detach().cpu()

        # 异常检测
        if self.check_nan_inf(spike_sample):
            warnings.warn(
                f"{self.name} 第 {self._step} 步检测到 NaN/Inf。",
                RuntimeWarning
            )

        if self.store_sparse:
            # 存储活跃神经元的 numpy 索引数组
            active_indices = torch.nonzero(spike_sample).squeeze(-1).numpy()
            self.spike_records.append(active_indices)
        else:
            # 存储完整的二进制张量
            self.spike_records.append(spike_sample.clone())

        self.firing_rates.append(spike_sample.float().mean().item())
        self._timestamps.append(self._step)
        self._step += 1

    def get_spike_counts(self) -> np.ndarray:
        """返回每个神经元在整个记录期间的总发放次数（形状 (num_neurons,)）。"""
        if self.store_sparse:
            raise NotImplementedError("稀疏存储模式下暂不支持。")
        raster = torch.stack(self.spike_records, dim=0)  # (T, N)
        return raster.sum(dim=0).numpy()

    def get_firing_rates(self) -> np.ndarray:
        """返回每个神经元的平均输出率（形状 (num_neurons,)）。"""
        if self.store_sparse:
            raise NotImplementedError("稀疏存储模式下暂不支持。")
        raster = torch.stack(self.spike_records, dim=0).float()
        return raster.mean(dim=0).numpy()

    def get_isi(self, neuron_idx: int) -> List[float]:
        """计算指定神经元的脉冲间期（ISI）。

        参数:
            neuron_idx: 目标神经元索引。

        返回:
            List[float]: 按时间顺序排列的脉冲间期列表。
        """
        if self.store_sparse:
            raise NotImplementedError("稀疏存储模式下暂不支持。")
        spikes = torch.stack(self.spike_records, dim=0)[:, neuron_idx].numpy()
        spike_times = np.where(spikes > 0)[0]
        if len(spike_times) < 2:
            return []
        return np.diff(spike_times).tolist()

    def to_raster(self) -> np.ndarray:
        """返回栅格图数据矩阵，形状 (T, num_neurons)，元素为 0/1。"""
        if self.store_sparse:
            raise NotImplementedError("稀疏存储模式下暂不支持。")
        return torch.stack(self.spike_records, dim=0).numpy()

    def to_csv(self, filepath: str) -> None:
        """导出栅格图数据为 CSV 文件（每行一个时间步，每列一个神经元）。"""
        raster = self.to_raster()
        np.savetxt(filepath, raster, delimiter=',', fmt='%d')

    def _write_h5(self, h5file: Any) -> None:
        """写入 HDF5 数据集。"""
        raster = self.to_raster()
        h5file.create_dataset('raster', data=raster)
        h5file.create_dataset('firing_rates', data=np.array(self.firing_rates))

    def to_tensorboard(self, writer: Any, step: int, tag: Optional[str] = None) -> None:
        """记录平均发放率和栅格图的直方图。"""
        tag = tag or self.name
        if self.firing_rates:
            writer.add_scalar(f'{tag}/mean_firing_rate', np.mean(self.firing_rates), step)
        if not self.store_sparse and len(self.spike_records) > 10:
            raster = self.to_raster()
            writer.add_histogram(f'{tag}/spike_raster_flat', raster.flatten(), step)


# ============================================================================
# 膜电位监视器
# ============================================================================

class MembraneMonitor(BaseMonitor):
    """膜电位监视器：追踪指定神经元的膜电位轨迹 u。

    通过 `MorphoNeuron.get_internal_state()` 钩子非侵入式获取膜电位数据。
    满足公理一（内在性优先）：正确感知神经元的健康度和膜电位，反映稳态维持情况。
    """

    def __init__(self,
                 neuron: MorphoNeuron,
                 indices: Optional[List[int]] = None,
                 batch_index: int = 0,
                 name: str = "MembraneMonitor") -> None:
        """
        参数:
            neuron: 目标形态神经元实例。
            indices: 要追踪的神经元局部索引列表；默认为 None 表示全部神经元。
            batch_index: 批次样本索引（默认 0）。
            name: 监视器名称。
        """
        super().__init__(name)
        self.neuron = neuron
        self.batch_index = batch_index
        self.indices = indices if indices is not None else list(range(neuron.num_neurons))
        self.num_tracked = len(self.indices)

        # 存储每步膜电位 (num_tracked,) numpy 数组
        self.membrane_potentials: List[np.ndarray] = []

    def record(self) -> None:
        """通过 `get_internal_state()` 获取当前膜电位并记录。"""
        state = self.neuron.get_internal_state()
        u = state['u'][self.batch_index, self.indices].detach().cpu()

        if self.check_nan_inf(u):
            warnings.warn(
                f"{self.name} 第 {self._step} 步检测到 NaN/Inf。",
                RuntimeWarning
            )

        self.membrane_potentials.append(u.numpy())
        self._timestamps.append(self._step)
        self._step += 1

    def get_traces(self) -> np.ndarray:
        """返回完整的膜电位轨迹，形状 (T, num_tracked)。"""
        return np.array(self.membrane_potentials)

    def to_csv(self, filepath: str) -> None:
        """导出膜电位轨迹为 CSV 文件。"""
        traces = self.get_traces()
        np.savetxt(filepath, traces, delimiter=',', header='Columns correspond to neuron indices')

    def _write_h5(self, h5file: Any) -> None:
        """写入 HDF5 数据集。"""
        traces = self.get_traces()
        h5file.create_dataset('membrane_potential', data=traces)
        h5file.create_dataset('indices', data=np.array(self.indices))

    def to_tensorboard(self, writer: Any, step: int, tag: Optional[str] = None) -> None:
        """记录当前步膜电位的分布直方图。"""
        tag = tag or self.name
        if self.membrane_potentials:
            last_u = self.membrane_potentials[-1]
            writer.add_histogram(f'{tag}/u_distribution', last_u, step)


# ============================================================================
# 突触权重监视器
# ============================================================================

class WeightMonitor(BaseMonitor):
    """突触权重监视器：记录耦合强度 strength 和交往记忆统计。

    通过 `ResonanceSynapse.get_synaptic_state()` 钩子非侵入式获取数据。
    满足公理二（共鸣取代权重）：监视器能读取动态耦合强度，反映节律匹配程度。
    """

    def __init__(self,
                 synapse: ResonanceSynapse,
                 indices: Optional[List[int]] = None,
                 batch_index: int = 0,
                 name: str = "WeightMonitor") -> None:
        """
        参数:
            synapse: 目标共鸣突触实例。
            indices: 要监视的突触局部索引列表；默认为 None 表示全部。
            batch_index: 批次样本索引（默认 0）。
            name: 监视器名称。
        """
        super().__init__(name)
        self.synapse = synapse
        self.batch_index = batch_index
        self.indices = indices if indices is not None else list(range(synapse.num_synapses))
        self.num_tracked = len(self.indices)

        # 存储每步的 strength 快照
        self.strength_history: List[np.ndarray] = []
        # 存储每步的记忆统计量（平均记忆条目数）
        self.memory_stats: List[float] = []

    def record(self) -> None:
        """通过 `get_synaptic_state()` 钩子获取当前突触状态并记录。"""
        state = self.synapse.get_synaptic_state()
        strength = state['strength'][self.batch_index, self.indices].detach().cpu()

        if self.check_nan_inf(strength):
            warnings.warn(
                f"{self.name} 第 {self._step} 步检测到 NaN/Inf。",
                RuntimeWarning
            )

        self.strength_history.append(strength.numpy())

        # 交往记忆统计：记忆队列中 timestamp > 0 表示曾记录过强共鸣事件
        memory_buf = state['memory_buffer'][self.batch_index, self.indices].detach().cpu()
        mem_counts = (memory_buf[..., 0] > 0).sum(dim=-1).float().mean().item()
        self.memory_stats.append(mem_counts)

        self._timestamps.append(self._step)
        self._step += 1

    def get_strength_traces(self) -> np.ndarray:
        """返回强度轨迹，形状 (T, num_tracked)。"""
        return np.array(self.strength_history)

    def to_csv(self, filepath: str) -> None:
        """导出强度轨迹为 CSV 文件。"""
        strengths = self.get_strength_traces()
        np.savetxt(filepath, strengths, delimiter=',')

    def _write_h5(self, h5file: Any) -> None:
        """写入 HDF5 数据集。"""
        strengths = self.get_strength_traces()
        h5file.create_dataset('strength', data=strengths)
        h5file.create_dataset('memory_stats', data=np.array(self.memory_stats))

    def to_tensorboard(self, writer: Any, step: int, tag: Optional[str] = None) -> None:
        """记录强度分布直方图和平均记忆条目数。"""
        tag = tag or self.name
        if self.strength_history:
            writer.add_histogram(f'{tag}/strength_distribution', self.strength_history[-1], step)
        if self.memory_stats:
            writer.add_scalar(f'{tag}/mean_memory_entries', self.memory_stats[-1], step)


# ============================================================================
# 可塑性事件监视器
# ============================================================================

class PlasticityMonitor(BaseMonitor):
    """可塑性事件监视器：记录增强/修剪事件的时间戳、强度和位置。

    通过比较连续两步的突触状态检测变化：
        - 强度变化超过阈值 → 增强/减弱事件
        - prune_flag 从 0 变为 1 → 修剪标记事件

    满足公理三（演化优于训练）：监视器独立观测局部可塑性和结构演化事件，
    不依赖反向传播信号。
    """

    def __init__(self,
                 synapse: ResonanceSynapse,
                 indices: Optional[List[int]] = None,
                 batch_index: int = 0,
                 threshold_change: float = 0.01,
                 name: str = "PlasticityMonitor",
                 synapse_ids: Optional[List[int]] = None) -> None:
        """
        参数:
            synapse: 目标共鸣突触实例。
            indices: 监视的突触局部索引；默认为全部。
            batch_index: 批次样本索引。
            threshold_change: strength 变化超过此阈值时记录事件。
            name: 监视器名称。
            synapse_ids: 可选的全局突触 ID 映射，用于在事件中标注位置；
                         若为 None，则使用局部索引。
        """
        super().__init__(name)
        self.synapse = synapse
        self.batch_index = batch_index
        self.indices = indices if indices is not None else list(range(synapse.num_synapses))
        self.num_tracked = len(self.indices)
        self.threshold_change = threshold_change
        self.synapse_ids = synapse_ids if synapse_ids else list(range(self.num_tracked))

        # 保存上一步的 strength 和 prune_flag（numpy 数组）
        self._prev_strength: Optional[np.ndarray] = None
        self._prev_prune_flag: Optional[np.ndarray] = None

        # 事件列表：每个事件为字典
        self.events: List[Dict[str, Any]] = []

    def record(self) -> None:
        """检测可塑性事件并记录。"""
        state = self.synapse.get_synaptic_state()
        strength = state['strength'][self.batch_index, self.indices].detach().cpu().numpy()
        prune_flag = state['prune_flag'][self.batch_index, self.indices].detach().cpu().numpy().astype(np.int8)

        if self.check_nan_inf(torch.from_numpy(strength)):
            warnings.warn(
                f"{self.name} 第 {self._step} 步检测到 NaN/Inf。",
                RuntimeWarning
            )

        # ---- 强度变化事件 ----
        if self._prev_strength is not None:
            diff = strength - self._prev_strength
            significant = np.abs(diff) >= self.threshold_change
            for i in range(self.num_tracked):
                if significant[i]:
                    event_type = 'potentiation' if diff[i] > 0 else 'depression'
                    self.events.append({
                        'step': self._step,
                        'synapse_id': self.synapse_ids[i],
                        'type': event_type,
                        'change': float(diff[i]),
                        'new_strength': float(strength[i]),
                    })

        # ---- 修剪标记事件（prune_flag 上升沿） ----
        if self._prev_prune_flag is not None:
            new_prune = (prune_flag == 1) & (self._prev_prune_flag == 0)
            for i in range(self.num_tracked):
                if new_prune[i]:
                    self.events.append({
                        'step': self._step,
                        'synapse_id': self.synapse_ids[i],
                        'type': 'pruning_marked',
                        'change': 0.0,
                        'new_strength': float(strength[i]),
                    })

        # 更新缓存状态
        self._prev_strength = strength
        self._prev_prune_flag = prune_flag

        self._timestamps.append(self._step)
        self._step += 1

    def get_events_df(self) -> Any:
        """将事件列表转换为 pandas DataFrame（需要 pandas 已安装）。"""
        import pandas as pd
        return pd.DataFrame(self.events)

    def to_csv(self, filepath: str) -> None:
        """导出事件列表为 CSV。"""
        df = self.get_events_df()
        df.to_csv(filepath, index=False)

    def _write_h5(self, h5file: Any) -> None:
        """写入事件数据集。"""
        if not self.events:
            return
        steps = np.array([e['step'] for e in self.events])
        ids = np.array([e['synapse_id'] for e in self.events])
        types = np.array([e['type'].encode() for e in self.events])
        changes = np.array([e['change'] for e in self.events])
        h5file.create_dataset('event_steps', data=steps)
        h5file.create_dataset('event_ids', data=ids)
        h5file.create_dataset('event_types', data=types)
        h5file.create_dataset('event_changes', data=changes)

    def to_tensorboard(self, writer: Any, step: int, tag: Optional[str] = None) -> None:
        """记录最近事件的数量。"""
        tag = tag or self.name
        # 简单记录累计事件数（或最新步的事件数）
        recent_events = sum(1 for e in self.events if e['step'] == self._step - 1)
        writer.add_scalar(f'{tag}/events_per_step', recent_events, step)


# ============================================================================
# 调制场监视器
# ============================================================================

class ModulatorMonitor(BaseMonitor):
    """调制场监视器：记录调制原浓度场的空间分布数据。

    每步存储每个调制原的源列表（位置与强度），若提供 `spatial_grid`，
    则自动在该网格点上采样浓度并存储快照，便于后续可视化。
    满足公理五（价值内生）：调制原浓度源于内部稳态偏差，本监视器不引入外部指标。
    """

    def __init__(self,
                 modulator_system: ModulatorSystem,
                 name: str = "ModulatorMonitor",
                 spatial_grid: Optional[Dict[str, np.ndarray]] = None) -> None:
        """
        参数:
            modulator_system: 全局调制场管理器实例。
            name: 监视器名称。
            spatial_grid: 字典，键为调制原名称，值为采样点坐标数组 (N_points, dim)。
                          若提供，则自动记录这些点上的浓度；否则仅记录源数据。
        """
        super().__init__(name)
        self.system = modulator_system
        self.grid = spatial_grid or {}

        # 每步存储源列表的深拷贝（嵌套字典）
        self.source_history: List[Dict[str, List[Tuple[Tuple[float, ...], float]]]] = []
        # 若使用网格，则存储浓度快照
        self.concentration_history: List[Dict[str, np.ndarray]] = []

    def record(self) -> None:
        """记录当前所有调制原的源以及（可选的）网格浓度。"""
        step_sources: Dict[str, List[Tuple[Tuple[float, ...], float]]] = {}
        step_conc: Dict[str, np.ndarray] = {}

        for name, mod in self.system.modulators.items():
            # 深拷贝源列表（避免后续修改影响历史）
            src_copy = [(pos, strength) for pos, strength in mod.sources]
            step_sources[name] = src_copy

            # 如果配置了该调制原的采样网格，则计算浓度
            if name in self.grid:
                pts = self.grid[name]  # (N_pts, dim)
                conc = np.array([mod.query_concentration(tuple(pt)) for pt in pts])
                step_conc[name] = conc

        self.source_history.append(step_sources)
        if step_conc:
            self.concentration_history.append(step_conc)

        self._timestamps.append(self._step)
        self._step += 1

    def get_concentration_at_step(self, step_idx: int, modulator_name: str) -> Optional[np.ndarray]:
        """返回指定步和调制原的网格浓度数组（若存在）。"""
        if step_idx < 0 or step_idx >= len(self.concentration_history):
            return None
        return self.concentration_history[step_idx].get(modulator_name, None)

    def to_csv(self, filepath: str) -> None:
        """将第一个调制原的网格浓度数据导出为 CSV。"""
        if not self.concentration_history:
            raise ValueError("未提供 spatial_grid，无法导出浓度 CSV。请先设置网格。")
        # 取第一个调制原的浓度
        first_mod = list(self.concentration_history[0].keys())[0]
        conc_matrix = np.stack(
            [step[first_mod] for step in self.concentration_history], axis=0
        )  # (T, N_pts)
        np.savetxt(filepath, conc_matrix, delimiter=',')

    def _write_h5(self, h5file: Any) -> None:
        """写入源历史 JSON 和浓度快照。"""
        # 源历史序列化为 JSON（将元组转为列表）
        source_json = []
        for step_dict in self.source_history:
            step_serial = {}
            for mod_name, src_list in step_dict.items():
                step_serial[mod_name] = [(list(pos), strength) for pos, strength in src_list]
            source_json.append(step_serial)
        h5file.create_dataset('source_history_json', data=json.dumps(source_json, ensure_ascii=False))

        # 浓度历史
        for name in self.concentration_history[0].keys():
            concs = np.stack([step[name] for step in self.concentration_history], axis=0)
            h5file.create_dataset(f'concentration_{name}', data=concs)

    def to_tensorboard(self, writer: Any, step: int, tag: Optional[str] = None) -> None:
        """记录最新浓度分布直方图。"""
        tag = tag or self.name
        if self.concentration_history:
            for name in self.concentration_history[-1].keys():
                conc = self.concentration_history[-1][name]
                writer.add_histogram(f'{tag}/{name}_concentration', conc, step)


# ============================================================================
# 便捷回放与可视化函数
# ============================================================================

def replay_membrane(monitor: MembraneMonitor,
                    neuron_indices: Optional[List[int]] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    title: Optional[str] = None,
                    save_path: Optional[str] = None) -> None:
    """基于 MembraneMonitor 数据绘制膜电位轨迹。

    参数:
        monitor: 已记录的 MembraneMonitor 实例。
        neuron_indices: 要绘制的神经元索引列表；默认绘制前 5 个。
        figsize: 图像尺寸。
        title: 自定义标题。
        save_path: 若提供，图像保存到该路径而不是显示。

    公理一验证：可视化膜电位动态，反映神经元内在稳态维持情况。
    """
    import matplotlib.pyplot as plt
    traces = monitor.get_traces()
    if traces.size == 0:
        print("没有记录的膜电位数据。")
        return

    T, N = traces.shape
    if neuron_indices is None:
        neuron_indices = list(range(min(N, 5)))

    plt.figure(figsize=figsize)
    for idx in neuron_indices:
        if idx < N:
            plt.plot(range(T), traces[:, idx], label=f'Neuron {idx}')
    plt.xlabel('Time step')
    plt.ylabel('Membrane potential (u)')
    plt.title(title or f'Membrane Potential - {monitor.name}')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def replay_modulator_field(monitor: ModulatorMonitor,
                           modulator_name: str,
                           step_idx: int = -1,
                           grid_shape: Tuple[int, int] = (50, 50),
                           slice_axis: int = 2,
                           slice_index: Optional[float] = None,
                           figsize: Tuple[int, int] = (8, 6),
                           cmap: str = 'viridis',
                           title: Optional[str] = None,
                           save_path: Optional[str] = None) -> None:
    """可视化调制原浓度场的正交切片。

    从记录的调制原源数据重建浓度场，并在指定平面上绘制热力图。
    适用于空间维度为 3 的情况；对于低维空间自动回退为散点图。

    参数:
        monitor: 已记录的 ModulatorMonitor 实例。
        modulator_name: 要可视化的调制原名称。
        step_idx: 时间步索引；负值表示从末尾倒数（例如 -1 为最后一步）。
        grid_shape: 切片平面的采样网格尺寸。
        slice_axis: 固定的轴（0=X, 1=Y, 2=Z）。
        slice_index: 固定轴上的切片坐标；默认取源分布中心。
        figsize: 图像尺寸。
        cmap: 颜色映射。
        title: 自定义标题。
        save_path: 若提供，图像保存到该路径。

    公理四验证：可视化调制场在零外部输入下的自发扩散与分布。
    """
    import matplotlib.pyplot as plt
    if not monitor.source_history:
        print("没有记录的调制原源数据。")
        return

    # 解析时间步索引
    if step_idx < 0:
        step_idx = len(monitor.source_history) + step_idx
    if step_idx < 0 or step_idx >= len(monitor.source_history):
        raise IndexError(f"step_idx {step_idx} 超出范围 [0, {len(monitor.source_history)-1}]")

    step_sources = monitor.source_history[step_idx]
    if modulator_name not in step_sources:
        raise ValueError(f"调制原 '{modulator_name}' 未记录。可用: {list(step_sources.keys())}")

    sources = step_sources[modulator_name]
    if not sources:
        print(f"调制原 '{modulator_name}' 在步 {step_idx} 无活跃源。")
        return

    # 确定空间维度
    dim = len(sources[0][0])
    if dim < 3:
        # 低维空间：绘制散点图
        plt.figure(figsize=figsize)
        if dim == 1:
            xs = [pos[0] for pos, _ in sources]
            ys = [strength for _, strength in sources]
            plt.scatter(xs, ys, c=ys, cmap=cmap)
            plt.xlabel('Position')
            plt.ylabel('Source strength')
        elif dim == 2:
            xs = [pos[0] for pos, _ in sources]
            ys = [pos[1] for pos, _ in sources]
            strengths = [s for _, s in sources]
            plt.scatter(xs, ys, c=strengths, cmap=cmap)
            plt.colorbar(label='Strength')
            plt.xlabel('X')
            plt.ylabel('Y')
        plt.title(title or f'{modulator_name} sources (step {step_idx})')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        return

    # 3D 切片可视化
    axes_labels = ['X', 'Y', 'Z']
    axes_free = [i for i in range(3) if i != slice_axis]

    all_pos = np.array([pos for pos, _ in sources])
    min_pos = all_pos.min(axis=0) - 1.0
    max_pos = all_pos.max(axis=0) + 1.0

    if slice_index is None:
        # 默认取中心位置
        slice_index = float(np.mean(all_pos[:, slice_axis]))

    # 自由轴上的采样网格
    x_vals = np.linspace(min_pos[axes_free[0]], max_pos[axes_free[0]], grid_shape[0])
    y_vals = np.linspace(min_pos[axes_free[1]], max_pos[axes_free[1]], grid_shape[1])
    X, Y = np.meshgrid(x_vals, y_vals)

    # 构造查询点（固定轴填充 slice_index）
    pts = np.zeros((grid_shape[0] * grid_shape[1], 3))
    pts[:, axes_free[0]] = X.ravel()
    pts[:, axes_free[1]] = Y.ravel()
    pts[:, slice_axis] = slice_index

    # 从 ModulatorSystem 获取对应调制原实例并计算浓度
    modulator = monitor.system.modulators.get(modulator_name)
    if modulator is None:
        raise ValueError(f"调制系统未包含调制原 '{modulator_name}'。")

    concentrations = np.array([modulator.query_concentration(tuple(pt)) for pt in pts])
    Z = concentrations.reshape(grid_shape)

    plt.figure(figsize=figsize)
    plt.contourf(X, Y, Z, levels=50, cmap=cmap)
    plt.colorbar(label='Concentration')
    plt.xlabel(f'{axes_labels[axes_free[0]]} axis')
    plt.ylabel(f'{axes_labels[axes_free[1]]} axis')
    plt.title(title or f'{modulator_name} slice at {axes_labels[slice_axis]}={slice_index:.2f} (step {step_idx})')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ============================================================================
# 自包含测试（用于验证导入与基本功能）
# ============================================================================
if __name__ == "__main__":
    # 关闭 matplotlib 的交互模式，避免在无 GUI 环境中报错（使用 Agg 后端）
    import matplotlib
    matplotlib.use('Agg')

    print("Monitor 模块自包含演示开始...")

    # 创建简单组件（用于测试，不依赖完整网络）
    from .neuron import LIFNeuron
    from .synapse import STDPSynapse
    from .modulator import create_default_modulators
    from . import genesis_core  # 确保常量可用

    neurom = LIFNeuron(num_neurons=10, num_dendrites=1)
    syn = STDPSynapse(num_synapses=20)
    mod_sys = create_default_modulators()
    # 手动添加一些源
    reward_mod = mod_sys.modulators[genesis_core.MODULATOR_REWARD]
    reward_mod.release_position = (0., 0., 0.)
    reward_mod.sources.append(((0., 1., 2.), 1.0))

    # 初始化监视器
    spike_mon = SpikeMonitor(num_neurons=10)
    membrane_mon = MembraneMonitor(neurom, indices=[0, 1, 2])
    weight_mon = WeightMonitor(syn)
    plast_mon = PlasticityMonitor(syn, threshold_change=0.0)  # 低阈值以触发事件
    mod_mon = ModulatorMonitor(mod_sys,
                               spatial_grid={'奖赏信号': np.array([[0.,0.,0.],[1.,1.,1.]])})

    print("开始模拟 5 步...")
    for t in range(5):
        # 神经元前向
        fake_input = torch.randn(1, 1, 10)  # (batch, dendrites, neurons)
        spike_out = neurom(fake_input)
        spike_mon.record(spike_out)
        membrane_mon.record()

        # 突触前向
        pre_spk = torch.randint(0, 2, (1, 20)).float()
        post_spk = torch.randint(0, 2, (1, 20)).float()
        pre_r = torch.randn(1, 20, 1) * 0.1
        post_r = torch.randn(1, 20, 1) * 0.1
        syn(pre_spk, post_spk, pre_r, post_r,
            modulator_concentrations={genesis_core.MODULATOR_REWARD: 0.5})

        weight_mon.record()
        plast_mon.record()

        # 调制场更新
        mod_sys.step({genesis_core.GLOBAL_REWARD_ERROR: 0.2})
        mod_mon.record()

    print("模拟结束。")
    print(f"SpikeMonitor: 记录 {len(spike_mon.spike_records)} 步，平均输出率 {np.mean(spike_mon.firing_rates):.4f}")
    print(f"MembraneMonitor: 膜电位轨迹形状 {membrane_mon.get_traces().shape}")
    print(f"WeightMonitor: 强度轨迹形状 {weight_mon.get_strength_traces().shape}")
    print(f"PlasticityMonitor: 检测到 {len(plast_mon.events)} 个事件")
    print(f"ModulatorMonitor: 记录 {len(mod_mon.source_history)} 步源历史")

    # 尝试导出 CSV（临时文件）
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='_spike.csv', delete=False) as f:
        try:
            spike_mon.to_csv(f.name)
            print(f"Spike CSV 已保存至 {f.name}")
        except NotImplementedError:
            pass

    # 回放测试（保存图片到临时文件）
    try:
        replay_membrane(membrane_mon, neuron_indices=[0, 1], save_path='temp_membrane.png')
        print("膜电位回放图保存至 temp_membrane.png")
    except Exception as e:
        print(f"回放膜电位失败: {e}")

    try:
        replay_modulator_field(mod_mon, '奖赏信号', step_idx=-1,
                               grid_shape=(20, 20), slice_axis=2, slice_index=1.0,
                               save_path='temp_modulator.png')
        print("调制场回放图保存至 temp_modulator.png")
    except Exception as e:
        print(f"回放调制场失败: {e}")

    print("演示完成。生成的文件可用于验证。")