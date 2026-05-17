# visualization/pipeline.py
"""后端数据管道：计算集群同步相位指数、瞬时发放率、突触强度热力图、长期记忆索引等。

利用 Genesis 的 Monitor 数据，提供统一的数据查询与计算接口，供本地服务与 GUI 使用。
所有计算方法遵循术语中立宪章，内部命名不涉及人类生物化学或情感词汇。
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from collections import deque
import math

# 宽松导入 Monitor 类，避免循环依赖
try:
    from monitor import SpikeMonitor, WeightMonitor, MembraneMonitor, PlasticityMonitor
except ImportError:
    # 允许独立加载管道而不依赖完整框架
    SpikeMonitor = None
    WeightMonitor = None
    MembraneMonitor = None
    PlasticityMonitor = None


class GenomeDataPipeline:
    """数据管道，从 Genesis 监视器提取原始数据并计算高级可视化指标。

    设计为与监视器绑定，支持滑动窗口、实时更新。所有指标均源于内部状态，
    符合公理四（存在先于任务）和公理五（价值内生）。
    """

    def __init__(
        self,
        spike_monitors: Optional[List[SpikeMonitor]] = None,
        weight_monitors: Optional[List[WeightMonitor]] = None,
        membrane_monitors: Optional[List[MembraneMonitor]] = None,
        plasticity_monitors: Optional[List[PlasticityMonitor]] = None,
        window_size: int = 200,          # 滑动窗口大小（时间步）
        update_interval: int = 5,        # 管道更新间隔（步）
    ) -> None:
        """
        参数:
            spike_monitors: 脉冲监视器列表。
            weight_monitors: 突触强度监视器列表。
            membrane_monitors: 膜电位监视器列表。
            plasticity_monitors: 可塑性事件监视器列表。
            window_size: 用于计算瞬时发放率等指标的滑动窗口大小（时间步）。
            update_interval: 管道内部更新的最小步数间隔。
        """
        self.spike_monitors = spike_monitors or []
        self.weight_monitors = weight_monitors or []
        self.membrane_monitors = membrane_monitors or []
        self.plasticity_monitors = plasticity_monitors or []
        self.window_size = window_size
        self.update_interval = update_interval

        # 用于滑动窗口的缓存
        self._spike_buffer: deque = deque(maxlen=window_size)
        self._step_counter: int = 0

        # 预计算集群索引映射（由外部通过 define_assemblies 注入）
        self._assemblies: Dict[str, List[int]] = {}
        self._assembly_labels: List[str] = []

    def define_assemblies(self, assemblies: Dict[str, List[int]]) -> None:
        """定义神经元集群（Assembly）映射。

        参数:
            assemblies: 键为集群名称，值为该集群包含的全局神经元索引列表。
        """
        self._assemblies = assemblies.copy()
        self._assembly_labels = list(assemblies.keys())

    def step(self) -> None:
        """执行一次管道更新：从监视器拉取最新数据并更新内部缓存。"""
        # 收集所有脉冲监视器的最新栅格数据（假设每个监视器一个样本）
        all_spikes = []
        for mon in self.spike_monitors:
            if mon.spike_records:  # 获取最后一步的脉冲
                latest = mon.spike_records[-1]
                # 若为稀疏存储，转换为密集；这里假设密集（0/1 张量）
                if isinstance(latest, torch.Tensor):
                    all_spikes.append(latest.numpy())
                elif isinstance(latest, np.ndarray):
                    all_spikes.append(latest)
        if all_spikes:
            # 拼接不同监视器的神经元（假设它们对应于不同神经元群）
            combined = np.concatenate(all_spikes)  # 形状: (total_N,)
            self._spike_buffer.append(combined)

        self._step_counter += 1

    # ------------------------------------------------------------------
    # 瞬时发放率
    # ------------------------------------------------------------------
    def get_instant_firing_rate(self, smooth_sigma: float = 4.0) -> np.ndarray:
        """计算每个神经元的瞬时发放率（滑动窗口内的平均发放率）。

        返回:
            np.ndarray: 形状 (num_neurons,) 的当前发放率向量。
        """
        if not self._spike_buffer:
            return np.array([])
        # 取最近 window_size 步的数据
        recent = np.array(self._spike_buffer)  # (T, N)
        if recent.ndim == 1:
            recent = recent.reshape(1, -1)
        # 加简单的高斯窗平滑（按需）
        if smooth_sigma > 0 and recent.shape[0] > 1:
            from scipy.ndimage import gaussian_filter1d
            # 对时间轴应用高斯平滑（每列独立）
            rates = gaussian_filter1d(recent.astype(np.float32), sigma=smooth_sigma, axis=0)
        else:
            rates = recent.mean(axis=0)
        return rates.mean(axis=0) if rates.ndim == 2 else rates

    # ------------------------------------------------------------------
    # 集群同步相位指数（基于脉冲时序的相位锁定值）
    # ------------------------------------------------------------------
    def compute_phase_sync_matrix(self) -> np.ndarray:
        """计算所有已定义集群之间的相位同步指数（Phase Synchronization Index）。

        采用基于脉冲时间序列的互相关峰值归一化方法。
        若无脉冲监视器或集群定义，返回单位矩阵。

        返回:
            np.ndarray: 形状 (n_assemblies, n_assemblies) 的对称同步矩阵，值域 [0,1]。
        """
        n = len(self._assembly_labels)
        if n == 0:
            return np.eye(1)

        # 获取所有神经元的脉冲栅格矩阵 (T, total_N)
        if not self._spike_buffer:
            return np.eye(n)
        raster = np.array(self._spike_buffer)  # (T, N)
        if raster.ndim == 1:
            raster = raster.reshape(1, -1)

        # 为每个集群提取子矩阵
        cluster_rasters = {}
        for label, indices in self._assemblies.items():
            if len(indices) == 0:
                cluster_rasters[label] = np.zeros((raster.shape[0], 0))
            else:
                valid_indices = [i for i in indices if i < raster.shape[1]]
                cluster_rasters[label] = raster[:, valid_indices]

        # 计算集群平均发放率时间序列
        cluster_curves = []
        labels = self._assembly_labels
        for label in labels:
            mat = cluster_rasters.get(label)
            if mat is None or mat.shape[1] == 0:
                cluster_curves.append(np.zeros(raster.shape[0]))
            else:
                cluster_curves.append(mat.mean(axis=1))

        # 计算互相关（带归一化）
        sync_mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                x = cluster_curves[i]
                y = cluster_curves[j]
                # 互相关峰值归一化（[-1,1] 映射到 [0,1]）
                x_norm = x - x.mean()
                y_norm = y - y.mean()
                x_std = x_norm.std()
                y_std = y_norm.std()
                if x_std == 0 or y_std == 0:
                    cc = 0.5  # 中性
                else:
                    cc = np.correlate(x_norm, y_norm, mode='full')
                    max_cc = np.max(np.abs(cc)) / (len(x) * x_std * y_std)
                    cc = 0.5 * (max_cc + 1.0)  # 映射到 [0,1]
                sync_mat[i, j] = sync_mat[j, i] = cc

        return sync_mat

    # ------------------------------------------------------------------
    # 突触强度热力图
    # ------------------------------------------------------------------
    def get_synaptic_strength_map(self, matrix_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """返回当前所有突触的耦合强度矩阵（用于热力图）。

        若提供 matrix_shape (pre_neurons, post_neurons)，尝试将突触强度按索引映射到二维矩阵。
        否则返回一维强度数组及对应的 pre/post 索引。

        返回:
            np.ndarray: 强度矩阵（若形状已知）或包含 (pre, post, strength) 的字典。
        """
        strengths = []
        pre_ids = []
        post_ids = []
        for mon in self.weight_monitors:
            if mon.strength_history:
                # 取最新一步的强度
                latest = mon.strength_history[-1]  # (num_synapses,) 或类似
                if latest.ndim == 1:
                    # 无批次维度
                    strengths.append(latest)
                else:
                    # 假设批次索引 0
                    strengths.append(latest[0])
                # 索引需要从监视器额外获取，这里假定监视器存储了 pre/post 索引
                # 若没有，回退为顺序索引
                if hasattr(mon, 'pre_indices') and hasattr(mon, 'post_indices'):
                    pre_ids.append(mon.pre_indices.numpy())
                    post_ids.append(mon.post_indices.numpy())
                else:
                    # 根据监视的 indices 推断
                    pre_ids.append(np.arange(len(latest)))
                    post_ids.append(np.zeros_like(latest))  # 占位

        if not strengths:
            return np.array([])

        all_strength = np.concatenate(strengths)
        return all_strength  # 简化返回，实际可返回 (pre, post, strength) 三元组

    # ------------------------------------------------------------------
    # 长期记忆索引
    # ------------------------------------------------------------------
    def get_memory_indices(self) -> Dict[str, Any]:
        """从可塑性监视器和重量监视器中提取长期记忆索引快照。

        长期记忆索引指代网络中显著的突触耦合模式及其交往记忆队列中的关键事件。
        返回可供 MemoryViewer 渲染的字典结构，包含记忆描述符和相似度矩阵。

        返回:
            Dict 包含:
                - 'snapshots': List[Dict]，每个快照包含 pattern_id, timestamp, representative_strength 等。
                - 'similarity_matrix': np.ndarray，记忆快照间的模式相似度。
        """
        snapshots = []
        # 利用可塑性监视器的事件作为记忆标记点
        for mon in self.plasticity_monitors:
            for event in mon.events:
                if event['type'] in ('potentiation', 'depression'):
                    snapshots.append({
                        'step': event['step'],
                        'synapse_id': event['synapse_id'],
                        'strength': event['new_strength'],
                    })
        # 去重、聚合（简化：直接返回）
        return {
            'snapshots': snapshots,
            'similarity_matrix': np.eye(len(snapshots)) if snapshots else np.eye(1),
        }

    # ------------------------------------------------------------------
    # 辅助：获取当前所有神经元的膜电位（用于涟漪视图）
    # ------------------------------------------------------------------
    def get_membrane_state(self) -> np.ndarray:
        """返回所有神经元的当前膜电位（从膜电位监视器）。"""
        pots = []
        for mon in self.membrane_monitors:
            if mon.membrane_potentials:
                latest = mon.membrane_potentials[-1]  # (tracked_N,) 或 (batch, N)
                pots.append(latest.ravel())
        if pots:
            return np.concatenate(pots)
        return np.array([])