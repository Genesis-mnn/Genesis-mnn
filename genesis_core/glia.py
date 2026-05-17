# glia.py — Genesis 框架胶质细胞网络
# 实现星形胶质细胞、胶质互联网与小胶质细胞。
# 严格遵循白皮书六条核心公理，特别是公理一、三、四。
# 精细化项：Microglia 每连接点独立决策；Astrocyte 通过调制原接口反馈。

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import math
from .genesis_core import MorphoNeuron, ResonanceSynapse, MODULATOR_STRESS


class Astrocyte:
    """星形胶质细胞。

    包裹一小群神经元和连接点，通过活动感知器机制感知局部神经活动水平，
    利用状态波扩散和信号释放实现反馈调节，维护网络的整体内部稳态。
    对应白皮书3.1节，公理一、四。
    """

    def __init__(self,
                 astrocyte_id: int,
                 neuron_groups: List[Tuple[MorphoNeuron, List[int]]],
                 synapse_groups: List[Tuple[ResonanceSynapse, List[int]]],
                 position: Tuple[float, ...] = (0.0,),
                 ca_initial: float = 0.0,
                 ca_tau: float = 2.0,
                 ca_beta: float = 0.5,
                 signal_threshold: float = 0.3,
                 neuron_mod_gain: float = 0.05,
                 synapse_mod_gain: float = 0.1):
        """
        参数:
            astrocyte_id: 标识符。
            neuron_groups: 关联的神经元对象及其索引列表的元组序列。
            synapse_groups: 关联的连接点对象及其索引列表的元组序列。
            position: 细胞在空间中的坐标（用于直接耦合构建）。
            ca_initial: 初始钙浓度。
            ca_tau: 钙衰减时间常数。
            ca_beta: 活动到钙的转换增益。
            signal_threshold: 释放信号的钙浓度阈值。
            neuron_mod_gain: 对神经元兴奋性的调节增益。
            synapse_mod_gain: 对连接点释放概率的调节增益。
        """
        self.id = astrocyte_id
        self.position = position
        self.ca = ca_initial
        self.ca_tau = ca_tau
        self.ca_beta = ca_beta
        self.signal_threshold = signal_threshold
        self.neuron_mod_gain = neuron_mod_gain
        self.synapse_mod_gain = synapse_mod_gain

        # 处理神经元关联，存储原始参数基准（保留以兼容，但不再用于直接修改）
        self.neuron_refs = []
        for n_obj, indices in neuron_groups:
            if len(indices) == 0:
                continue
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            orig_threshold = n_obj.threshold.data[idx_tensor].clone()
            orig_rest_h = n_obj.rest_h.data[idx_tensor].clone()
            self.neuron_refs.append({
                'obj': n_obj,
                'indices': idx_tensor,
                'orig_threshold': orig_threshold,
                'orig_rest_h': orig_rest_h,
            })
            # 公理一：通过钩子完成星形胶质细胞与神经元的关联
            for idx in indices:
                n_obj.register_astrocyte(self.id)

        # 处理连接点关联
        self.synapse_refs = []
        for s_obj, indices in synapse_groups:
            if len(indices) == 0:
                continue
            idx_tensor = torch.tensor(indices, dtype=torch.long)
            self.synapse_refs.append({
                'obj': s_obj,
                'indices': idx_tensor,
            })

        # 直接耦合邻居，将在 GlialNetwork 中填充
        self.neighbors: Dict['Astrocyte', float] = {}

    # 公理一：内在性优先——感知局部神经活动，用于反馈调节维持稳态
    def perceive(self) -> float:
        """感知局部神经活动水平，返回标量活动值。

        综合所包裹神经元的平均发放率和连接点的近期活动。
        """
        activity = 0.0
        n_groups = 0
        # 神经元活动：使用平均发放率
        for ref in self.neuron_refs:
            n_obj = ref['obj']
            idx = ref['indices']
            rates = n_obj._firing_rate_avg[0, idx]
            activity += rates.mean().item()
            n_groups += 1
        # 连接点活动：使用残余钙浓度作为近期活动指标
        s_groups = 0
        for ref in self.synapse_refs:
            s_obj = ref['obj']
            idx = ref['indices']
            ca = s_obj.resid_Ca[0, idx]
            activity += ca.mean().item()
            s_groups += 1

        total_groups = n_groups + s_groups
        if total_groups > 0:
            activity /= total_groups
        return activity

    def update_calcium(self, activity: float) -> None:
        """根据感知活动更新自身钙浓度（公理一）。"""
        self.ca = self.ca + (-self.ca / self.ca_tau + self.ca_beta * activity)
        self.ca = max(0.0, self.ca)

    def release_signal(self) -> float:
        """根据钙浓度计算信号释放量（公理一、四）。"""
        if self.ca < self.signal_threshold:
            return 0.0
        else:
            return min((self.ca - self.signal_threshold) * 2.0, 1.0)

    # 公理五：通过调制原接口施加信号反馈，不直接修改阈值或逆行信号
    def modulate(self, signal: float) -> None:
        """释放信号，通过局部调制原浓度调节神经元和连接点（公理五）。

        改用新的局部调制原 "local_glial_bias" 直接设置浓度，
        避免与全局调制场冲突。
        """
        if signal <= 0:
            return
        for ref in self.neuron_refs:
            n_obj = ref['obj']
            concentration = self.neuron_mod_gain * signal
            n_obj.apply_modulator("local_glial_bias", concentration)

        for ref in self.synapse_refs:
            s_obj = ref['obj']
            concentration = self.synapse_mod_gain * signal
            s_obj.apply_modulator("local_glial_bias", concentration)

    # 新增：内源性学习规则（公理五：价值内生）
    def intrinsic_learning_step(self, novelty_level: float = 0.0):
        """基于预测误差的胶质细胞学习规则（公理五：价值内生）。

        当全局新奇度较高或局部神经元健康度显著偏离预设值时触发学习。
        遍历所包裹的突触，根据预测误差（|pre_trace - post_trace|）决定
        突触的修剪、保护或增强，最后衰减自身钙浓度以模拟能量消耗。

        参数:
            novelty_level: 全局新奇度水平，由 GlobalStateMonitor 提供。

        返回:
            (pruned_count, enhanced_count) 本次学习步被标记修剪和增强的突触数量。
        """
        # 计算所包裹神经元的平均健康度偏差
        total_dev = 0.0
        total_n = 0
        for ref in self.neuron_refs:
            n_obj = ref['obj']
            indices = ref['indices']
            if len(indices) == 0:
                continue
            # 获取健康度状态
            state = n_obj.get_internal_state()
            h = state['h']  # (batch, num_neurons)
            rest_h = n_obj.rest_h.data  # (num_neurons,)
            # 在批次上取平均，然后选择对应神经元
            h_mean = h.mean(dim=0)  # (num_neurons,)
            h_sel = h_mean[indices]
            rest_sel = rest_h[indices]
            dev = torch.abs(h_sel - rest_sel).sum().item()
            total_dev += dev
            total_n += len(indices)

        avg_health_dev = total_dev / total_n if total_n > 0 else 0.0

        # 学习触发条件：新奇度高 或 局部健康度严重偏离
        trigger = (novelty_level > 0.3) or (avg_health_dev > 0.15)

        pruned = 0
        enhanced = 0

        if trigger:
            # 遍历所包裹的突触组
            for ref in self.synapse_refs:
                s_obj = ref['obj']
                indices = ref['indices']
                if len(indices) == 0:
                    continue
                syn_state = s_obj.get_synaptic_state()
                pre_trace = syn_state['pre_trace']  # (batch, num_synapses)
                post_trace = syn_state['post_trace']
                # 对批次求平均以反映稳定模式
                pre_mean = pre_trace.mean(dim=0)  # (num_synapses,)
                post_mean = post_trace.mean(dim=0)
                pre_sel = pre_mean[indices]
                post_sel = post_mean[indices]
                error = torch.abs(pre_sel - post_sel)

                # 针对每个突触独立决策
                for i, idx in enumerate(indices):
                    e = error[i].item()
                    idx = int(idx)  # 转为 Python int，确保索引兼容
                    if e > 0.3:
                        # 预测误差大：标记可修剪（通过公共属性精确控制单个突触）
                        s_obj.prune_flag[:, idx] = True
                        s_obj.protected[:, idx] = False
                        pruned += 1
                    elif e < 0.1:
                        # 预测误差小：保护并增强
                        s_obj.prune_flag[:, idx] = False
                        s_obj.protected[:, idx] = True
                        s_obj.strength[:, idx] += 0.01
                        s_obj.strength[:, idx].clamp_(max=1.0)
                        enhanced += 1
                    # 中间情况不做处理，由其他机制自然演化

        # 学习步完成后钙浓度衰减（模拟能量消耗）
        self.ca = max(0.0, self.ca * 0.99)

        return pruned, enhanced


class Microglia(nn.Module):
    """小胶质细胞。

    持续监测其所负责的每一个连接点的活动水平和功能状态，独立执行修剪与保护。
    对应白皮书3.2节，公理三、四。
    """

    def __init__(self,
                 synapse: ResonanceSynapse,
                 activity_smoothing: float = 0.01,
                 pruning_threshold: float = 0.1,
                 protection_threshold: float = 0.5,
                 check_interval: int = 100):
        """
        参数:
            synapse: 所监控的完整共鸣连接点对象（该实例中所有连接点由本细胞独立管理）。
            activity_smoothing: 活动历史滑动平均的平滑因子 (0,1]。
            pruning_threshold: 长期活动低于此值时标记修剪。
            protection_threshold: 长期活动高于此值时施加保护。
            check_interval: 每隔多少个胶质步执行一次修剪/保护决策。
        """
        super().__init__()
        self.synapse = synapse
        self.num_synapses = synapse.num_synapses
        self.smoothing = activity_smoothing
        self.pruning_threshold = pruning_threshold
        self.protection_threshold = protection_threshold
        self.check_interval = check_interval
        self.step_counter = 0

        # 公理三：每个连接点独立维护活动历史，使用 register_buffer 注册
        activity_avg_init = self._compute_current_activity().detach().clone()
        self.register_buffer('activity_avg', activity_avg_init)

    # 公理三：基于局部信息（strength 和 pre_trace）独立计算每个连接点的活动水平
    def _compute_current_activity(self) -> torch.Tensor:
        """计算每个连接点的当前活动水平，返回形状 (num_synapses,) 的张量。"""
        with torch.no_grad():
            # 组合耦合强度与连接点前迹反映当前功能状态
            # 修复：按批次求平均强度，兼容任意 batch_size
            return (self.synapse.strength.mean(dim=0) + self.synapse.pre_trace.mean(dim=0)) / 2.0

    def monitor(self) -> None:
        """按每个连接点独立更新其活动历史滑动平均。"""
        current = self._compute_current_activity()  # (num_synapses,)
        self.activity_avg = (1.0 - self.smoothing) * self.activity_avg + self.smoothing * current

    def step(self) -> None:
        """执行一步胶质时间尺度的更新，包含监测与修剪决策。"""
        self.step_counter += 1
        self.monitor()
        # 公理四：慢速时间尺度，按需执行结构演化
        if self.step_counter % self.check_interval == 0:
            self._execute_pruning()

    # 公理三：演化优于训练——修剪与保护独立施加于每个连接点，仅依赖其局部活动历史
    def _execute_pruning(self) -> None:
        """基于每个连接点的长期活动历史，独立执行修剪或保护操作。

        本方法会遍历所有批次样本，为每个批次独立施加相同的修剪/保护标记，
        确保在多批次训练场景下行为一致。
        """
        with torch.no_grad():
            # 低于修剪阈值的连接点标记为可修剪
            prune_mask = self.activity_avg < self.pruning_threshold
            # 高于保护阈值的连接点施加保护
            protect_mask = self.activity_avg > self.protection_threshold

            # 对所有批次应用修剪标记
            self.synapse.prune_flag[:, prune_mask] = True
            # 对受保护的连接点，清除修剪标记并设置保护标记
            self.synapse.prune_flag[:, protect_mask] = False
            self.synapse.protected[:, protect_mask] = True


class GlialNetwork:
    """胶质互联网。

    管理星形胶质细胞群体，构建直接耦合图，驱动状态波扩散与整合，
    协调“感知 → 钙动力学更新 → 直接耦合扩散 → 信号反馈”的完整流程。
    同时驱动小胶质细胞的监测与修剪。
    对应白皮书3.1-3.2节，公理一、三、四。
    """

    def __init__(self,
                 astrocytes: List[Astrocyte],
                 microglias: Optional[List[Microglia]] = None,
                 gap_junction_radius: float = 0.3,
                 diffusion_rate: float = 0.01,
                 time_scale: float = 1.0):
        """
        参数:
            astrocytes: 星形胶质细胞列表。
            microglias: 小胶质细胞列表。
            gap_junction_radius: 直接耦合形成的距离阈值（基于归一化空间坐标）。
            diffusion_rate: 钙离子经直接耦合的扩散系数。
            time_scale: 胶质时间步长缩放因子（用于调整扩散速度）。
        """
        self.astrocytes = astrocytes
        self.microglias = microglias if microglias is not None else []
        self.gap_junction_radius = gap_junction_radius
        self.diffusion_rate = diffusion_rate
        self.dt = time_scale

        # 公理四：基于空间位置构建独立的直接耦合图，形成慢速胶质互联网
        self._build_gap_junctions()

    def _build_gap_junctions(self) -> None:
        """基于星形胶质细胞空间位置构建直接耦合图（公理四）。"""
        num = len(self.astrocytes)
        positions = torch.tensor([a.position for a in self.astrocytes])
        dist = torch.cdist(positions, positions)
        adj = (dist < self.gap_junction_radius) & (~torch.eye(num, dtype=torch.bool))
        weights = torch.exp(-dist ** 2 / (2 * self.gap_junction_radius ** 2))
        weights[~adj] = 0.0
        for i, astro in enumerate(self.astrocytes):
            astro.neighbors = {}
            for j in range(num):
                if adj[i, j]:
                    astro.neighbors[self.astrocytes[j]] = weights[i, j].item()

    def step(self) -> None:
        """执行一次胶质时间步完整更新（公理四：慢速时间尺度）。"""
        # 1. 感知：每个星形胶质细胞计算局部活动
        for astro in self.astrocytes:
            activity = astro.perceive()
            astro.update_calcium(activity)

        # 2. 直接耦合扩散（公理四：状态波在胶质互联网中传播）
        self._diffuse_calcium()

        # 3. 信号释放与反馈调节（公理一）
        for astro in self.astrocytes:
            signal = astro.release_signal()
            astro.modulate(signal)

        # 4. 小胶质细胞更新（公理三：结构演化）
        for micro in self.microglias:
            micro.step()

        # 5. 物理修剪：实际移除被标记的突触（修复二）
        self.remove_pruned_synapses()

    def _diffuse_calcium(self) -> None:
        """执行状态波在直接耦合网络中的扩散（公理四）。"""
        for astro in self.astrocytes:
            dca = 0.0
            for neighbor, weight in astro.neighbors.items():
                dca += weight * self.diffusion_rate * (neighbor.ca - astro.ca)
            astro.ca += dca * self.dt
            astro.ca = max(0.0, astro.ca)

    def remove_pruned_synapses(self) -> None:
        """遍历所有小胶质细胞，实际移除标记为 prune_flag=True 的突触（强度置零，记忆清除）。"""
        for micro in self.microglias:
            syn = micro.synapse
            mask = syn.prune_flag  # 布尔张量，形状 (batch, num_synapses)
            if mask.any():
                # 将耦合强度永久置零
                syn.strength.data.masked_fill_(mask, 0.0)
                # 从交往记忆队列中移除对应条目（全部置零）
                expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(syn.memory_buffer)
                syn.memory_buffer.masked_fill_(expanded_mask, 0.0)
                # 清除修剪标记，避免重复处理
                syn.prune_flag.masked_fill_(mask, False)