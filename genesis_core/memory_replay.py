# memory_replay.py — Genesis v0.3 梦境重放模块
# 在睡眠状态下向网络注入温和的内部噪声，驱动突触传播与神经元自发重放，
# 以巩固记忆并维持网络临界性。
# 严格遵循 Genesis v2.6 白皮书六条核心公理，所有术语保持中立。

import torch
from typing import Dict, List, Tuple, Optional
from .genesis_core import MorphoNeuron, ResonanceSynapse


class MemoryReplay:
    """梦境重放模块：睡眠期间利用噪声驱动自发神经活动以巩固记忆。

    理论依据（白皮书 v2.6 第十五章）：自发的神经重放可以巩固记忆、防止遗忘。
    噪声驱动活动是维持网络临界性和记忆巩固的涌现功能。
    在睡眠状态下，本模块向所有健康度足够的神经元注入温和的随机噪声电流，
    使其在觉醒期间被强化的共鸣模式自发复现。

    构造参数:
        neurons: 形态神经元列表。要求每个实例仅管理单个神经元（num_neurons=1），
                 以便精确控制每个神经元的噪声注入和状态跟踪。
        connections: 突触连接列表，每个元素为 (synapse, pre_neuron, post_neuron)。
                     建议其中每个突触实例的 num_synapses=1，以确保映射明确。
        noise_amplitude: 噪声电流的基础幅度，默认 0.05。实际注入幅度 = base * h，
                         健康度 h 越高的神经元获得越强的噪声电流。
    """

    def __init__(
        self,
        neurons: List[MorphoNeuron],
        connections: List[Tuple[ResonanceSynapse, MorphoNeuron, MorphoNeuron]],
        noise_amplitude: float = 0.05,
    ) -> None:
        # 基本参数验证
        for nrn in neurons:
            if nrn.num_neurons != 1:
                raise ValueError(
                    f"MemoryReplay 要求每个神经元实例的 num_neurons=1，"
                    f"但发现实例含 {nrn.num_neurons} 个神经元。"
                    f"请将层拆分为单个神经元实例。"
                )

        self.neurons = neurons
        self.connections = connections
        self.noise_amplitude = noise_amplitude

        # 内部缓存：上一时间步的脉冲和共振敏感度，用于突触传播
        self._prev_spikes: Dict[int, torch.Tensor] = {}
        self._prev_rs: Dict[int, torch.Tensor] = {}

        # 批大小与设备，首次调用 replay_step 时自动推断
        self._batch_size: Optional[int] = None
        self._device: Optional[torch.device] = None

        # 最近一步的统计快照
        self._last_stats: Dict[str, float] = {}

    def _initialize_caches(self, batch_size: int, device: torch.device) -> None:
        """根据当前网络状态初始化脉冲及敏感度缓存。"""
        self._batch_size = batch_size
        self._device = device
        for nrn in self.neurons:
            nid = id(nrn)
            # 初始脉冲为零（休息状态）
            self._prev_spikes[nid] = torch.zeros(batch_size, device=device)
            # 初始共振敏感度取自当前神经元状态
            state = nrn.get_internal_state()
            self._prev_rs[nid] = state['r'].clone().detach().to(device)

    def replay_step(self) -> None:
        """执行一步梦境重放：注入噪声、突触传播、神经元发放。

        该方法应由 SleepWakeCycle 在睡眠状态下周期性调用。
        具体流程：
        1. 感知神经元健康度 h，计算噪声幅度并直接扰动膜电位 u。
        2. 基于上一步的脉冲和共振敏感度，通过共鸣突触计算突触后电流。
        3. 各神经元汇总电流（无外部输入），更新状态并产生输出。
        4. 记录本步的发放率与参与重放的神经元比例（统计值已适配任意批次大小）。

        异常:
            RuntimeError: 若神经元状态尚未初始化（_u 为空）。
        """
        # ---- 推断批次大小与设备 ----
        if self._batch_size is None:
            if not self.neurons:
                return
            sample_nrn = self.neurons[0]
            state = sample_nrn.get_internal_state()
            u = state['u']
            if u.numel() == 0:
                raise RuntimeError("神经元状态未初始化，请先调用 reset_state。")
            batch_size = u.shape[0]
            device = u.device
            self._initialize_caches(batch_size, device)
        else:
            batch_size = self._batch_size
            device = self._device

        # ---- 1. 健康度感知与噪声注入 ----
        # 健康度 >= 0.2 的神经元将参与重放：噪声幅度 = base * h
        # 改为逐样本计算，批次内每个样本的健康度差异被正确反映
        injected_count = 0
        total_neurons = len(self.neurons)

        with torch.no_grad():
            for nrn in self.neurons:
                state = nrn.get_internal_state()
                h = state['h']  # 形状 (batch, 1) 因为 num_neurons=1
                # 构建掩码：健康度 >= 0.2 的样本参与重放
                mask = (h >= 0.2).float()  # (batch, 1)
                # 注入噪声：幅度 = mask * noise_amplitude * h
                # 低健康度样本对应位置的噪声被置零
                noise = mask * self.noise_amplitude * h * torch.randn_like(nrn._u)
                nrn._u.add_(noise)  # 直接扰动膜电位，等效于注入电流
                # 累计被注入噪声的样本总数
                injected_count += mask.sum().item()

        # ---- 2. 突触传播 ----
        accum_current: Dict[int, torch.Tensor] = {}
        for syn, pre_nrn, post_nrn in self.connections:
            pre_id = id(pre_nrn)
            post_id = id(post_nrn)

            # 上一时间步的脉冲与共振敏感度
            pre_spike = self._prev_spikes.get(pre_id,
                                              torch.zeros(batch_size, device=device))
            post_spike = self._prev_spikes.get(post_id,
                                               torch.zeros(batch_size, device=device))
            pre_r = self._prev_rs.get(pre_id)
            post_r = self._prev_rs.get(post_id)
            if pre_r is None:
                pre_r = torch.zeros(batch_size, 1, 1, device=device)
            if post_r is None:
                post_r = torch.zeros(batch_size, 1, 1, device=device)

            # 调整为突触期望的输入形状：(batch, 1) 和 (batch, 1, r_dim)
            pre_spike_syn = pre_spike.unsqueeze(1)   # (batch, 1)
            post_spike_syn = post_spike.unsqueeze(1)
            if pre_r.dim() == 2:
                pre_r = pre_r.unsqueeze(1)           # (batch, r_dim) -> (batch, 1, r_dim)
            if post_r.dim() == 2:
                post_r = post_r.unsqueeze(1)

            syn_current = syn(pre_spike_syn, post_spike_syn, pre_r, post_r)
            syn_current = syn_current.squeeze(1)     # (batch,)

            if post_id not in accum_current:
                accum_current[post_id] = torch.zeros(batch_size, device=device)
            accum_current[post_id] += syn_current

        # ---- 3. 神经元更新 ----
        new_spikes: Dict[int, torch.Tensor] = {}
        new_rs: Dict[int, torch.Tensor] = {}
        fired_count = 0

        for nrn in self.neurons:
            nid = id(nrn)
            syn_input = accum_current.get(nid, torch.zeros(batch_size, device=device))
            # 无外部输入，噪声已通过膜电位扰动注入；将电流转为 (batch, 1, 1)
            total_input = syn_input.unsqueeze(1).unsqueeze(1)
            spike = nrn(total_input)  # (batch, 1)
            spike = spike.squeeze(1)  # (batch,)
            new_spikes[nid] = spike
            new_rs[nid] = nrn.get_internal_state()['r'].clone().detach()

            # 逐样本统计发放数
            fired_count += (spike > 0).sum().item()

        # 更新缓存
        self._prev_spikes = new_spikes
        self._prev_rs = new_rs

        # ---- 4. 统计信息 ----
        total_samples = total_neurons * batch_size
        mean_firing_rate = fired_count / total_samples if total_samples > 0 else 0.0
        if injected_count > 0:
            participation_rate = fired_count / injected_count
        else:
            participation_rate = 0.0

        self._last_stats = {
            'mean_firing_rate': mean_firing_rate,
            'participation_rate': participation_rate,
            'injected_neurons': injected_count,   # 现在表示注入噪声的样本总数
            'total_neurons': total_neurons,
            'fired_neurons': fired_count,         # 现在表示发放的样本总数
        }

    def get_replay_statistics(self) -> Dict[str, float]:
        """返回最近一次 replay_step 的重放统计信息（已适配批处理）。

        返回值字典包含以下键：
            - 'mean_firing_rate': 所有样本所有神经元的平均发放率（发放样本数 / 总样本数）。
            - 'participation_rate': 健康度足够（>=0.2）的样本中发放的比例。
            - 'injected_neurons': 本步被注入噪声的样本总数。
            - 'total_neurons': 网络中的神经元实例总数。
            - 'fired_neurons': 本步实际发放的样本总数。
        """
        return self._last_stats.copy()