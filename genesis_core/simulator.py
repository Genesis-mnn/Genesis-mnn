# genesis_core/simulator.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genesis 统一运行器 (simulator.py)

封装神经元、突触、胶质网络、调制场、全局状态调度器和神经发生凋亡模块，
提供统一的 step() 方法和便捷工厂函数。
"""
import torch
import random
from typing import List, Optional, Dict, Any, Tuple, Union
import warnings

from .genesis_core import MorphoNeuron, ResonanceSynapse
from .neuron import LIFNeuron
from .synapse import STDPSynapse
from .glia import GlialNetwork
from .modulator import ModulatorSystem, create_default_modulators
from .global_state import GlobalStateMonitor, GlobalStateScheduler
from .learning import IntrinsicLearningScheduler
from .neurogenesis import NeurogenesisScheduler


class GenesisSimulator:
    """创世纪模拟器——封装形态神经网络的完整仿真循环。

    提供统一的 step() 方法，自动处理突触传递、可塑性更新、胶质调控、
    调制场扩散、神经发生与凋亡以及全局状态调度，使用户无需手动编写复杂的仿真代码。

    参数:
        neurons: 网络中所有独立神经元的列表，每个元素为 MorphoNeuron 实例（num_neurons=1）。
        connections: 网络中所有突触连接的列表，每个元素为 (synapse, pre_neuron, post_neuron)。
        glia: 可选的 GlialNetwork 实例。
        modulator_system: 可选的 ModulatorSystem 实例。
        global_scheduler: 可选的 GlobalStateScheduler 实例。
        batch_size: 默认批次大小。
        intrinsic_scheduler: 可选的 IntrinsicLearningScheduler 实例。
        neurogenesis_scheduler: 可选的 NeurogenesisScheduler 实例。
    """

    def __init__(
        self,
        neurons: List[MorphoNeuron],
        connections: List[Tuple[ResonanceSynapse, MorphoNeuron, MorphoNeuron]],
        glia: Optional[GlialNetwork] = None,
        modulator_system: Optional[ModulatorSystem] = None,
        global_scheduler: Optional[GlobalStateScheduler] = None,
        batch_size: int = 1,
        intrinsic_scheduler: Optional[IntrinsicLearningScheduler] = None,
        neurogenesis_scheduler: Optional[NeurogenesisScheduler] = None,
    ) -> None:
        if not neurons:
            raise ValueError("至少需要提供一个神经元。")
        self.neurons = neurons
        self.connections = connections
        self.glia = glia
        self.modulator_system = modulator_system
        self.global_scheduler = global_scheduler
        self.batch_size = batch_size
        self.intrinsic_scheduler = intrinsic_scheduler
        self.neurogenesis_scheduler = neurogenesis_scheduler

        # 内部缓存：上一时间步的脉冲和共振敏感度（用于突触传递）
        self.prev_spikes: Dict[int, torch.Tensor] = {}
        self.prev_rs: Dict[int, torch.Tensor] = {}

        self._step_counter: int = 0
        self.reset(batch_size)

    def reset(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        """重置所有组件的内部状态并重新分配缓存。"""
        self.batch_size = batch_size
        if device is None:
            device = next(self.neurons[0].parameters()).device

        for nrn in self.neurons:
            nrn.reset_state(batch_size, device)

        for syn, _, _ in self.connections:
            syn.reset_state(batch_size, device)

        if self.glia is not None and hasattr(self.glia, 'reset'):
            self.glia.reset()
        if self.global_scheduler is not None and hasattr(self.global_scheduler.monitor, 'reset'):
            self.global_scheduler.monitor.reset()

        self.prev_spikes.clear()
        self.prev_rs.clear()
        for nrn in self.neurons:
            self.prev_spikes[id(nrn)] = torch.zeros(batch_size, device=device)
            r = nrn.get_internal_state()['r'].clone().detach().to(device)
            self.prev_rs[id(nrn)] = r

        self._step_counter = 0

    def step(
        self,
        external_input: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        """执行一个完整时间步的仿真，返回当前步的监视快照。"""
        device = next(self.neurons[0].parameters()).device
        batch_size = self.batch_size

        if external_input is not None:
            if len(external_input) != len(self.neurons):
                raise ValueError(
                    f"external_input 长度 ({len(external_input)}) 与神经元数量 ({len(self.neurons)}) 不匹配。"
                )
        else:
            external_input = [None] * len(self.neurons)

        # ---- 突触脉冲传播与可塑性更新（基于 t-1 脉冲） ----
        current_accum: Dict[int, torch.Tensor] = {}
        for syn, pre_nrn, post_nrn in self.connections:
            pre_id = id(pre_nrn)
            post_id = id(post_nrn)

            pre_spike = self.prev_spikes.get(pre_id,
                                             torch.zeros(batch_size, device=device))
            post_spike = self.prev_spikes.get(post_id,
                                              torch.zeros(batch_size, device=device))
            pre_r = self.prev_rs.get(pre_id)
            if pre_r is None:
                pre_r = torch.zeros(batch_size, 1, 1, device=device)
            post_r = self.prev_rs.get(post_id)
            if post_r is None:
                post_r = torch.zeros(batch_size, 1, 1, device=device)

            # 适配形状：synapse 期望 (batch, num_synapses), (batch, num_synapses, r_dim)
            pre_spike_syn = pre_spike.unsqueeze(1)   # (batch, 1)
            post_spike_syn = post_spike.unsqueeze(1)
            syn_current = syn(pre_spike_syn, post_spike_syn, pre_r, post_r)  # (batch, 1)
            syn_current = syn_current.squeeze(1)      # (batch,)

            if post_id not in current_accum:
                current_accum[post_id] = torch.zeros(batch_size, device=device)
            current_accum[post_id] += syn_current

        # ---- 神经元前向传播 ----
        new_spikes: Dict[int, torch.Tensor] = {}
        new_rs: Dict[int, torch.Tensor] = {}
        for idx, nrn in enumerate(self.neurons):
            nid = id(nrn)
            syn_input = current_accum.get(nid, torch.zeros(batch_size, device=device))
            ext = external_input[idx]
            if ext is not None:
                if ext.dim() == 1:
                    ext = ext.unsqueeze(1).unsqueeze(1)  # (batch,1,1)
                elif ext.dim() == 2:
                    ext = ext.unsqueeze(1)
                total_input = syn_input.unsqueeze(1).unsqueeze(1) + ext
            else:
                total_input = syn_input.unsqueeze(1).unsqueeze(1)  # (batch,1,1)

            spike = nrn(total_input)  # (batch, num_neurons) = (batch,1)
            spike = spike.squeeze(1)  # (batch,)
            new_spikes[nid] = spike
            new_rs[nid] = nrn.get_internal_state()['r'].clone().detach()

        self.prev_spikes = new_spikes
        self.prev_rs = new_rs

        # ---- 胶质网络步进 ----
        if self.glia is not None:
            self.glia.step()

        # ---- 神经发生与凋亡 ----
        neurogenesis_stats = None
        if self.neurogenesis_scheduler is not None:
            # 将脉冲信息转换为调度器需要的格式
            spikes_dict = {}
            for nrn in self.neurons:
                nid = id(nrn)
                val = new_spikes.get(nid, torch.zeros(batch_size, device=device))
                spikes_dict[nid] = val.mean().item()
            self.neurogenesis_scheduler.record_spikes(spikes_dict)
            self.neurogenesis_scheduler.step()

            # 同步神经元状态：调度器可能新增或移除了神经元
            current_ids = {id(n) for n in self.neurons}
            for nid in list(self.prev_spikes.keys()):
                if nid not in current_ids:
                    del self.prev_spikes[nid]
                    del self.prev_rs[nid]
            for nrn in self.neurons:
                nid = id(nrn)
                if nid not in self.prev_spikes:
                    self.prev_spikes[nid] = torch.zeros(batch_size, device=device)
                    self.prev_rs[nid] = nrn.get_internal_state()['r'].clone().detach().to(device)
            neurogenesis_stats = self.neurogenesis_scheduler.get_statistics()

        # ---- 全局状态调度器 ----
        global_state = {}
        if self.global_scheduler is not None:
            global_state = self.global_scheduler.step()

        # ---- 内源性学习调度器 ----
        learn_stats = None
        if self.intrinsic_scheduler is not None:
            learn_stats = self.intrinsic_scheduler.step()

        # ---- 收集快照 ----
        all_spikes = []
        all_u = []
        all_h = []
        for nrn in self.neurons:
            state = nrn.get_internal_state()
            nid = id(nrn)
            spk = new_spikes.get(nid, torch.zeros(batch_size, device=device))
            all_spikes.append(spk)
            all_u.append(state['u'].squeeze(1) if state['u'].dim() == 2 else state['u'])
            all_h.append(state['h'].squeeze(1) if state['h'].dim() == 2 else state['h'])

        if all_spikes:
            mean_fr = torch.stack(all_spikes).float().mean().item()
            mean_u = torch.cat([u.unsqueeze(0) for u in all_u]).float().mean().item()
            mean_h = torch.cat([h.unsqueeze(0) for h in all_h]).float().mean().item()
        else:
            mean_fr = mean_u = mean_h = 0.0

        self._step_counter += 1
        snapshot = {
            'step': self._step_counter,
            'mean_firing_rate': mean_fr,
            'mean_membrane': mean_u,
            'mean_health': mean_h,
            'global_state': global_state if global_state else None,
            'intrinsic_learning': learn_stats,
            'neurogenesis': neurogenesis_stats,
        }
        return snapshot

    def apply_modulator(self, name: str, concentration: float) -> None:
        """手动向所有神经元和突触施加调制原浓度。"""
        for nrn in self.neurons:
            nrn.apply_modulator(name, concentration)
        for syn, _, _ in self.connections:
            syn.apply_modulator(name, concentration)
        if self.glia is not None:
            pass


def create_default_simulator(
    num_neurons: int = 100,
    connection_prob: float = 0.1,
    with_glia: bool = False,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> GenesisSimulator:
    """一键创建包含默认 LIF 神经元、随机连接、调制场和全局调度器的模拟器。"""
    if device is None:
        device = torch.device('cpu')

    # 1. 创建独立神经元
    neurons = []
    for _ in range(num_neurons):
        nrn = LIFNeuron(
            num_neurons=1, num_dendrites=1,
            tau=10.0, v_threshold=1.0, v_reset=0.0
        ).to(device)
        neurons.append(nrn)

    # 2. 随机自连接
    connections = []
    for i in range(num_neurons):
        for j in range(num_neurons):
            if i == j:
                continue
            if random.random() < connection_prob:
                syn = STDPSynapse(
                    num_synapses=1,
                    pool_total=1.0, k_cyc=0.01, k_rp=0.1, p_base=0.2,
                    A_plus=0.005, A_minus=0.005,
                    tau_pre=20.0, tau_post=20.0,
                    lr=1.0, reward_scale=1.0,
                ).to(device)
                connections.append((syn, neurons[i], neurons[j]))

    # 3. 调制场系统
    modulator_system = create_default_modulators()
    for nrn in neurons:
        modulator_system.register_target(nrn, (0.0, 0.0, 0.0))
    for syn, _, _ in connections:
        modulator_system.register_target(syn, (0.0, 0.0, 0.0))

    # 4. 全局状态监测与调度器
    syn_instances = list(set(syn for syn, _, _ in connections))
    monitor = GlobalStateMonitor(neurons=neurons, connections=syn_instances, ema_alpha=0.1)
    global_scheduler = GlobalStateScheduler(monitor=monitor, modulator_system=modulator_system)

    # 5. 可选胶质网络
    glia = None
    if with_glia:
        from .glia import Astrocyte, Microglia, GlialNetwork
        neuron_groups = [(nrn, [0]) for nrn in neurons]  # 每个神经元实例只有一个神经元
        synapse_groups = [(syn, [0]) for syn in syn_instances]
        astro = Astrocyte(
            astrocyte_id=0,
            neuron_groups=neuron_groups,
            synapse_groups=synapse_groups,
            position=(0.0, 0.0, 0.0),
        )
        microglias = [Microglia(synapse=syn) for syn in syn_instances]
        glia = GlialNetwork(astrocytes=[astro], microglias=microglias)

    # 6. 组装模拟器
    sim = GenesisSimulator(
        neurons=neurons,
        connections=connections,
        glia=glia,
        modulator_system=modulator_system,
        global_scheduler=global_scheduler,
        batch_size=batch_size,
    )
    sim.reset(batch_size, device=device)
    return sim