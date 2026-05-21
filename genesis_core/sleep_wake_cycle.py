# sleep_wake_cycle.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genesis v0.3 睡眠-觉醒周期与突触稳态模块 (sleep_wake_cycle.py)

基于突触稳态假说 (SHY) 实现周期性离线阶段，通过疲劳累积子浓度自动触发睡眠，
并在睡眠期间通过 SynapticHomeostasis 对突触强度进行向基线的随机衰减，
同时保护被标记的关键突触。

严格遵循 Genesis v2.6 白皮书六条核心公理，所有术语保持中立。
"""

import torch
from typing import List, Tuple, Dict, Any, Optional

# Genesis 核心常量与基类
from .genesis_core import (
    MODULATOR_FATIGUE,
    GLOBAL_SATIETY,
    MorphoNeuron,
    ResonanceSynapse,
)
from .modulator import ModulatorSystem, FatigueAccumulator
from .global_state import GlobalStateScheduler
from .glia import GlialNetwork


class SleepWakeCycle:
    """睡眠-觉醒周期状态机。

    根据调制场中疲劳累积子的浓度自动切换 WAKE 和 SLEEP 两种状态。
    进入睡眠时挂起外部输入、启动睡眠可塑性窗口并降低神经发生概率；
    唤醒时恢复外部输入、正常可塑性和神经发生概率。

    构造参数:
        modulator_system: 调制场系统实例 (ModulatorSystem)。
        global_state_scheduler: 全局状态调度器实例 (GlobalStateScheduler)。
        glia: 胶质网络实例 (GlialNetwork)。
        sleep_threshold: 疲劳累积子浓度阈值，超过则进入睡眠 (默认 0.7)。
        wake_health_threshold: 全局健康度阈值，高于此值且睡眠足够久则唤醒 (默认 0.75)。
        min_sleep_steps: 最短睡眠步数 (默认 50)。
    """

    def __init__(
        self,
        modulator_system: ModulatorSystem,
        global_state_scheduler: GlobalStateScheduler,
        glia: GlialNetwork,
        sleep_threshold: float = 0.7,
        wake_health_threshold: float = 0.75,
        min_sleep_steps: int = 50,
    ) -> None:
        self.modulator_system = modulator_system
        self.global_state_scheduler = global_state_scheduler
        self.glia = glia

        self.sleep_threshold = sleep_threshold
        self.wake_health_threshold = wake_health_threshold
        self.min_sleep_steps = min_sleep_steps

        # 当前状态，WAKE 或 SLEEP
        self._state: str = 'WAKE'
        self._sleep_counter: int = 0

        # 外部输入挂起标志
        self._input_suppressed: bool = False
        # 神经发生概率倍率因子，1.0 为正常，睡眠时降低
        self._neurogenesis_factor: float = 1.0

        # 保存睡眠前星形胶质细胞增益，用于唤醒时恢复
        self._original_astrocyte_gains: List[Tuple[int, float, float]] = []

    @property
    def state(self) -> str:
        """返回当前状态 ('WAKE' 或 'SLEEP')。"""
        return self._state

    @property
    def is_sleeping(self) -> bool:
        """是否处于睡眠状态。"""
        return self._state == 'SLEEP'

    @property
    def input_suppressed(self) -> bool:
        """外部输入是否应被挂起。"""
        return self._input_suppressed

    @property
    def neurogenesis_factor(self) -> float:
        """神经发生概率的倍率因子（1.0 为正常，<1.0 表示抑制）。"""
        return self._neurogenesis_factor

    def step(self, global_state: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """执行一步睡眠-觉醒状态机更新。

        接受可选的 global_state 参数，若未提供则向后兼容地自行调用全局调度器获取。
        在统一运行器 (simulator) 中，全局调度器由 simulator 统一调用并传入此方法，
        以避免重复计算。

        参数:
            global_state: 全局状态指标字典，由 simulator 提供。若为 None，
                          则内部调用 global_state_scheduler.step() 自行获取。

        返回:
            字典，包含当前状态、疲劳浓度、健康度、睡眠计数等快照。
        """
        # 获取全局状态（优先使用传入的，否则向后兼容）
        if global_state is None:
            global_state = self.global_state_scheduler.step()

        # 计算疲劳累积子总源强度作为全局疲劳浓度
        fatigue_concentration = self._get_fatigue_concentration()

        # 获取全局平均健康度
        health = global_state.get(GLOBAL_SATIETY, 0.5)

        # 状态机逻辑
        if self._state == 'WAKE':
            if fatigue_concentration > self.sleep_threshold:
                self._enter_sleep()
        elif self._state == 'SLEEP':
            self._sleep_counter += 1
            if health >= self.wake_health_threshold and self._sleep_counter >= self.min_sleep_steps:
                self._enter_wake()

        return {
            'state': self._state,
            'fatigue_concentration': fatigue_concentration,
            'global_health': health,
            'sleep_counter': self._sleep_counter,
            'input_suppressed': self._input_suppressed,
            'neurogenesis_factor': self._neurogenesis_factor,
        }

    def _get_fatigue_concentration(self) -> float:
        """计算疲劳累积子的总源强度（作为当前整体疲劳浓度的近似）。"""
        if MODULATOR_FATIGUE not in self.modulator_system.modulators:
            return 0.0
        fatigue: FatigueAccumulator = self.modulator_system.modulators[MODULATOR_FATIGUE]
        total = sum(strength for _, strength in fatigue.sources)
        return total

    def _enter_sleep(self) -> None:
        """进入睡眠状态时执行的操作。"""
        self._state = 'SLEEP'
        self._sleep_counter = 0

        # 挂起所有外部输入
        self._input_suppressed = True

        # 降低神经发生概率
        self._neurogenesis_factor = 0.1

        # 启动睡眠特异性可塑性窗口：通过放大星形胶质细胞的突触调节增益实现
        self._activate_sleep_plasticity()

    def _enter_wake(self) -> None:
        """唤醒时执行的操作。"""
        self._state = 'WAKE'
        self._sleep_counter = 0

        # 恢复外部输入
        self._input_suppressed = False

        # 恢复神经发生概率
        self._neurogenesis_factor = 1.0

        # 恢复可塑性窗口
        self._restore_plasticity()

    def _activate_sleep_plasticity(self) -> None:
        """保存当前星形胶质细胞增益并临时提高突触调节增益。"""
        self._original_astrocyte_gains.clear()
        for astro in self.glia.astrocytes:
            # 保存原始增益
            self._original_astrocyte_gains.append(
                (astro.id, astro.synapse_mod_gain, astro.neuron_mod_gain)
            )
            # 提高突触调节增益以增强睡眠期间的可塑性
            astro.synapse_mod_gain *= 2.0
            # 可选择是否也调整神经元增益，根据需求按需开启
            # astro.neuron_mod_gain *= 1.5

    def _restore_plasticity(self) -> None:
        """将星形胶质细胞的增益恢复至睡眠前水平。"""
        restore_map = {aid: (sg, ng) for aid, sg, ng in self._original_astrocyte_gains}
        for astro in self.glia.astrocytes:
            if astro.id in restore_map:
                astro.synapse_mod_gain, astro.neuron_mod_gain = restore_map[astro.id]
        self._original_astrocyte_gains.clear()


class SynapticHomeostasis:
    """突触稳态维护：在睡眠状态下对突触强度进行朝向基线的随机衰减。

    模拟突触稳态假说 (SHY)，未被保护 (protect_flag=False) 的突触每步衰减并趋向基线，
    而被保护 (protected=True) 的突触维持强度。

    构造参数:
        connections: 突触连接列表，元素为 (synapse, pre_neuron, post_neuron)。
        decay_factor: 每步强度乘性衰减因子 (默认 0.999)。
        baseline: 目标基线强度 (默认 0.01)。
        move_rate: 朝基线移动的线性速率 (默认 0.001)。
    """

    def __init__(
        self,
        connections: List[Tuple[ResonanceSynapse, MorphoNeuron, MorphoNeuron]],
        decay_factor: float = 0.999,
        baseline: float = 0.01,
        move_rate: float = 0.001,
    ) -> None:
        self.connections = connections
        self.decay_factor = decay_factor
        self.baseline = baseline
        self.move_rate = move_rate

    def renormalize(self) -> None:
        """执行一次突触稳态重归一化：对每个未被保护的突触强度先乘性衰减，再向基线移动一小步。"""
        with torch.no_grad():
            for syn, _, _ in self.connections:
                protected = syn.protected  # shape: (batch_size, num_synapses)
                mask = ~protected          # True 表示未受保护，需衰减

                if mask.any():
                    # 克隆当前强度以生成新值
                    strength = syn.strength.clone()

                    # 乘性衰减
                    decayed = strength * self.decay_factor
                    # 向基线移动
                    moved = decayed + (self.baseline - decayed) * self.move_rate

                    # 仅更新未保护的突触
                    new_strength = torch.where(mask, moved, strength)
                    new_strength.clamp_(0.0, 1.0)

                    # 用新张量替换突触强度缓冲区
                    syn.strength.data = new_strength