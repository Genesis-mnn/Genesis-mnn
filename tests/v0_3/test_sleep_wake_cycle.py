# tests/v0_3/test_sleep_wake_cycle.py

import unittest
import torch

# 导入需要测试的核心模块
from genesis_core.sleep_wake_cycle import SleepWakeCycle, SynapticHomeostasis
from genesis_core.modulator import (
    ModulatorSystem, FatigueAccumulator, create_default_modulators
)
from genesis_core.global_state import GlobalStateMonitor, GlobalStateScheduler
from genesis_core.glia import GlialNetwork, Astrocyte
from genesis_core.neuron import LIFNeuron
from genesis_core.synapse import STDPSynapse
from genesis_core.genesis_core import MODULATOR_FATIGUE, GLOBAL_SATIETY


class TestSleepWakeCycle(unittest.TestCase):
    """测试睡眠-觉醒周期状态机的状态切换逻辑。"""

    def setUp(self):
        """构建测试所需的最小化组件。"""
        # 单个神经元，用于全局状态监测器计算内在健康度
        self.neuron = LIFNeuron(num_neurons=1, num_dendrites=1)
        # 创建测试用的神经元和突触实例
        self.neurons = [LIFNeuron(num_neurons=5)]
        self.synapses = [STDPSynapse(num_synapses=5)]
        # 包含五种预置调制原的调制场系统
        self.modulator_system = create_default_modulators()
        # 全局状态监测器 （EMA 平滑系数设为 1.0，避免历史值干扰测试）
        self.monitor = GlobalStateMonitor(
            neurons=[self.neuron],
            connections=[],
            ema_alpha=1.0,
        )
        # 全局状态调度器（闭环“监测 → 调制”）
        self.global_scheduler = GlobalStateScheduler(
            monitor=self.monitor,
            modulator_system=self.modulator_system,
        )
        # 先创建一个星形胶质细胞，包裹第一个神经元和第一个突触
        astro = Astrocyte(
            astrocyte_id=1,
            neuron_groups=[(self.neurons[0], [0])],
            synapse_groups=[(self.synapses[0], [0])],
            position=(0.0, 0.0, 0.0)
        )
        self.glia = GlialNetwork(astrocytes=[astro], microglias=[])

        # 构造睡眠-觉醒状态机，使用较短的睡眠时间以加速测试
        self.swc = SleepWakeCycle(
            modulator_system=self.modulator_system,
            global_state_scheduler=self.global_scheduler,
            glia=self.glia,
            sleep_threshold=0.7,
            wake_health_threshold=0.75,
            min_sleep_steps=1,       # 一步后即可唤醒
        )

    def test_initial_state_is_wake(self):
        """创建 SleepWakeCycle 实例后，初始状态应为 WAKE。"""
        self.assertEqual(self.swc.state, 'WAKE')
        self.assertFalse(self.swc.is_sleeping)
        self.assertFalse(self.swc.input_suppressed)
        self.assertEqual(self.swc.neurogenesis_factor, 1.0)

    def test_fatigue_triggers_sleep(self):
        """疲劳累积子浓度超过阈值时，step() 应切换状态为 SLEEP。"""
        # 获取疲劳累积子并注入足够强的源
        fatigue = self.modulator_system.modulators[MODULATOR_FATIGUE]
        self.assertIsInstance(fatigue, FatigueAccumulator)
        # 源总强度需大于 sleep_threshold (0.7)。考虑到 step 中会先衰减一次，
        # 此处设置较高值以确保衰减后仍满足条件。
        fatigue.sources = [((0.0, 0.0, 0.0), 1.0)]
        total = sum(strength for _, strength in fatigue.sources)
        self.assertGreater(total, self.swc.sleep_threshold)

        # 执行一步，触发睡眠
        result = self.swc.step()
        self.assertEqual(self.swc.state, 'SLEEP', f"状态仍为 {self.swc.state}")
        self.assertTrue(self.swc.is_sleeping)
        self.assertTrue(self.swc.input_suppressed)
        self.assertLess(self.swc.neurogenesis_factor, 1.0)

    def test_health_recovery_triggers_wake(self):
        """在睡眠状态下，若全局健康度恢复并满足最短睡眠步数，应唤醒为 WAKE。"""
        # 先进入睡眠
        fatigue = self.modulator_system.modulators[MODULATOR_FATIGUE]
        fatigue.sources = [((0.0, 0.0, 0.0), 1.0)]
        self.swc.step()
        self.assertEqual(self.swc.state, 'SLEEP')

        # 手动将神经元健康度提升至高于 wake_health_threshold (0.75)
        # 直接修改内部张量，使 GlobalStateMonitor 下一步读到高健康度
        self.neuron._h = torch.tensor([[0.85]], dtype=torch.float32)

        # 执行一步（此时 sleep_counter 已为 1 >= min_sleep_steps）
        result = self.swc.step()
        self.assertEqual(self.swc.state, 'WAKE')
        self.assertFalse(self.swc.is_sleeping)
        self.assertFalse(self.swc.input_suppressed)
        self.assertEqual(self.swc.neurogenesis_factor, 1.0)


class TestSynapticHomeostasis(unittest.TestCase):
    """测试突触稳态重归一化功能。"""

    def test_synaptic_homeostasis_renormalize(self):
        """对未保护突触调用 renormalize() 应使强度向基线方向衰减。"""
        # 创建两个占位神经元（用于构建连接元组，实际不被使用）
        neuron1 = LIFNeuron(num_neurons=1, num_dendrites=1)
        neuron2 = LIFNeuron(num_neurons=1, num_dendrites=1)

        # 创建 STDP 突触实例并设置初始强度
        syn = STDPSynapse(num_synapses=1)
        with torch.no_grad():
            syn.strength.fill_(0.5)
        # 默认 protected 为全零，即所有突触未受保护
        self.assertFalse(syn.protected.any())

        # 构建连接列表
        connections = [(syn, neuron1, neuron2)]

        # 创建突触稳态对象，使用默认衰减参数
        homeostasis = SynapticHomeostasis(
            connections=connections,
            decay_factor=0.999,
            baseline=0.01,
            move_rate=0.001,
        )

        # 记录初始强度
        initial_strength = syn.strength.clone()

        # 执行一次重归一化
        homeostasis.renormalize()

        # 获取更新后的强度
        new_strength = syn.strength

        # 验证每一个突触的强度都有所降低
        for i in range(syn.num_synapses):
            self.assertLess(
                new_strength[0, i].item(),
                initial_strength[0, i].item(),
                f"突触 {i} 强度未降低（初始 {initial_strength[0, i].item():.6f}，"
                f"更新后 {new_strength[0, i].item():.6f}）"
            )


if __name__ == '__main__':
    unittest.main()