# tests/v0_3/test_v0_3_full_integration.py
import unittest
import torch

# Genesis 核心组件
from genesis_core.neuron import LIFNeuron
from genesis_core.synapse import STDPSynapse
from genesis_core.glia import Astrocyte, Microglia, GlialNetwork
from genesis_core.modulator import create_default_modulators
from genesis_core.genesis_core import (
    MODULATOR_FATIGUE,
    GLOBAL_SATIETY,
)
from genesis_core.global_state import GlobalStateMonitor, GlobalStateScheduler
from genesis_core.sleep_wake_cycle import SleepWakeCycle, SynapticHomeostasis
from genesis_core.memory_replay import MemoryReplay
from genesis_core.dendritic_spine_sleep import DendriticSpineSleepRemodeling
from genesis_core.simulator import GenesisSimulator


class TestV03FullIntegration(unittest.TestCase):
    """v0.3 端到端集成测试：覆盖睡眠-觉醒周期、梦境重放、突触稳态及唤醒恢复。"""

    def setUp(self):
        """构建小型形态神经网络及所有 v0.3 组件。

        顺序要求：先创建神经元与突触，再创建胶质细胞（Astrocyte 包裹第一个神经元和第一个突触），
        最后创建 GlialNetwork（传入非空列表）。
        """
        batch_size = 1

        # ---- 1. 神经元：10 个独立 LIFNeuron ----
        self.neurons = [
            LIFNeuron(num_neurons=1, num_dendrites=1, tau=10.0,
                      v_threshold=1.0, v_reset=0.0, rest_h=0.7)
            for _ in range(10)
        ]

        # ---- 2. 突触：10 个 STDPSynapse，每个连接一对神经元 ----
        self.synapses = [STDPSynapse(num_synapses=1) for _ in range(10)]
        self.connections = [
            (self.synapses[i], self.neurons[i], self.neurons[(i + 1) % 10])
            for i in range(10)
        ]

        # ---- 3. Astrocyte：包裹第一个神经元和第一个突触 ----
        self.astro = Astrocyte(
            astrocyte_id=0,
            neuron_groups=[(self.neurons[0], [0])],
            synapse_groups=[(self.synapses[0], [0])],
            position=(0.0, 0.0, 0.0),
        )

        # ---- 4. Microglia：每个突触一个 ----
        self.micros = [Microglia(synapse=syn) for syn in self.synapses]

        # ---- 5. GlialNetwork（非空列表）----
        self.glia = GlialNetwork(astrocytes=[self.astro], microglias=self.micros)

        # ---- 6. 调制场系统 ----
        self.mod_sys = create_default_modulators()
        for nrn in self.neurons:
            self.mod_sys.register_target(nrn, (0.0, 0.0, 0.0))
        for syn in self.synapses:
            self.mod_sys.register_target(syn, (0.0, 0.0, 0.0))

        # ---- 7. 全局状态监测与调度器 ----
        monitor = GlobalStateMonitor(
            neurons=self.neurons,
            connections=self.synapses,
            ema_alpha=0.5,  # 较大平滑系数，便于快速测试
        )
        self.global_scheduler = GlobalStateScheduler(
            monitor=monitor, modulator_system=self.mod_sys
        )

        # ---- 8. 睡眠-觉醒周期（减小最短睡眠步数以提速）----
        self.sleep_wake = SleepWakeCycle(
            modulator_system=self.mod_sys,
            global_state_scheduler=self.global_scheduler,
            glia=self.glia,
            sleep_threshold=0.7,
            wake_health_threshold=0.75,
            min_sleep_steps=3,
        )

        # ---- 9. 梦境重放与树突棘睡眠重塑 ----
        self.replay = MemoryReplay(
            neurons=self.neurons, connections=self.connections
        )
        self.spine_remodel = DendriticSpineSleepRemodeling(
            neurons=self.neurons, connections=self.connections
        )

        # ---- 10. GenesisSimulator 注入全部组件 ----
        self.sim = GenesisSimulator(
            neurons=self.neurons,
            connections=self.connections,
            glia=self.glia,
            modulator_system=self.mod_sys,
            global_scheduler=self.global_scheduler,
            batch_size=batch_size,
            sleep_wake_cycle=self.sleep_wake,
            memory_replay=self.replay,
            dendritic_spine_remodeling=self.spine_remodel,
        )
        self.sim.reset(batch_size=batch_size)

    # ========================================================================
    # 测试方法
    # ========================================================================

    def test_full_sleep_wake_cycle(self):
        """
        完整睡眠-觉醒周期测试：
        - 初始状态为 WAKE
        - 手动设置疲劳累积子浓度超过阈值，触发睡眠
        - 验证状态切换为 SLEEP 且梦境重放、树突棘重塑被执行
        - 手动将全局健康度恢复到 0.75 以上，触发唤醒
        - 验证状态切换回 WAKE
        """
        # 初始 WAKE 状态，运行 10 步验证
        for _ in range(10):
            self.sim.step()
        self.assertEqual(self.sleep_wake.state, 'WAKE',
                         "初始状态应为 WAKE")

        # 手动添加疲劳源，使疲劳浓度 > 0.7
        fatigue = self.mod_sys.modulators[MODULATOR_FATIGUE]
        fatigue.sources.append((fatigue.release_position, 1.0))

        # 运行足够步数让睡眠触发
        for _ in range(10):
            self.sim.step()
            if self.sleep_wake.state == 'SLEEP':
                break
        self.assertEqual(self.sleep_wake.state, 'SLEEP',
                         "疲劳浓度超过阈值后应进入 SLEEP")

        # 验证睡眠期间梦境重放和树突棘重塑已被执行
        replay_stats = self.replay.get_replay_statistics()
        spine_stats = self.spine_remodel.get_remodel_statistics()
        self.assertIn('injected_neurons', replay_stats,
                      "梦境重放统计应包含 injected_neurons")
        self.assertIn('new_spines', spine_stats,
                      "树突棘重塑统计应包含 new_spines")
        # 若健康度允许，应有实际注入
        self.assertGreaterEqual(replay_stats['injected_neurons'], 1,
                                "睡眠期间应至少有一个神经元获得噪声注入")

        # 手动恢复全局健康度以唤醒：将所有神经元健康度拉高
        for nrn in self.neurons:
            nrn._h.data.fill_(0.9)

        # 运行几步让 EMA 健康度上升并满足唤醒条件
        for _ in range(10):
            self.sim.step()
            if self.sleep_wake.state == 'WAKE':
                break
        self.assertEqual(self.sleep_wake.state, 'WAKE',
                         "健康度恢复后应唤醒回到 WAKE")

    def test_memory_replay_during_sleep(self):
        """
        梦境重放测试：
        - 在睡眠状态下验证神经元膜电位因重放噪声注入而发生变化
        """
        # 触发睡眠
        fatigue = self.mod_sys.modulators[MODULATOR_FATIGUE]
        fatigue.sources.append((fatigue.release_position, 1.0))
        for _ in range(10):
            self.sim.step()
            if self.sleep_wake.state == 'SLEEP':
                break
        self.assertEqual(self.sleep_wake.state, 'SLEEP')

        # 记录睡眠前三个神经元的膜电位
        u_before = [
            nrn.get_internal_state()['u'].clone()
            for nrn in self.neurons[:3]
        ]

        # 继续运行几步睡眠仿真（确保 replay_step 被多次调用）
        for _ in range(5):
            self.sim.step()

        # 记录睡眠后的膜电位
        u_after = [
            nrn.get_internal_state()['u'].clone()
            for nrn in self.neurons[:3]
        ]

        # 验证至少有一个神经元的膜电位发生了变化
        any_changed = any(
            not torch.allclose(before, after, atol=1e-5)
            for before, after in zip(u_before, u_after)
        )
        self.assertTrue(any_changed,
                        "睡眠期间梦境重放应使神经元膜电位发生变化")

    def test_synaptic_homeostasis_during_sleep(self):
        """
        突触稳态测试：
        - 在睡眠状态下对突触强度执行多次基于基线的衰减
        - 验证未被保护的突触强度明显衰减，被保护的突触免于衰减
        """
        # 创建突触稳态模块（加速参数以便观察）
        homeostasis = SynapticHomeostasis(
            connections=self.connections,
            decay_factor=0.9,
            baseline=0.01,
            move_rate=0.2,
        )

        # 标记第一个突触为保护状态
        self.synapses[0].protected.data.fill_(True)

        # 记录所有突触的初始强度
        initial_strengths = [
            syn.strength.clone() for syn in self.synapses
        ]

        # 触发睡眠
        fatigue = self.mod_sys.modulators[MODULATOR_FATIGUE]
        fatigue.sources.append((fatigue.release_position, 1.0))
        for _ in range(5):
            self.sim.step()
            if self.sleep_wake.state == 'SLEEP':
                break
        self.assertEqual(self.sleep_wake.state, 'SLEEP')

        # 睡眠期间连续执行多次稳态重归一化
        for _ in range(20):
            self.sim.step()  # 内部更新突触动力学（STDP 影响较小）
            homeostasis.renormalize()  # 施加稳态衰减

        # 验证结果
        for i, syn in enumerate(self.synapses):
            final_strength = syn.strength.detach().cpu().numpy()
            init_strength = initial_strengths[i].detach().cpu().numpy()
            if i == 0:
                # 被保护突触强度应基本不变（允许微小 STDP 波动）
                self.assertTrue(
                    (final_strength - init_strength).std() < 0.05,
                    f"被保护突触 {i} 强度不应发生显著变化"
                )
            else:
                # 未保护突触应向基线（0.01）方向衰减
                self.assertLess(
                    final_strength.mean(), init_strength.mean() * 0.8,
                    f"未保护突触 {i} 强度应明显衰减"
                )

    def test_wake_restores_normal_operation(self):
        """
        唤醒恢复测试：
        - 经历完整的睡眠-觉醒周期后，验证网络恢复正常运行：
          a. 外部输入恢复接收
          b. 神经发生概率恢复正常
          c. 梦境重放和树突棘重塑停止执行
        """
        # ---- 进入睡眠 ----
        fatigue = self.mod_sys.modulators[MODULATOR_FATIGUE]
        fatigue.sources.append((fatigue.release_position, 1.0))
        for _ in range(10):
            self.sim.step()
            if self.sleep_wake.state == 'SLEEP':
                break
        self.assertEqual(self.sleep_wake.state, 'SLEEP')

        # ---- 手动恢复健康度并唤醒 ----
        for nrn in self.neurons:
            nrn._h.data.fill_(0.9)
        for _ in range(10):
            self.sim.step()
            if self.sleep_wake.state == 'WAKE':
                break
        self.assertEqual(self.sleep_wake.state, 'WAKE',
                         "应唤醒回到 WAKE 状态")

        # ---- 验证唤醒后属性 ----
        # a. 外部输入不再被抑制
        self.assertFalse(self.sleep_wake.input_suppressed,
                         "唤醒后外部输入应恢复正常接收")
        # b. 神经发生概率倍率恢复
        self.assertEqual(self.sleep_wake.neurogenesis_factor, 1.0,
                         "唤醒后神经发生概率应恢复正常")
        # c. 梦境重放与树突棘重塑应停止执行（WAKE 状态下不会再被调用）
        before_replay_stats = self.replay.get_replay_statistics()
        before_spine_stats = self.spine_remodel.get_remodel_statistics()
        # 运行额外一步，统计信息不应变化
        self.sim.step()
        after_replay_stats = self.replay.get_replay_statistics()
        after_spine_stats = self.spine_remodel.get_remodel_statistics()
        self.assertEqual(before_replay_stats, after_replay_stats,
                         "唤醒后梦境重放统计不应更新")
        self.assertEqual(before_spine_stats, after_spine_stats,
                         "唤醒后树突棘重塑统计不应更新")