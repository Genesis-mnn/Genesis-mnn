# tests/v0_3/test_v0_3_integration.py
import unittest
import torch

# 从公开 API 导入核心组件（假设 genesis_core 包已正确安装或在 PYTHONPATH 中）
from genesis_core import (
    LIFNeuron,
    STDPSynapse,
    Astrocyte,
    Microglia,
    GlialNetwork,
    ModulatorSystem,
    create_default_modulators,
    GlobalStateMonitor,
    GlobalStateScheduler,
    GenesisSimulator,
    MODULATOR_FATIGUE,
)
from genesis_core.sleep_wake_cycle import SleepWakeCycle, SynapticHomeostasis


class TestV03Integration(unittest.TestCase):
    """v0.3 睡眠-觉醒周期与模拟器集成测试"""

    def setUp(self):
        """构建包含10个 LIF 神经元、5个 STDP 突触、胶质网络、全局调度器、
        睡眠-觉醒状态机以及突触稳态组件的最小完整网络。"""
        # ---------- 1. 神经元 ----------
        self.neurons = []
        for _ in range(10):
            nrn = LIFNeuron(
                num_neurons=1,
                num_dendrites=1,
                tau=10.0,
                v_threshold=100.0,   # 高阈值避免发放干扰测试
                v_reset=0.0,
                rest_h=0.7,
            )
            nrn.noise_scale = 0.0     # 消除噪声，使行为确定
            self.neurons.append(nrn)

        # ---------- 2. 突触连接 ----------
        self.connections = []
        # 简单线性连接：0->5, 1->6, 2->7, 3->8, 4->9
        for i in range(5):
            syn = STDPSynapse(
                num_synapses=1,
                A_plus=0.0,
                A_minus=0.0,
                lr=0.0,                # 禁用 STDP 以隔离稳态测试
                tau_pre=20.0,
                tau_post=20.0,
            )
            self.connections.append((syn, self.neurons[i], self.neurons[i+5]))

        # ---------- 3. 调制场系统 ----------
        self.modulator_system = create_default_modulators()

        # ---------- 4. 全局状态监测与调度 ----------
        syn_instances = [conn[0] for conn in self.connections]
        self.monitor = GlobalStateMonitor(
            neurons=self.neurons,
            connections=syn_instances,
            ema_alpha=0.1,
        )
        self.global_scheduler = GlobalStateScheduler(
            monitor=self.monitor,
            modulator_system=self.modulator_system,
        )

        # ---------- 5. 星形胶质细胞（包裹第一个神经元和第一个突触） ----------
        self.astro = Astrocyte(
            astrocyte_id=0,
            neuron_groups=[(self.neurons[0], [0])],
            synapse_groups=[(self.connections[0][0], [0])],
            position=(0.0, 0.0, 0.0),
        )

        # ---------- 6. 小胶质细胞（至少一个，满足 glia 非空要求） ----------
        self.micro = Microglia(synapse=self.connections[0][0])

        # ---------- 7. 胶质网络 ----------
        self.glia = GlialNetwork(
            astrocytes=[self.astro],
            microglias=[self.micro],
        )

        # ---------- 8. 睡眠-觉醒周期 ----------
        self.swc = SleepWakeCycle(
            modulator_system=self.modulator_system,
            global_state_scheduler=self.global_scheduler,
            glia=self.glia,
            sleep_threshold=0.7,
            wake_health_threshold=0.75,
            min_sleep_steps=2,          # 加速唤醒测试
        )

        # ---------- 9. 突触稳态（较快衰减便于观察） ----------
        self.homeostasis = SynapticHomeostasis(
            connections=self.connections,
            decay_factor=0.9,
            baseline=0.01,
            move_rate=0.1,
        )

        # ---------- 10. 统一模拟器 ----------
        self.sim = GenesisSimulator(
            neurons=self.neurons,
            connections=self.connections,
            glia=self.glia,
            modulator_system=self.modulator_system,
            global_scheduler=self.global_scheduler,
            sleep_wake_cycle=self.swc,
            batch_size=1,
        )
        self.sim.reset(1)

    # ========================================================================
    # 测试用例
    # ========================================================================

    def test_sleep_wake_cycle_integrated_in_simulator(self):
        """创建小型网络并注入所有组件，运行10步仿真，验证初始状态为WAKE；
        手动提高疲劳累积子浓度后，再运行几步应切换为SLEEP。"""
        # 初始10步内应保持 WAKE
        for _ in range(10):
            self.sim.step()
            self.assertEqual(self.swc.state, 'WAKE',
                             "初始状态应为 WAKE")

        # 手动设置疲劳累积子浓度超过阈值（0.7）
        fatigue_mod = self.modulator_system.modulators[MODULATOR_FATIGUE]
        fatigue_mod.sources.append(((0.0, 0.0, 0.0), 0.8))

        # 再运行一步应触发睡眠
        self.sim.step()
        self.assertEqual(self.swc.state, 'SLEEP',
                         "疲劳累积子浓度超过阈值后应切换为 SLEEP")

    def test_sleep_state_blocks_external_input(self):
        """先触发睡眠状态，然后在睡眠状态下传入外部输入，验证输入被忽略，
        网络仅靠内部噪声驱动（本例噪声已关闭，因此膜电位应无变化）。"""
        # 1. 在 WAKE 状态下验证外部输入有效
        self.sim.reset(1)
        input_tensor = [torch.tensor([1.0]) for _ in self.neurons]
        self.sim.step(external_input=input_tensor)
        u_wake = self.neurons[0].get_internal_state()['u'].item()
        self.assertGreater(u_wake, 0.0,
                           "WAKE 状态下外部输入应导致膜电位升高")

        # 2. 触发 SLEEP
        fatigue_mod = self.modulator_system.modulators[MODULATOR_FATIGUE]
        fatigue_mod.sources.append(((0.0, 0.0, 0.0), 0.8))
        self.sim.step()
        self.assertEqual(self.swc.state, 'SLEEP')

        # 3. 重置网络状态（保持睡眠），在睡眠状态下传入相同输入
        self.sim.reset(1)
        u_before = self.neurons[0].get_internal_state()['u'].item()
        self.sim.step(external_input=input_tensor)
        u_after = self.neurons[0].get_internal_state()['u'].item()

        # 膜电位应无显著变化（输入被忽略）
        self.assertAlmostEqual(u_before, u_after, delta=0.01,
                               msg="SLEEP 状态下外部输入应被忽略，膜电位不变")
        self.assertTrue(self.swc.input_suppressed,
                        "睡眠标志 input_suppressed 应为 True")

    def test_health_recovery_triggers_wake(self):
        """先触发睡眠状态并确保睡眠次数满足最小要求，手动将全局健康度恢复到0.75以上，
        再运行一步应切换回 WAKE。"""
        # 进入 SLEEP
        fatigue_mod = self.modulator_system.modulators[MODULATOR_FATIGUE]
        fatigue_mod.sources.append(((0.0, 0.0, 0.0), 0.8))
        self.sim.step()
        self.assertEqual(self.swc.state, 'SLEEP')

        # 运行 min_sleep_steps 步，满足最短睡眠时间
        for _ in range(self.swc.min_sleep_steps):
            self.sim.step()
        self.assertGreaterEqual(self.swc._sleep_counter,
                                self.swc.min_sleep_steps)

        # 手动将全局健康度提升至 0.75 以上
        for nrn in self.neurons:
            nrn._h.fill_(0.9)
        # 确保 EMA 也立即反映高健康度（跳过平滑）
        self.monitor._ema_satiety = 0.9

        # 再运行一步应唤醒
        self.sim.step()
        self.assertEqual(self.swc.state, 'WAKE',
                         "健康度恢复且睡眠足够久后应唤醒回 WAKE")

    def test_synaptic_homeostasis_during_sleep(self):
        """记录所有突触初始强度，触发睡眠后运行足够步数并执行突触稳态衰减，
        验证突触强度向基线方向（0.01）衰减。"""
        # 记录初始强度
        init_strengths = []
        for conn in self.connections:
            syn = conn[0]
            init_strengths.append(syn.strength.mean().item())

        # 触发 SLEEP
        fatigue_mod = self.modulator_system.modulators[MODULATOR_FATIGUE]
        fatigue_mod.sources.append(((0.0, 0.0, 0.0), 0.8))
        self.sim.step()
        self.assertEqual(self.swc.state, 'SLEEP')

        # 在睡眠期间运行多步，每步施加稳态衰减
        n_sleep_steps = 20
        for _ in range(n_sleep_steps):
            self.sim.step()
            self.homeostasis.renormalize()

        # 检查所有突触强度是否衰减（初始 > 基线）
        for i, conn in enumerate(self.connections):
            syn = conn[0]
            final_strength = syn.strength.mean().item()
            self.assertLess(
                final_strength, init_strengths[i],
                f"突触 {i} 强度应衰减：初始 {init_strengths[i]:.4f}，"
                f"最终 {final_strength:.4f}"
            )
            # 进一步验证朝向基线变化
            self.assertLess(
                abs(final_strength - 0.01),
                abs(init_strengths[i] - 0.01),
                f"突触 {i} 强度应更接近基线 0.01"
            )


if __name__ == '__main__':
    unittest.main()