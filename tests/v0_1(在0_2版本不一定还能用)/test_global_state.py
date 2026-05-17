# test_global_state.py
import unittest
import torch
from genesis_core.global_state import GlobalStateMonitor, GlobalStateScheduler
from genesis_core.neuron import LIFNeuron
from genesis_core.synapse import STDPSynapse
from genesis_core.modulator import create_default_modulators
from genesis_core import (
    GLOBAL_REWARD_ERROR, GLOBAL_SATIETY, GLOBAL_NOVELTY,
    GLOBAL_STRESS, GLOBAL_ENERGY_DEFICIT
)

class TestGlobalState(unittest.TestCase):
    def setUp(self):
        self.neuron = LIFNeuron(num_neurons=10, num_dendrites=1)
        self.synapse = STDPSynapse(num_synapses=10)
        self.neuron.reset_state(1)
        self.synapse.reset_state(1)

        # 产生一些活动
        for _ in range(10):
            inp = torch.randn(1, 1, 10) * 0.5
            self.neuron(inp)

        self.monitor = GlobalStateMonitor(
            neurons=[self.neuron],
            connections=[self.synapse],
            optimal_rate=0.1,
            ema_alpha=0.5
        )
        self.mod_system = create_default_modulators()
        self.scheduler = GlobalStateScheduler(self.monitor, self.mod_system)

    def test_monitor_metrics(self):
        """测试五种全局指标的计算。"""
        state = self.monitor.step()
        for key in [GLOBAL_REWARD_ERROR, GLOBAL_SATIETY, GLOBAL_NOVELTY,
                    GLOBAL_STRESS, GLOBAL_ENERGY_DEFICIT]:
            self.assertIn(key, state)
            self.assertIsInstance(state[key], float)

    def test_monitor_empty_neurons(self):
        """无神经元时，指标应全为0。"""
        empty_mon = GlobalStateMonitor(neurons=[], connections=[])
        state = empty_mon.step()
        for v in state.values():
            self.assertEqual(v, 0.0)

    def test_scheduler_step(self):
        """测试调度器的“监测→调制”闭环。"""
        self.mod_system.register_target(self.neuron, (0.0, 0.0, 0.0))
        global_state = self.scheduler.step()
        self.assertTrue(global_state)
        # 调制系统应有源产生
        reward_mod = self.mod_system.modulators['奖赏信号']
        self.assertGreater(len(reward_mod.sources), 0,
                           "Reward modulator should have sources after scheduler step")

    def test_reset(self):
        """测试监视器重置功能。"""
        self.monitor.step()
        self.monitor.reset()
        self.assertIsNone(self.monitor._prev_firing_rates)
        self.assertIsNone(self.monitor._ema_reward_error)


if __name__ == '__main__':
    unittest.main()