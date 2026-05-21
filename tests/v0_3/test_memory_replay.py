# tests/v0_3/test_memory_replay.py

import unittest
import torch
from genesis_core.neuron import LIFNeuron
from genesis_core.genesis_core import MODULATOR_STRESS, MODULATOR_FATIGUE
from genesis_core.memory_replay import MemoryReplay


class QuietLIFNeuron(LIFNeuron):
    """用于测试的 LIF 神经元变体，关闭了内在噪声，以便精确测量重放注入。"""

    def _update_soma(self, v_dend, modulators):
        self._refractory_counter = torch.clamp(self._refractory_counter - 1, min=0)
        in_ref = (self._refractory_counter > 0).float()
        not_ref = 1.0 - in_ref

        I_homeo = -0.1 * (self._h - self.rest_h.unsqueeze(0))
        I_noise = 0.0  # 关闭内在噪声
        I_int = I_homeo + I_noise
        if MODULATOR_STRESS in modulators:
            I_int += 0.1 * modulators[MODULATOR_STRESS]
        if MODULATOR_FATIGUE in modulators:
            I_int -= 0.05 * modulators[MODULATOR_FATIGUE]

        I_total = v_dend + I_int
        du = (-(self._u - self.u_rest) + self.R.unsqueeze(0) * I_total) / self.tau.unsqueeze(0)
        self._u = not_ref * (self._u + du) + in_ref * self._u

        spike = not_ref * (self._u >= self.threshold.unsqueeze(0)).float()
        self._u = (1.0 - spike) * self._u + spike * self.v_reset
        self._refractory_counter = torch.where(
            spike > 0,
            torch.full_like(self._refractory_counter, self.refractory_period),
            self._refractory_counter
        )
        return spike


class TestMemoryReplay(unittest.TestCase):
    """MemoryReplay 模块的单元测试。"""

    def test_replay_noise_injection(self):
        """创建 MemoryReplay 并调用 replay_step()，验证神经元膜电位发生了变化（噪声注入）。"""
        neuron = QuietLIFNeuron(num_neurons=1, num_dendrites=1)
        neuron.reset_state(batch_size=1)
        initial_u = neuron._u.clone()

        replay = MemoryReplay(
            neurons=[neuron],
            connections=[],
            noise_amplitude=0.05
        )
        replay.replay_step()

        final_u = neuron._u
        self.assertFalse(
            torch.allclose(initial_u, final_u),
            "膜电位在重放步骤后应发生变化（噪声注入）。"
        )

    def test_healthy_neurons_get_stronger_noise(self):
        """健康度高的神经元获得更强的噪声注入，膜电位变化幅度更大。"""
        neuron_high = QuietLIFNeuron(num_neurons=1, num_dendrites=1, rest_h=0.9)
        neuron_low = QuietLIFNeuron(num_neurons=1, num_dendrites=1, rest_h=0.1)
        neuron_high.reset_state(1)
        neuron_low.reset_state(1)

        with torch.no_grad():
            neuron_high._h[0, 0] = 0.9
            neuron_low._h[0, 0] = 0.1

        initial_u_high = neuron_high._u.clone()
        initial_u_low = neuron_low._u.clone()

        replay = MemoryReplay(
            neurons=[neuron_high, neuron_low],
            connections=[],
            noise_amplitude=0.05
        )
        replay.replay_step()

        delta_high = (neuron_high._u - initial_u_high).abs().max().item()
        delta_low = (neuron_low._u - initial_u_low).abs().max().item()

        self.assertGreater(
            delta_high, delta_low,
            "健康度高的神经元膜电位变化幅度应大于健康度低的。"
        )

    def test_low_health_neurons_excluded(self):
        """健康度低于阈值的神经元不会被注入噪声，膜电位保持不变。"""
        neuron = QuietLIFNeuron(num_neurons=1, num_dendrites=1, rest_h=0.1)
        neuron.reset_state(1)

        with torch.no_grad():
            neuron._h[0, 0] = 0.1

        initial_u = neuron._u.clone()

        replay = MemoryReplay(
            neurons=[neuron],
            connections=[],
            noise_amplitude=0.05
        )
        replay.replay_step()

        # 使用低噪声神经元后，膜电位应完全不变（重放注入被跳过，内在噪声也已关闭）
        self.assertTrue(
            torch.allclose(initial_u, neuron._u, atol=1e-6),
            "低健康度神经元的膜电位在重放步骤后应保持不变（排除重放注入）。"
        )

    def test_get_replay_statistics(self):
        """运行 replay_step() 后调用 get_replay_statistics()，返回字典包含必要的键且值合理。"""
        neurons = [
            QuietLIFNeuron(num_neurons=1, num_dendrites=1) for _ in range(10)
        ]
        for nrn in neurons:
            nrn.reset_state(1)

        replay = MemoryReplay(
            neurons=neurons,
            connections=[],
            noise_amplitude=0.05
        )
        replay.replay_step()
        stats = replay.get_replay_statistics()

        self.assertIn('mean_firing_rate', stats)
        self.assertIn('participation_rate', stats)

        mean_fr = stats['mean_firing_rate']
        part_rate = stats['participation_rate']
        self.assertGreaterEqual(mean_fr, 0.0)
        self.assertLessEqual(mean_fr, 1.0)
        self.assertGreaterEqual(part_rate, 0.0)
        self.assertLessEqual(part_rate, 1.0)