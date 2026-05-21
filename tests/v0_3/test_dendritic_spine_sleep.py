# tests/v0_3/test_dendritic_spine_sleep.py

import unittest
import torch
from genesis_core.neuron import LIFNeuron
from genesis_core.dendritic_spine_sleep import DendriticSpineSleepRemodeling


class TestDendriticSpineSleepRemodeling(unittest.TestCase):
    """DendriticSpineSleepRemodeling 模块的单元测试。"""

    def setUp(self):
        """准备一个包含多个树突棘的神经元实例，并将钙浓度设为中性值。"""
        self.neuron = LIFNeuron(num_neurons=1, num_dendrites=5)
        self.neuron.reset_state(batch_size=1)
        with torch.no_grad():
            self.neuron._spine_ca.fill_(0.5)

    def test_remodel_lowers_thresholds(self):
        """测试重塑步骤能够触发树突棘数量的变化（新生或回缩）。"""
        initial_dendrites = self.neuron.num_dendrites

        # 修改一个树突棘的钙浓度为高值，以触发新生
        with torch.no_grad():
            self.neuron._spine_ca[0, 0, 0] = 1.0  # batch=0, dendrite=0, neuron=0

        remodel = DendriticSpineSleepRemodeling(
            neurons=[self.neuron],
            spine_birth_threshold=0.5,
            spine_retraction_threshold=0.1,
            max_total_spines=32,
            min_total_spines=1,
        )

        remodel.remodel_step()

        final_dendrites = self.neuron.num_dendrites
        self.assertNotEqual(
            initial_dendrites, final_dendrites,
            "重塑步骤应导致树突棘数量的变化。"
        )

    def test_remodel_statistics(self):
        """测试 get_remodel_statistics 返回包含预期键的字典。"""
        with torch.no_grad():
            self.neuron._spine_ca[0, 0, 0] = 1.0

        remodel = DendriticSpineSleepRemodeling(neurons=[self.neuron])
        remodel.remodel_step()

        stats = remodel.get_remodel_statistics()

        self.assertIn('new_spines', stats)
        self.assertIn('retracted_spines', stats)
        self.assertIn('net_change', stats)

        self.assertEqual(
            stats['net_change'],
            stats['new_spines'] - stats['retracted_spines'],
            "net_change 应等于 new_spines 减去 retracted_spines"
        )