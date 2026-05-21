# tests/test_glia.py
"""单元测试：胶质细胞模块 (Astrocyte, Microglia, GlialNetwork)"""

import unittest
import torch
import itertools

from genesis_core.neuron import LIFNeuron
from genesis_core.synapse import STDPSynapse
from genesis_core.glia import Astrocyte, Microglia, GlialNetwork

class TestAstrocyte(unittest.TestCase):
    def setUp(self):
        self.nrn = LIFNeuron(num_neurons=4)
        self.syn = STDPSynapse(num_synapses=4)
        # 简单跑一步，让活动迹不为零
        self.nrn(torch.randn(1, 1, 4) * 0.5)
        self.syn(torch.ones(1, 4), torch.ones(1, 4),
                 torch.randn(1, 4, 1), torch.randn(1, 4, 1))

    def test_astrocyte_perceive_and_modulate(self):
        """星形胶质细胞感知活动并通过信号反馈调节"""
        astro = Astrocyte(
            astrocyte_id=1,
            neuron_groups=[(self.nrn, [0,1,2,3])],
            synapse_groups=[(self.syn, [0,1,2,3])],
            position=(0.0, 0.0, 0.0)
        )
        # 感知活动
        activity = astro.perceive()
        self.assertGreaterEqual(activity, 0.0)

        # 直接拉高钙浓度以强制触发信号释放 (绕过 signal_threshold)
        astro.ca = 0.9
        signal_level = astro.release_signal()
        self.assertGreater(signal_level, 0.0)

        # 施加信号调制，验证调制原已被注入 neuron
        astro.modulate(signal_level)
        self.assertGreater(
            self.nrn._modulator_concentrations.get("local_glial_bias", 0.0),
            0
        )

class TestMicroglia(unittest.TestCase):
    def setUp(self):
        self.syn = STDPSynapse(num_synapses=8)
        # 让突触先跑一步，确保内部状态非零
        self.syn(torch.ones(1, 8), torch.ones(1, 8),
                 torch.randn(1, 8, 1), torch.randn(1, 8, 1))

    def test_microglia_monitor_and_prune(self):
        """小胶质细胞监测活动并标记低活动突触为修剪"""
        mg = Microglia(synapse=self.syn, pruning_threshold=0.9, check_interval=1)
        # 强制所有活动指标非常低
        mg.activity_avg.fill_(0.0)
        mg._execute_pruning()
        # 应当有至少一个突触被标记为修剪
        self.assertTrue(self.syn.prune_flag.any())

    def test_microglia_protection(self):
        """小胶质细胞保护高活动突触"""
        mg = Microglia(synapse=self.syn, protection_threshold=0.05, check_interval=1)
        mg.activity_avg.fill_(0.5)
        mg._execute_pruning()
        self.assertTrue(self.syn.protected.any())


class TestGlialNetwork(unittest.TestCase):
    def setUp(self):
        self.nrn1 = LIFNeuron(num_neurons=2)
        self.nrn2 = LIFNeuron(num_neurons=2)
        self.syn = STDPSynapse(num_synapses=4)

        self.astro1 = Astrocyte(
            astrocyte_id=1,
            neuron_groups=[(self.nrn1, [0,1])],
            synapse_groups=[(self.syn, [0,1])],
            position=(0.0, 0.0, 0.0)
        )
        self.astro2 = Astrocyte(
            astrocyte_id=2,
            neuron_groups=[(self.nrn2, [0,1])],
            synapse_groups=[(self.syn, [2,3])],
            position=(0.5, 0.0, 0.0)
        )
        self.micro = Microglia(synapse=self.syn, check_interval=1)
        self.gnet = GlialNetwork(astrocytes=[self.astro1, self.astro2],
                                 microglias=[self.micro],
                                 gap_junction_radius=1.0,
                                 diffusion_rate=0.01)

    def test_glial_network_step(self):
        """胶质互联网能正常步进"""
        # 先让神经元产生一些活动
        self.nrn1(torch.randn(1, 1, 2) * 0.5)
        self.nrn2(torch.randn(1, 1, 2) * 0.5)
        self.gnet.step()
        # 钙浓度应该有非负值
        self.assertGreaterEqual(self.astro1.ca, 0.0)
        self.assertGreaterEqual(self.astro2.ca, 0.0)

    def test_gap_junction_diffusion(self):
        """缝隙连接能进行钙波扩散"""
        self.astro1.ca = 1.0
        self.astro2.ca = 0.0
        self.gnet._diffuse_calcium()
        # astro2 应该从 astro1 扩散得到部分钙
        self.assertGreater(self.astro2.ca, 0.0)


if __name__ == '__main__':
    unittest.main()