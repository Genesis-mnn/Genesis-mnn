# tests/v0_2/test_neurogenesis.py
import unittest
import torch
import random
from genesis_core.neurogenesis import (NeurogenesisPool, check_apoptosis,
                                       NeurogenesisScheduler)
from genesis_core.neuron import LIFNeuron
from genesis_core.synapse import STDPSynapse
from genesis_core.glia import GlialNetwork, Astrocyte


class TestNeurogenesis(unittest.TestCase):
    """v0.2 神经发生与凋亡模块核心功能测试。"""

    def setUp(self):
        self.device = torch.device('cpu')
        random.seed(42)
        torch.manual_seed(42)

    def test_neuron_apoptosis_by_health_collapse(self):
        """健康度崩溃（h < 0.05）应触发神经元凋亡标记。"""
        neuron = LIFNeuron(num_neurons=1).to(self.device)
        neuron._h.data.fill_(0.01)
        neurons = [neuron]
        connections = []   # 空连接列表
        silent_counters = {}
        candidates = check_apoptosis(neurons, connections, silent_counters,
                                     silence_window=10000)
        self.assertIn(neuron, candidates, "健康度崩溃的神经元应被标记为可凋亡")

    def test_neuron_apoptosis_by_long_silence(self):
        """长期静默应触发凋亡标记。"""
        neuron = LIFNeuron(num_neurons=1).to(self.device)
        neurons = [neuron]
        connections = []
        silent_counters = {id(neuron): 10000}
        candidates = check_apoptosis(neurons, connections, silent_counters,
                                     silence_window=10000)
        self.assertIn(neuron, candidates, "长期静默的神经元应被标记为可凋亡")

    def test_neurogenesis_pool_spawn(self):
        """神经干细胞池生成新神经元，内部状态应为默认初始值。"""
        pool = NeurogenesisPool(neuron_class=LIFNeuron, param_variability=0.0)
        neuron = pool.spawn_neuron()

        self.assertIsInstance(neuron, LIFNeuron)
        state = neuron.get_internal_state()
        h = state['h'][0, 0].item()
        u = state['u'][0, 0].item()
        r = state['r']
        r_norm = r[0, 0, :].norm().item()

        self.assertAlmostEqual(h, 0.8, delta=0.01)
        self.assertAlmostEqual(u, 0.0, delta=0.01)
        self.assertAlmostEqual(r_norm, 0.5, delta=0.05)

    def test_scheduler_step_without_crash(self):
        """调度器在小型网络中连续运行 50 步不应崩溃。"""
        num_neurons = 10
        neuron_instances = [LIFNeuron(num_neurons=1).to(self.device)
                            for _ in range(num_neurons)]

        # 构建连接：每个元组 (synapse, pre_neuron, post_neuron)
        connections = []
        for i in range(num_neurons):
            for j in [(i + 1) % num_neurons, (i + 2) % num_neurons]:
                syn = STDPSynapse(num_synapses=1).to(self.device)
                connections.append((syn, neuron_instances[i], neuron_instances[j]))

        # 胶质网络
        neuron_groups = [(n, [0]) for n in neuron_instances]
        synapse_groups = [(syn, [0]) for syn, _, _ in connections]
        astro = Astrocyte(astrocyte_id=0,
                          neuron_groups=neuron_groups,
                          synapse_groups=synapse_groups,
                          position=(0.0,))
        glia = GlialNetwork(astrocytes=[astro], microglias=[])

        scheduler = NeurogenesisScheduler(
            neurons=neuron_instances,
            connections=connections,
            glia=glia,
            p_neurogenesis_base=0.0001,
            silence_window=10000,
        )

        for step in range(50):
            spike_dict = {}
            for idx, nrn in enumerate(neuron_instances):
                x = torch.randn(1, 1, device=self.device) * 0.5
                spike = nrn(x)
                spike_dict[id(nrn)] = float(spike.item() > 0.0)
            scheduler.record_spikes(spike_dict)
            _ = scheduler.step()

        stats = scheduler.get_statistics()
        self.assertIn('active_neurons', stats)
        self.assertGreaterEqual(stats['active_neurons'], 1)
        self.assertIn('synaptic_connections', stats)
        self.assertIn('global_mean_health', stats)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()