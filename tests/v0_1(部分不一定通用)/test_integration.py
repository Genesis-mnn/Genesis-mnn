# test_integration.py
import unittest
import torch
import tempfile
import os
from genesis_core.simulator import create_default_simulator
from genesis_core.life import GenesisNetwork, LifeSaver, LifeLoader, SynapticConnection
from genesis_core import MorphoNeuron, ResonanceSynapse

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # 创建包含胶质网络的默认模拟器
        self.simulator = create_default_simulator(
            num_neurons=10,
            connection_prob=0.2,
            with_glia=True,
            batch_size=1
        )

    def test_full_simulation_50_steps(self):
        """运行50步仿真并验证网络正常演化。"""
        sim = self.simulator
        snap = None
        for t in range(50):
            # 提供轻微外部输入以保证活动
            ext = [torch.randn(1, 10) * 0.1]
            #snap = sim.step(external_input=ext)
            snap = sim.step()
            self.assertIn('mean_health', snap)
            self.assertIn('mean_firing_rate', snap)
        # 健康度不应崩溃
        self.assertGreater(snap['mean_health'], 0.0, "Mean health should be positive after simulation")
        # 调制场系统应有活动（至少一个调制原产生了源）
        mod_sys = sim.modulator_system
        self.assertTrue(any(len(mod.sources) > 0 for mod in mod_sys.modulators.values()),
                        "At least one modulator should have active sources")
        # 胶质网络正常运行（微胶质步进）
        if sim.glia is not None:
            self.assertTrue(all(m.step_counter > 0 for m in sim.glia.microglias),
                            "Microglia should have stepped at least once")

    def test_save_and_load_consistency(self):
        """保存为 .life 文件并重新加载，验证状态一致性。"""
        sim = self.simulator
        # 预热网络
        for _ in range(20):
            sim.step(external_input=[torch.randn(1, 10) * 0.1])

        # 构建 GenesisNetwork 容器
        network = GenesisNetwork()
        network.neurons = list(sim.neurons)
        network.connections = [
            SynapticConnection(
                synapse=conn.synapse,
                pre_indices=conn.pre_indices.clone(),
                post_indices=conn.post_indices.clone(),
                name=conn.name
            ) for conn in sim.connections
        ]
        network.glia = sim.glia
        network.modulators = sim.modulator_system

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.life', delete=False) as tmp:
            tmppath = tmp.name
        try:
            LifeSaver.save(network, tmppath)
            loaded_net = LifeLoader.load(tmppath, device=torch.device('cpu'))

            # 比较神经元状态
            for orig_nrn, load_nrn in zip(network.neurons, loaded_net.neurons):
                orig_state = orig_nrn.get_internal_state()
                load_state = load_nrn.get_internal_state()
                for key in ['u', 'h', 'r']:
                    diff = (orig_state[key] - load_state[key]).abs().max().item()
                    self.assertLess(diff, 1e-4,
                                    f"Neuron state '{key}' differs by {diff}")

            # 比较突触状态
            for orig_conn, load_conn in zip(network.connections, loaded_net.connections):
                orig_syn = orig_conn.synapse.get_synaptic_state()
                load_syn = load_conn.synapse.get_synaptic_state()
                for key in ['strength', 'resid_Ca']:
                    diff = (orig_syn[key] - load_syn[key]).abs().max().item()
                    self.assertLess(diff, 1e-4,
                                    f"Synapse state '{key}' differs by {diff}")
        finally:
            os.unlink(tmppath)


if __name__ == '__main__':
    unittest.main()