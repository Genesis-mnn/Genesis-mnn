# tests/v0_2/test_v0_2_learning_closed_loop.py
import unittest
import torch
import numpy as np
from genesis_core.simulator import create_default_simulator
from genesis_core.glia import Astrocyte, GlialNetwork
from genesis_core.learning import IntrinsicLearningScheduler
from genesis_core.global_state import GlobalStateMonitor, GlobalStateScheduler
from genesis_core.modulator import create_default_modulators, ModulatorSystem
from genesis_core.neuron import LIFNeuron
from genesis_core.synapse import STDPSynapse
from genesis_core.utils import create_layer, connect


class TestClosedLoopLearning(unittest.TestCase):
    """验证 v0.2 中“监测 → 调制 → 学习”的完整闭环。"""

    def test_closed_loop_learning_executes(self):
        """使用 create_default_simulator 构建小型网络，运行仿真，验证学习统计信息。"""
        sim = create_default_simulator(
            num_neurons=10,
            with_glia=True,
            batch_size=1,
            device=torch.device('cpu')
        )

        self.assertIsNotNone(sim.glia)
        self.assertIsNotNone(sim.global_scheduler)

        scheduler = IntrinsicLearningScheduler(
            glial_network=sim.glia,
            global_state_scheduler=sim.global_scheduler
        )
        sim.intrinsic_scheduler = scheduler

        for step in range(50):
            ext_input = torch.randn(1, 10) * 0.5   # 外部输入已适配新 simulator
            #snapshot = sim.step(external_input=[ext_input])
            snapshot = sim.step()  # 不传外部输入，网络自发运行
            self.assertIn('intrinsic_learning', snapshot)
            learn_stats = snapshot['intrinsic_learning']
            self.assertIsInstance(learn_stats, dict)
            self.assertIn('pruned_synapses', learn_stats)
            self.assertIn('enhanced_synapses', learn_stats)

    def test_novelty_driven_learning_trigger(self):
        """手动构建网络，通过控制新奇度验证学习触发并修改突触状态。"""
        device = torch.device('cpu')

        layer = create_layer(num_neurons=10, neuron_type='LIF', name='test_layer')
        neuron_group = layer.neuron_group

        conn = connect(
            src=layer, dst=layer,
            rule='random', p=0.5,
            synapse_class=STDPSynapse,
            synapse_kwargs={
                'pool_total': 1.0, 'k_cyc': 0.01, 'k_rp': 0.1,
                'p_base': 0.2,
                'A_plus': 0.005, 'A_minus': 0.005,
                'tau_pre': 20.0, 'tau_post': 20.0,
            },
            name='test_connection'
        )
        syn = conn.synapse

        astro = Astrocyte(
            astrocyte_id=0,
            neuron_groups=[(neuron_group, list(range(10)))],
            synapse_groups=[(syn, list(range(5)))],
            position=(0.0, 0.0, 0.0),
            ca_tau=2.0, ca_beta=0.5,
            signal_threshold=0.3,
            neuron_mod_gain=0.05,
            synapse_mod_gain=0.1,
        )
        glia = GlialNetwork(astrocytes=[astro], microglias=[])

        monitor = GlobalStateMonitor(
            neurons=[neuron_group],
            connections=[syn],
            ema_alpha=1.0
        )
        mod_sys = create_default_modulators()
        mod_sys.register_target(neuron_group, (0.0, 0.0, 0.0))
        mod_sys.register_target(syn, (0.0, 0.0, 0.0))
        global_scheduler = GlobalStateScheduler(monitor=monitor, modulator_system=mod_sys)

        neuron_group.reset_state(1, device)
        syn.reset_state(1, device)

        pre_idx = conn.pre_indices.to(device)
        post_idx = conn.post_indices.to(device)

        x1 = torch.randn(1, 1, 10, device=device) * 0.5
        spike1 = neuron_group(x1)
        r1 = neuron_group.get_internal_state()['r']

        post_spike_init = torch.zeros_like(spike1)
        post_r_init = torch.zeros_like(r1)
        syn(
            spike1[:, pre_idx], post_spike_init[:, post_idx],
            r1[:, pre_idx, :], post_r_init[:, post_idx, :]
        )

        prev_spike = spike1.clone()
        prev_r = r1.clone()

        x2 = torch.randn(1, 1, 10, device=device) * 2.0
        spike2 = neuron_group(x2)
        r2 = neuron_group.get_internal_state()['r']

        syn(
            spike2[:, pre_idx], prev_spike[:, post_idx],
            r2[:, pre_idx, :], prev_r[:, post_idx, :]
        )

        syn.pre_trace[0, 0] = 1.0
        syn.post_trace[0, 0] = 0.0
        syn.pre_trace[0, 1] = 0.9
        syn.post_trace[0, 1] = 0.1

        monitor.last_novelty = 0.8

        strength_before = syn.strength.clone()
        prune_flag_before = syn.prune_flag.clone()

        learn_scheduler = IntrinsicLearningScheduler(
            glial_network=glia, global_state_scheduler=global_scheduler
        )
        stats = learn_scheduler.step()

        self.assertIn('pruned_synapses', stats)
        self.assertIn('enhanced_synapses', stats)
        total_events = stats['pruned_synapses'] + stats['enhanced_synapses']
        self.assertGreater(total_events, 0, "学习步应至少触发一个修剪/增强事件")

        self.assertTrue(syn.prune_flag[0, 0].item(), "索引 0 的突触应被标记为修剪")
        self.assertTrue(syn.prune_flag[0, 1].item(), "索引 1 的突触应被标记为修剪")

        strength_after = syn.strength.clone()
        if stats['enhanced_synapses'] > 0:
            self.assertGreater(strength_after[0, 2].item(),
                               strength_before[0, 2].item(),
                               "被增强的突触强度应有所增加")

        if torch.equal(strength_before, strength_after) and torch.equal(prune_flag_before, syn.prune_flag):
            self.fail("突触状态未发生任何变化，学习的闭环失效")


if __name__ == '__main__':
    unittest.main()