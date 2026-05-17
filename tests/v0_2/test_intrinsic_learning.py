# tests/v0_2/test_intrinsic_learning.py
import unittest
import torch
from genesis_core import (
    LIFNeuron,
    STDPSynapse,
    Astrocyte,
    GlialNetwork,
    GlobalStateMonitor,
    GlobalStateScheduler,
    IntrinsicLearningScheduler,
)
from genesis_core.modulator import ModulatorSystem


class TestIntrinsicLearning(unittest.TestCase):
    """内源性学习系统的单元测试。"""

    def test_learning_triggered_by_high_novelty(self):
        neuron = LIFNeuron(num_neurons=2, num_dendrites=1)
        synapse = STDPSynapse(num_synapses=2, tau_pre=20.0, tau_post=20.0)

        synapse.pre_trace[0, 0] = 1.0
        synapse.post_trace[0, 0] = 0.0
        synapse.pre_trace[0, 1] = 0.0
        synapse.post_trace[0, 1] = 0.0

        astro = Astrocyte(
            astrocyte_id=0,
            neuron_groups=[(neuron, [0, 1])],
            synapse_groups=[(synapse, [0, 1])],
            position=(0.0,),
        )
        glial_net = GlialNetwork(astrocytes=[astro])

        monitor = GlobalStateMonitor(neurons=[neuron], connections=[])
        modulator_system = ModulatorSystem()   # 修复：提供有效实例
        global_scheduler = GlobalStateScheduler(monitor=monitor, modulator_system=modulator_system)
        monitor.last_novelty = 0.5

        scheduler = IntrinsicLearningScheduler(
            glial_network=glial_net, global_state_scheduler=global_scheduler
        )

        stats = scheduler.step()

        self.assertGreaterEqual(stats['pruned_synapses'], 1)
        self.assertGreaterEqual(stats['enhanced_synapses'], 1)
        self.assertTrue(synapse.prune_flag[0, 0].item())
        self.assertTrue(synapse.protected[0, 1].item())
        self.assertAlmostEqual(
            synapse.strength[0, 1].item(), 0.51, places=3
        )

    def test_learning_not_triggered_by_low_novelty(self):
        neuron = LIFNeuron(num_neurons=2, num_dendrites=1)
        synapse = STDPSynapse(num_synapses=2, tau_pre=20.0, tau_post=20.0)

        synapse.pre_trace[0, 0] = 0.1
        synapse.post_trace[0, 0] = 0.2
        synapse.pre_trace[0, 1] = 0.1
        synapse.post_trace[0, 1] = 0.1

        initial_strength_0 = synapse.strength[0, 0].item()
        initial_strength_1 = synapse.strength[0, 1].item()

        astro = Astrocyte(
            astrocyte_id=0,
            neuron_groups=[(neuron, [0, 1])],
            synapse_groups=[(synapse, [0, 1])],
            position=(0.0,),
        )
        glial_net = GlialNetwork(astrocytes=[astro])

        monitor = GlobalStateMonitor(neurons=[neuron], connections=[])
        modulator_system = ModulatorSystem()
        global_scheduler = GlobalStateScheduler(monitor=monitor, modulator_system=modulator_system)
        monitor.last_novelty = 0.05

        scheduler = IntrinsicLearningScheduler(
            glial_network=glial_net, global_state_scheduler=global_scheduler
        )

        stats = scheduler.step()

        self.assertEqual(stats['pruned_synapses'], 0)
        self.assertEqual(stats['enhanced_synapses'], 0)
        self.assertFalse(synapse.prune_flag[0, 0].item())
        self.assertFalse(synapse.prune_flag[0, 1].item())
        self.assertFalse(synapse.protected[0, 0].item())
        self.assertFalse(synapse.protected[0, 1].item())
        self.assertAlmostEqual(
            synapse.strength[0, 0].item(), initial_strength_0, places=4
        )
        self.assertAlmostEqual(
            synapse.strength[0, 1].item(), initial_strength_1, places=4
        )

    def test_prediction_error_based_pruning(self):
        synapse = STDPSynapse(num_synapses=1, tau_pre=20.0, tau_post=20.0)
        synapse.pre_trace[0, 0] = 1.0
        synapse.post_trace[0, 0] = 0.2

        astro = Astrocyte(
            astrocyte_id=0,
            neuron_groups=[],
            synapse_groups=[(synapse, [0])],
            position=(0.0,),
        )

        pruned, enhanced = astro.intrinsic_learning_step(novelty_level=0.5)

        self.assertEqual(pruned, 1)
        self.assertEqual(enhanced, 0)
        self.assertTrue(synapse.prune_flag[0, 0].item())
        self.assertFalse(synapse.protected[0, 0].item())

    def test_prediction_error_based_protection(self):
        synapse = STDPSynapse(num_synapses=1, tau_pre=20.0, tau_post=20.0)
        synapse.pre_trace[0, 0] = 0.1
        synapse.post_trace[0, 0] = 0.1

        initial_strength = synapse.strength[0, 0].item()

        astro = Astrocyte(
            astrocyte_id=0,
            neuron_groups=[],
            synapse_groups=[(synapse, [0])],
            position=(0.0,),
        )

        pruned, enhanced = astro.intrinsic_learning_step(novelty_level=0.5)

        self.assertEqual(pruned, 0)
        self.assertEqual(enhanced, 1)
        self.assertFalse(synapse.prune_flag[0, 0].item())
        self.assertTrue(synapse.protected[0, 0].item())
        self.assertAlmostEqual(
            synapse.strength[0, 0].item(),
            initial_strength + 0.01,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()