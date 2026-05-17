# genesis_core.py
"""
tests/v0_2/test_v0_2_integration.py – v0.2 最终端到端集成测试
"""

import unittest
import torch
from typing import Dict, List, Tuple

# 真实神经元与突触
from genesis_core.neuron import LIFNeuron
from genesis_core.synapse import STDPSynapse

# 胶质细胞网络（含星形胶质细胞、小胶质细胞）
from genesis_core.glia import GlialNetwork, Astrocyte, Microglia

# 神经发生与凋亡模块
from genesis_core.neurogenesis import (
    NeurogenesisPool,
    NeurogenesisScheduler,
    check_apoptosis,
)

# 按约束要求导入 Connection 和 SynapticConnection
from genesis_core.utils import Connection
from genesis_core.life import SynapticConnection


class TestV0_2Integration(unittest.TestCase):
    """v0.2 最终端到端集成测试套件"""

    def test_neuron_apoptosis_by_health_collapse(self) -> None:
        """
        验证：将神经元健康度手动降至阈值以下后，
        check_apoptosis 应将其标记为可凋亡。
        """
        # 1. 创建一个真实的 LIFNeuron 实例
        neuron = LIFNeuron(num_neurons=1, num_dendrites=1)
        neuron.reset_state(1)  # 初始化内部状态

        # 2. 手动将健康度 h 设置为 0.01（低于凋亡阈值 0.05）
        neuron._h[0, 0] = torch.tensor(0.01)

        # 3. 调用 check_apoptosis 函数
        candidates: List = check_apoptosis(
            neurons=[neuron],
            connections=[],                # 无突触连接，不影响健康度判定
            silent_counters={},
            silence_window=10000,
        )

        # 4. 验证：该神经元被标记为可凋亡
        self.assertIn(neuron, candidates,
                      "健康度低于阈值的神经元应被标记为可凋亡")

    def test_neuron_apoptosis_by_long_silence(self) -> None:
        """
        验证：长时间无发放（沉默）的神经元会被 check_apoptosis 标记为可凋亡。
        """
        # 1. 创建一个真实的 LIFNeuron 实例，并保持健康度正常
        neuron = LIFNeuron(num_neurons=1, num_dendrites=1)
        neuron.reset_state(1)
        neuron._h[0, 0] = torch.tensor(0.7)

        # 2. 设置沉默计数器使其超过窗口阈值
        silence_window: int = 100
        silent_counters: Dict[int, int] = {id(neuron): silence_window + 1}

        # 3. 调用 check_apoptosis 函数
        candidates: List = check_apoptosis(
            neurons=[neuron],
            connections=[],
            silent_counters=silent_counters,
            silence_window=silence_window,
        )

        # 4. 验证：该神经元被标记为可凋亡
        self.assertIn(neuron, candidates,
                      "长时间沉默的神经元应被标记为可凋亡")

    def test_neurogenesis_pool_spawn(self) -> None:
        """
        验证：NeurogenesisPool 能够生成一个新的 LIFNeuron，
        其内部状态为默认值（健康度接近 0.8，膜电位为零）。
        """
        # 1. 创建 NeurogenesisPool 实例
        pool = NeurogenesisPool(neuron_class=LIFNeuron)

        # 2. 调用 spawn_neuron() 必定生成一个新神经元
        new_neuron = pool.spawn_neuron()

        # 3. 验证：成功生成一个新的 LIFNeuron
        self.assertIsInstance(new_neuron, LIFNeuron,
                              "生成的神经元必须是 LIFNeuron 实例")

        # 验证内部状态为默认值
        state = new_neuron.get_internal_state()
        batch_h = state['h']   # shape (1, 1)
        batch_u = state['u']

        self.assertAlmostEqual(batch_h.item(), 0.8, delta=0.15,
                               msg="默认健康度应接近 0.8")
        self.assertEqual(batch_u.item(), 0.0,
                         msg="初始膜电位应为 0.0")

    def test_scheduler_step_without_crash(self) -> None:
        """
        验证：包含真实神经元、突触和胶质的小型网络，
        NeurogenesisScheduler 连续运行 50 步不会崩溃。
        """
        # 1. 创建小型网络（两个 LIFNeuron 实例，每个包含 1 个神经元）
        neuron1 = LIFNeuron(num_neurons=1, num_dendrites=1)
        neuron2 = LIFNeuron(num_neurons=1, num_dendrites=1)

        # 创建两个 STDPSynapse 实例，构成双向连接
        syn1 = STDPSynapse(num_synapses=1)
        syn2 = STDPSynapse(num_synapses=1)
        connections: List[Tuple] = [
            (syn1, neuron1, neuron2),
            (syn2, neuron2, neuron1),
        ]

        # 2. 创建 GlialNetwork（至少需要一个星形胶质细胞以避免构建间隙连接时出错）
        astro = Astrocyte(
            astrocyte_id=0,
            neuron_groups=[],   # 可留空，调度器内部会补全
            synapse_groups=[],
            position=(0.0, 0.0, 0.0),
        )
        glia = GlialNetwork(astrocytes=[astro], microglias=[])

        # 3. 创建 NeurogenesisScheduler，抑制神经发生和凋亡，专注于稳定性测试
        scheduler = NeurogenesisScheduler(
            neurons=[neuron1, neuron2],
            connections=connections,
            glia=glia,
            p_neurogenesis_base=0.0,      # 不触发神经发生
            silence_window=1000000,       # 极大的沉默窗口，防止意外凋亡
        )

        # 4. 连续运行 50 步，每步为神经元提供随机输入
        for step_idx in range(50):
            x1 = torch.randn(1, 1, 1) * 0.5
            x2 = torch.randn(1, 1, 1) * 0.5

            # 神经元前向传播
            _ = neuron1(x1)
            _ = neuron2(x2)

            # 记录发放情况（根据 firing_rate_avg 判断活性）
            spikes_dict = {
                id(neuron1): neuron1.get_internal_state()['firing_rate_avg'][0, 0].item(),
                id(neuron2): neuron2.get_internal_state()['firing_rate_avg'][0, 0].item(),
            }
            scheduler.record_spikes(spikes_dict)

            # 执行调度步（凋亡检测与神经发生检测）
            try:
                scheduler.step()
            except Exception as e:
                self.fail(f"调度器在第 {step_idx} 步崩溃: {e}")

        # 5. 验证：调度器稳定运行，且神经元未被意外移除
        stats = scheduler.get_statistics()
        self.assertEqual(stats['active_neurons'], 2,
                         "预计仍有 2 个活跃神经元")


if __name__ == "__main__":
    unittest.main()