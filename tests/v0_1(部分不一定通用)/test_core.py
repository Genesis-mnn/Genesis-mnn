# 文件: tests/test_core.py
"""
Genesis 核心模块单元测试
覆盖：形态神经元发放、STDP 突触可塑性、胶质细胞修剪、.life 文件持久化
"""

import torch
import numpy as np
import os
import tempfile
from genesis_core import (
    LIFNeuron,
    STDPSynapse,
    Microglia,
    GenesisNetwork,
    LifeSaver,
    LifeLoader,
    SynapticConnection,
    create_layer,
    connect,
)
from genesis_core.genesis_core import MorphoNeuron, ResonanceSynapse


def test_lif_neuron_firing():
    """测试 LIFNeuron 在恒定输入下能否发放脉冲并统计发放率"""
    lif = LIFNeuron(num_neurons=2, tau=10.0, v_threshold=1.0, v_reset=0.0, R=1.0)
    lif.reset_state(batch_size=1)

    # 注入超过阈值的恒定电流
    current = torch.ones(1, 1, 2) * 1.5  # (batch, dendrites, neurons)

    spikes = []
    for _ in range(50):
        out = lif(current)  # (1, 2)
        spikes.append(out.detach().squeeze().numpy())
    spikes = np.array(spikes)

    assert spikes.sum() > 0, "LIF neuron should fire with suprathreshold input"
    firing_rates = spikes.mean(axis=0)
    print(f"  [LIF] 发放率 (2 neurons): {firing_rates}")
    print("  ✓ 测试通过: LIF 神经元正常发放")


def test_stdp_plasticity():
    """测试 STDPSynapse 的权重更新方向是否正确"""
    stdp = STDPSynapse(num_synapses=1, lr=1.0, A_plus=0.01, A_minus=0.01,
                       tau_pre=10.0, tau_post=10.0)
    stdp.reset_state(batch_size=1)
    stdp.strength.data.fill_(0.5)

    pre_r = torch.zeros(1, 1, 1)   # r 向量初始全零，使 impedance 最小
    post_r = torch.zeros(1, 1, 1)

    # ---- LTP: pre 先于 post 发放 ----
    for _ in range(5):
        _ = stdp(torch.tensor([[1.0]]), torch.tensor([[0.0]]), pre_r, post_r)
        _ = stdp(torch.tensor([[0.0]]), torch.tensor([[1.0]]), pre_r, post_r)
    strength_ltp = stdp.strength.item()
    assert strength_ltp > 0.5, f"LTP should increase weight, got {strength_ltp}"

    # ---- LTD: post 先于 pre 发放 ----
    stdp.reset_state(1)
    stdp.strength.data.fill_(0.5)
    for _ in range(5):
        _ = stdp(torch.tensor([[0.0]]), torch.tensor([[1.0]]), pre_r, post_r)
        _ = stdp(torch.tensor([[1.0]]), torch.tensor([[0.0]]), pre_r, post_r)
    strength_ltd = stdp.strength.item()
    assert strength_ltd < 0.5, f"LTD should decrease weight, got {strength_ltd}"

    print(f"  [STDP] LTP 后 weight = {strength_ltp:.4f}, LTD 后 weight = {strength_ltd:.4f}")
    print("  ✓ 测试通过: STDP 可塑性方向正确")


def test_microglia_pruning():
    """测试小胶质细胞能否标记低活动突触为修剪状态"""
    syn = STDPSynapse(num_synapses=1)
    syn.reset_state(1)
    # 设置极低的活动状态
    syn.strength.data.fill_(0.05)
    syn.pre_trace.fill_(0.0)

    micro = Microglia(synapse=syn, pruning_threshold=0.2,
                      check_interval=1, activity_smoothing=0.2)
    for _ in range(3):
        micro.step()          # 逐步累积活动历史并触发修剪判断

    prune_flag = syn.get_synaptic_state()['prune_flag'][0].item()
    assert prune_flag == True, f"低活动突触应被标记为可修剪, 实际 flag={prune_flag}"
    print("  ✓ 测试通过: 胶质细胞修剪正常")


def test_life_save_load():
    """测试 .life 文件的保存与加载，并验证膜电位一致性"""
    # 构建简单网络
    net = GenesisNetwork()
    lif = LIFNeuron(num_neurons=5, tau=10.0, v_threshold=1.0)
    lif.reset_state(1)
    # 运行几步以改变状态
    for _ in range(10):
        lif(torch.randn(1, 1, 5) * 0.3)

    state_before = lif.get_internal_state()
    net.neurons.append(lif)

    # 添加一个突触连接（仅用于完整性，保存时不会丢失）
    pre = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    post = torch.tensor([1, 2, 3, 4, 0], dtype=torch.long)
    syn = STDPSynapse(num_synapses=5)
    conn = SynapticConnection(synapse=syn, pre_indices=pre, post_indices=post)
    net.connections.append(conn)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_network.life")
        LifeSaver.save(net, filepath, creator="UnitTest")

        loaded_net = LifeLoader.load(filepath, device=torch.device('cpu'))
        loaded_lif = loaded_net.neurons[0]
        state_after = loaded_lif.get_internal_state()

        # 比较关键状态变量（允许极小浮点误差）
        for key in ['u', 'h', 'r']:
            diff = (state_before[key] - state_after[key]).abs().max().item()
            assert diff < 1e-5, f"状态 '{key}' 不一致, 最大差异 {diff}"
        print("  ✓ 测试通过: .life 保存/加载无损 (膜电位等状态一致)")


if __name__ == "__main__":
    print("===== 开始核心模块单元测试 =====")
    test_lif_neuron_firing()
    test_stdp_plasticity()
    test_microglia_pruning()
    test_life_save_load()
    print("===== 所有测试通过 =====")
