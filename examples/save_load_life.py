# 文件: examples/save_load_life.py
"""
.life 文件保存与加载示例: 创建基础网络，运行几步，保存再加载，对比膜电位一致性
"""

import torch
import os
import tempfile
from genesis_core import LIFNeuron, GenesisNetwork, LifeSaver, LifeLoader, SynapticConnection

def main():
    # 1. 创建网络
    net = GenesisNetwork()
    lif = LIFNeuron(num_neurons=3, tau=10.0, v_threshold=1.0)
    lif.reset_state(batch_size=1)

    # 运行几步，让膜电位偏离初始值
    print("运行网络 20 步...")
    for step in range(20):
        # 随机输入
        inp = torch.randn(1, 1, 3) * 0.5
        _ = lif(inp)

    # 保存运行后的状态
    state_before = lif.get_internal_state()
    mean_u_before = state_before['u'].mean().item()
    print(f"保存前平均膜电位: {mean_u_before:.6f}")

    net.neurons.append(lif)
    # 添加一个虚拟突触连接（为了结构完整）
    pre_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    post_idx = torch.tensor([1, 2, 0], dtype=torch.long)
    from genesis_core import STDPSynapse
    syn = STDPSynapse(num_synapses=3)
    conn = SynapticConnection(synapse=syn, pre_indices=pre_idx, post_indices=post_idx)
    net.connections.append(conn)

    # 2. 保存为 .life 文件
    with tempfile.NamedTemporaryFile(suffix=".life", delete=False) as f:
        filepath = f.name
    LifeSaver.save(net, filepath, creator="ExampleUser")
    print(f"网络已保存至: {filepath}")

    # 3. 从文件加载
    loaded_net = LifeLoader.load(filepath, device=torch.device('cpu'))
    loaded_lif = loaded_net.neurons[0]
    state_after = loaded_lif.get_internal_state()
    mean_u_after = state_after['u'].mean().item()

    # 4. 比较膜电位
    max_diff = (state_before['u'] - state_after['u']).abs().max().item()
    print(f"加载后平均膜电位: {mean_u_after:.6f}")
    print(f"膜电位最大绝对差: {max_diff:.10f}")

    # 清理临时文件
    os.unlink(filepath)

    if max_diff < 1e-5:
        print("✓ 膜电位完全一致，.life 文件保存/加载无损。")
    else:
        print("✗ 存在轻微差异（可能由浮点精度导致）。")

if __name__ == "__main__":
    main()
