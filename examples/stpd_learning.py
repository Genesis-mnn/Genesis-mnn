# 文件: examples/stpd_learning.py
"""
STDP 学习示例: 构建两个神经元的网络，使用 STDP 突触，运行多步并展示权重变化
"""

import torch
from genesis_core import STDPSynapse

def main():
    # 创建一个 STDP 突触，连接 1 个 pre -> 1 个 post
    stdp = STDPSynapse(
        num_synapses=1,
        lr=1.0,
        A_plus=0.02,      # 增强系数
        A_minus=0.02,     # 减弱系数
        tau_pre=10.0,
        tau_post=10.0,
    )
    stdp.reset_state(batch_size=1)
    # 初始耦合强度设为 0.5
    stdp.strength.data.fill_(0.5)

    # 谐振敏感度向量（保持零，使阻抗最小、信号传递最高效）
    pre_r = torch.zeros(1, 1, 1)
    post_r = torch.zeros(1, 1, 1)

    print("===== STDP 学习演示 =====")
    print(f"初始 strength: {stdp.strength.item():.4f}")

    # ---- Phase 1: LTP (pre 先于 post) ----
    for i in range(10):
        # pre 发放
        _ = stdp(torch.tensor([[1.0]]), torch.tensor([[0.0]]), pre_r, post_r)
        # post 发放
        _ = stdp(torch.tensor([[0.0]]), torch.tensor([[1.0]]), pre_r, post_r)
    print(f"LTP 10 次后 strength: {stdp.strength.item():.4f}")

    # ---- Phase 2: LTD (post 先于 pre) ----
    for i in range(10):
        # post 发放
        _ = stdp(torch.tensor([[0.0]]), torch.tensor([[1.0]]), pre_r, post_r)
        # pre 发放
        _ = stdp(torch.tensor([[1.0]]), torch.tensor([[0.0]]), pre_r, post_r)
    print(f"LTD 10 次后 strength: {stdp.strength.item():.4f}")

    final_strength = stdp.strength.item()
    print(f"最终 strength: {final_strength:.4f}")
    if final_strength > 0.5:
        print("突触发生了净增强 (LTP 占优)。")
    elif final_strength < 0.5:
        print("突触发生了净减弱 (LTD 占优)。")
    else:
        print("强度未变化。")

if __name__ == "__main__":
    main()
