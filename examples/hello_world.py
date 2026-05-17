# 文件: examples/hello_world.py
"""
Hello World: 创建形态神经元，注入电流，观察脉冲发放模式并打印发放率统计
"""

import torch
from genesis_core import LIFNeuron

def main():
    # 实例化单个 LIF 神经元
    neuron = LIFNeuron(num_neurons=1, tau=10.0, v_threshold=1.0, v_reset=0.0)
    neuron.reset_state(batch_size=1)

    T = 200               # 模拟步数
    I_ext = 1.2           # 恒定注入电流 (超过阈值 1.0)
    current = torch.ones(1, 1, 1) * I_ext

    spike_history = []
    for t in range(T):
        out = neuron(current)          # out 形状 (1, 1)
        spike_history.append(out.item())

    spike_count = sum(spike_history)
    firing_rate = spike_count / T

    print("===== Hello World (形态神经元) =====")
    print(f"模拟步数: {T}")
    print(f"注入电流: {I_ext}")
    print(f"脉冲总数: {spike_count}")
    print(f"发放率 (每个时间步): {firing_rate:.4f}")
    # 打印前几个脉冲时间步
    spike_times = [i for i, s in enumerate(spike_history) if s > 0.5]
    print(f"脉冲发生时间步 (前10): {spike_times[:10]}")

    if firing_rate > 0:
        print("形态神经元在输入下成功产生脉冲输出。")
    else:
        print("未检测到脉冲 (可能需调整参数)。")

if __name__ == "__main__":
    main()
