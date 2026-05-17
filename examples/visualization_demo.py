"""
可视化演示脚本：
1. 创建一个小型网络并运行几步仿真
2. 通过 GenomeDataPipeline 计算可视化指标
3. 将数据传递给 VisualizationServer
"""

import torch
import numpy as np
from genesis_core.neuron import LIFNeuron
from genesis_core.synapse import STDPSynapse
from genesis_core.monitor import SpikeMonitor, MembraneMonitor
from genesis_core.modulator import ModulatorSystem, create_default_modulators
from visualization.pipeline import GenomeDataPipeline
from visualization.server import VisualizationServer

def main():
    print("===== 可视化演示 =====")

    # 1. 创建小型网络
    print("正在创建网络...")
    nrn_pre = LIFNeuron(num_neurons=10, v_threshold=1.0, tau=10.0)
    nrn_post = LIFNeuron(num_neurons=5, v_threshold=1.0, tau=10.0)
    syn = STDPSynapse(num_synapses=50)            # 10 x 5

    # 挂接监视器
    spike_mon = SpikeMonitor(num_neurons=10 + 5)  # 总神经元数
    mem_mon_pre = MembraneMonitor(neuron=nrn_pre)
    mem_mon_post = MembraneMonitor(neuron=nrn_post)

    # 构建全连接的 pre/post 索引
    pre_idx = torch.arange(10).repeat_interleave(5)
    post_idx = torch.arange(5).repeat(10)

    # 简化的脉冲传播函数
    def step_network(step_count=1):
        for _ in range(step_count):
            pre_spikes = nrn_pre(torch.randn(1, 1, 10) * 0.5).squeeze()
            post_spikes = nrn_post(torch.randn(1, 1, 5) * 0.5).squeeze()
            pre_r = nrn_pre.get_internal_state()['r']
            post_r = nrn_post.get_internal_state()['r']

            # 记录监视数据
            combined_spikes = torch.cat([pre_spikes, post_spikes])
            spike_mon.record(combined_spikes)
            mem_mon_pre.record()
            mem_mon_post.record()

            # 映射到突触
            pre_spike_syn = pre_spikes[pre_idx].unsqueeze(0)
            post_spike_syn = post_spikes[post_idx].unsqueeze(0)
            pre_r_syn = pre_r[0, pre_idx, :].unsqueeze(0)
            post_r_syn = post_r[0, post_idx, :].unsqueeze(0)

            syn(pre_spike_syn, post_spike_syn, pre_r_syn, post_r_syn)

    step_network(10)

    # 2. 初始化调制场
    mod_sys = create_default_modulators()
    mod_sys.register_target(nrn_pre, (0.0, 0.0, 0.0))
    mod_sys.register_target(nrn_post, (1.0, 0.0, 0.0))

    # 3. 创建数据管道（挂接真实监视器）
    print("正在启动可视化数据管道...")
    pipeline = GenomeDataPipeline(
        spike_monitors=[spike_mon],
        membrane_monitors=[mem_mon_pre, mem_mon_post],
        weight_monitors=[],
        plasticity_monitors=[],
    )

    # 4. 创建可视化服务器
    server = VisualizationServer(pipeline, update_interval_ms=200)
    server.start()
    print("可视化服务器已启动 (无 GUI 模式)")

    # 5. 运行几步并观察管道输出
    print("\n运行仿真并观察可视化指标...")
    for step in range(5):
        step_network(1)
        pipeline.step()

        firing_rates = pipeline.get_instant_firing_rate()
        sync_matrix = pipeline.compute_phase_sync_matrix()

        print(f"步 {step}: 平均发放率 = {np.mean(firing_rates):.4f}, "
              f"同步矩阵形状 = {sync_matrix.shape}")

        global_state = {
            'reward_error': 0.1,
            'satiety': 0.5,
            'novelty': 0.2,
            'stress': 0.0,
            'energy_deficit': 0.1,
        }
        mod_sys.step(global_state)

    # 6. 验证服务器
    latest_data = server.get_latest_data()
    print(f"\n最终可视化数据键: {list(latest_data.keys())}")
    print("✓ 可视化管道与服务器运行正常")

    server.stop()
    print("===== 演示结束 =====")

if __name__ == "__main__":
    main()