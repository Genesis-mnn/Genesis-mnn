"""
Genesis 可视化客户端启动脚本
启动本地 GUI，实时展示网络内部状态
"""
import os
import sys

# 获取当前Python环境的路径
env_path = sys.prefix
# 拼接出 Qt 运行时库所在目录
qt_path = os.path.join(env_path, 'Library', 'bin')
# 设置环境变量，强行让 PyQt5 去这里找插件
os.environ['PATH'] = qt_path + ';' + os.environ.get('PATH', '')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(env_path, 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins')

import torch
import sys
from genesis_core.simulator import create_default_simulator
from genesis_core.monitor import SpikeMonitor, MembraneMonitor
from visualization.pipeline import GenomeDataPipeline
from visualization.server import VisualizationServer
from visualization.client import MainVisualizationWindow, QT_AVAILABLE

def main():
    if not QT_AVAILABLE:
        print("错误：未安装 PyQt5/PySide2 或 pyqtgraph，无法启动 GUI。")
        print("请执行 pip install PyQt5 pyqtgraph PyOpenGL")
        sys.exit(1)

    # 1. 创建模拟器
    print("正在创建默认模拟器 (100 个 LIF 神经元)...")
    sim = create_default_simulator(num_neurons=100, connection_prob=0.1)

    # 2. 挂接监视器
    spike_mon = SpikeMonitor(num_neurons=100)
    membrane_mon = MembraneMonitor(neuron=sim.neurons[0])

    original_step = sim.step
    def step_with_monitor(*args, **kwargs):
        snapshot = original_step(*args, **kwargs)
        # 从模拟器中提取膜电位并模拟脉冲发放
        u = sim.neurons[0].get_internal_state()['u'].detach()
        spikes_binary = (u > 0.5).float().squeeze()
        spike_mon.record(spikes_binary)
        membrane_mon.record()
        return snapshot
    sim.step = step_with_monitor

    # 3. 创建数据管道与服务器
    pipeline = GenomeDataPipeline(
        spike_monitors=[spike_mon],
        membrane_monitors=[membrane_mon],
        weight_monitors=[],
        plasticity_monitors=[]
    )
    server = VisualizationServer(pipeline, update_interval_ms=50)

    # 4. 启动 GUI
    print("正在启动 Genesis 可视化客户端...")
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainVisualizationWindow(server)

    # 5. 定时器驱动仿真步进
    timer = window._timer
    timer.timeout.disconnect()
    timer.timeout.connect(lambda: (sim.step(), pipeline.step()))
    timer.start(50)

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()