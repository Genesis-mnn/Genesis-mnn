# visualization/client.py
"""本地 GUI 可视化客户端（基于 PyQt/PySide + pyqtgraph）。

包含三个核心视图：
    - ConceptViewer: 脑波涟漪视图，3D 彩色波纹干涉图展现全脑脉冲活动。
    - PhaseSyncViewer: 集群同步视图，以热力图或环形图展现集群相位同步关系。
    - MemoryViewer: 长期记忆碎片视图，展示记忆索引快照与模式补全。

同时提供 MainVisualizationWindow 主窗口及 jupyter_launch() 函数，
可直接在 Jupyter Lab 中嵌入运行。
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import sys
import warnings

# 尝试导入 Qt 相关库
try:
    from PyQt5 import QtWidgets, QtCore, QtGui
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QDockWidget, QMainWindow
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph import ColorMap
    QT_AVAILABLE = True
except ImportError:
    try:
        from PySide2 import QtWidgets, QtCore, QtGui
        from PySide2.QtCore import Qt, QTimer
        from PySide2.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QDockWidget, QMainWindow
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        from pyqtgraph import ColorMap
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False
        warnings.warn("PyQt5/PySide2 and pyqtgraph are required for the GUI client.")

from .server import VisualizationServer


# ============================================================================
# ConceptViewer：3D 脑波涟漪视图
# ============================================================================

class ConceptViewer(gl.GLViewWidget):
    """3D 动态彩色波纹干涉图，实时展示全脑脉冲活动。

    使用 OpenGL 渲染，每个神经元显示为一个彩色球体，其大小和颜色随膜电位/发放率变化。
    波纹干涉效果通过更新球体的发光强度与位置模拟。
    符合术语中立宪章：内部变量名为 neutral_spheres、colormap 等。
    """

    def __init__(self, parent: Optional[QWidget] = None, num_neurons: int = 100) -> None:
        """
        参数:
            parent: 父级 QWidget。
            num_neurons: 神经元总数，用于预分配渲染对象。
        """
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setWindowTitle('脑波涟漪视图 (ConceptViewer)')
        self.num_neurons = num_neurons

        # 3D 坐标：随机分布在球壳或网格上
        self._positions = self._init_positions(num_neurons)
        # 颜色映射：低活动（蓝）→ 高活动（红）
        self._cmap = ColorMap([0.0, 0.5, 1.0], [np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.array([1, 0, 0, 1])])

        # 创建散点图项（代表神经元）
        self._scatter = gl.GLScatterPlotItem(
            pos=self._positions,
            size=0.1 * np.ones(num_neurons, dtype=np.float32),
            color=np.ones((num_neurons, 4), dtype=np.float32),
        )
        self.addItem(self._scatter)

        # 可选：添加网格或波纹环（暂略）

        # 数据缓冲区
        self._firing_rate: np.ndarray = np.zeros(num_neurons)
        self._membrane: np.ndarray = np.zeros(num_neurons)

    def _init_positions(self, n: int) -> np.ndarray:
        """初始化神经元在 3D 空间中的位置。使用 Fibonacci 球面分布。"""
        indices = np.arange(0, n, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n)
        theta = np.pi * (1 + np.sqrt(5)) * indices
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        return np.column_stack([x, y, z]).astype(np.float32)

    def update_data(self, firing_rate: np.ndarray, membrane: np.ndarray) -> None:
        """更新当前神经元的发放率和膜电位数据，触发重绘。

        参数:
            firing_rate: 形状 (num_neurons,) 的瞬时发放率。
            membrane: 形状 (num_neurons,) 的膜电位。
        """
        if firing_rate.shape[0] != self.num_neurons:
            # 动态调整数量（简单截断或填充）
            n = min(firing_rate.shape[0], self.num_neurons)
            self._firing_rate[:n] = firing_rate[:n]
            self._membrane[:n] = membrane[:n]
        else:
            self._firing_rate = firing_rate
            self._membrane = membrane

        # 根据发放率缩放点的大小（0.05 ~ 0.3）
        norm_rate = np.clip(self._firing_rate / max(np.max(self._firing_rate), 1e-3), 0, 1)
        sizes = (0.08 + 0.22 * norm_rate).astype(np.float32)

        # 根据膜电位或发放率映射颜色
        values = self._firing_rate
        v_min = np.percentile(values, 5)
        v_max = np.percentile(values, 95)
        if v_max > v_min:
            norm = (np.clip(values, v_min, v_max) - v_min) / (v_max - v_min)
        else:
            norm = np.zeros_like(values)
        colors = self._cmap.map(norm, mode='float')
        colors[norm == 0] = np.array([0, 0, 1, 0.5])  # 静默时为半透明蓝

        # 更新散点图
        self._scatter.setData(pos=self._positions, size=sizes, color=colors)


# ============================================================================
# PhaseSyncViewer：集群同步视图（热力图）
# ============================================================================

class PhaseSyncViewer(QWidget):
    """集群相位同步指数热力图。

    显示为 N×N 矩阵热图，或可切换至环形图（Chord Diagram）。
    严格中立术语：内部称同步矩阵 sync_matrix。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        参数:
            parent: 父级 QWidget。
        """
        super().__init__(parent)
        self.setMinimumSize(300, 250)
        self.setWindowTitle('集群同步视图 (PhaseSyncViewer)')

        layout = QVBoxLayout(self)
        self._image_item = pg.ImageItem()
        self._view_box = pg.GraphicsLayoutWidget()
        self._plot = self._view_box.addPlot()
        self._plot.addItem(self._image_item)
        self._plot.setLabel('left', 'Cluster')
        self._plot.setLabel('bottom', 'Cluster')
        layout.addWidget(self._view_box)

        self._sync_matrix: np.ndarray = np.eye(1)

    def update_data(self, sync_matrix: np.ndarray, labels: Optional[List[str]] = None) -> None:
        """更新相位同步矩阵并刷新显示。

        参数:
            sync_matrix: 形状 (N, N) 的对称矩阵，值域 [0,1]。
            labels: 集群名称列表（用于轴标签）。
        """
        self._sync_matrix = sync_matrix
        self._image_item.setImage(sync_matrix, levels=(0, 1))
        if labels is not None and len(labels) == sync_matrix.shape[0]:
            # 设置轴刻度（简化）
            pass
        # 自动缩放
        self._plot.autoRange()


# ============================================================================
# MemoryViewer：长期记忆碎片视图（时间轴 + 网络图）
# ============================================================================

class MemoryViewer(QWidget):
    """长期记忆索引与模式补全视图。

    以时间轴折线图展示记忆快照的强度，或以网络图展示记忆模式间的关系。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        参数:
            parent: 父级 QWidget。
        """
        super().__init__(parent)
        self.setMinimumSize(400, 200)
        self.setWindowTitle('记忆碎片视图 (MemoryViewer)')

        layout = QVBoxLayout(self)
        self._view_box = pg.GraphicsLayoutWidget()
        self._plot = self._view_box.addPlot()
        self._plot.setLabel('bottom', 'Time Step')
        self._plot.setLabel('left', 'Strength')
        self._curve = self._plot.plot(pen='y')
        layout.addWidget(self._view_box)

        # 存储快照数据
        self._snapshots: List[Dict] = []

    def update_data(self, memory_data: Dict[str, Any]) -> None:
        """更新长期记忆索引数据。

        参数:
            memory_data: 由 pipeline 生成的包含 'snapshots' 键的字典。
        """
        self._snapshots = memory_data.get('snapshots', [])
        if self._snapshots:
            steps = [s['step'] for s in self._snapshots]
            strengths = [s['strength'] for s in self._snapshots]
            self._curve.setData(steps, strengths)
        else:
            self._curve.clear()


# ============================================================================
# MainVisualizationWindow：主窗口
# ============================================================================

class MainVisualizationWindow(QMainWindow):
    """可视化主窗口，包含三个视图和状态控制。

    通过连接到 VisualizationServer 的回调自动更新所有视图。
    """

    def __init__(self, server: VisualizationServer, parent: Optional[QWidget] = None) -> None:
        """
        参数:
            server: 已启动的 VisualizationServer 实例。
            parent: 父级 QWidget。
        """
        super().__init__(parent)
        self.setWindowTitle("Genesis 高级可视化系统 v0.1")
        self.resize(1200, 800)

        self._server = server

        # 创建三个视图
        self._concept_viewer = ConceptViewer(self, num_neurons=200)
        self._phase_sync_viewer = PhaseSyncViewer(self)
        self._memory_viewer = MemoryViewer(self)

        # 使用 Dock 布局
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.addWidget(self._concept_viewer)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(self._phase_sync_viewer)
        right_layout.addWidget(self._memory_viewer)
        layout.addWidget(left_widget, stretch=2)
        layout.addWidget(right_widget, stretch=1)

        # 注册数据更新回调
        self._server.register_callback(self._on_data_updated)

        # 启动服务器
        self._server.start()

        # 设置定时器触发视图更新（可选，回调已包含更新）
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(50)  # 50 ms

    def _on_data_updated(self, data: Dict[str, Any]) -> None:
        """从服务器回调接收最新数据并更新视图。"""
        # 更新 ConceptViewer
        fr = data.get('firing_rate', np.array([]))
        mem = data.get('membrane', np.array([]))
        if len(fr) > 0:
            self._concept_viewer.update_data(fr, mem)

        # 更新 PhaseSyncViewer
        sync_mat = data.get('phase_sync_matrix', np.eye(1))
        self._phase_sync_viewer.update_data(sync_mat)

        # 更新 MemoryViewer
        mem_data = data.get('memory', {})
        self._memory_viewer.update_data(mem_data)

    def _refresh(self) -> None:
        """强制重绘所有视图（用于动画）。"""
        self._concept_viewer.update()
        self._phase_sync_viewer.update()
        self._memory_viewer.update()

    def closeEvent(self, event):
        self._server.stop()
        super().closeEvent(event)


# ============================================================================
# Jupyter Lab 嵌入接口
# ============================================================================

def jupyter_launch(server: VisualizationServer) -> Optional[MainVisualizationWindow]:
    """在 Jupyter Lab 中启动可视化 GUI。

    需要 %gui qt 魔法命令预加载 Qt 事件循环，或通过 IPython 原生方法执行。
    返回主窗口实例，以便用户继续交互。

    参数:
        server: 已配置的 VisualizationServer 实例。
    返回:
        MainVisualizationWindow 实例，或 None（若 Qt 不可用）。
    """
    if not QT_AVAILABLE:
        warnings.warn("Qt 绑定不可用，无法启动本地 GUI。")
        return None

    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        # 启用 Qt 事件循环集成
        ipython.magic('gui qt')

    window = MainVisualizationWindow(server)
    window.show()
    return window