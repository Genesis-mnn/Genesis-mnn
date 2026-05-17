# visualization/__init__.py
"""
Genesis 高级可视化系统 v0.1
提供脑波涟漪视图、集群同步视图、记忆碎片视图的标准实时可视化后端与前端。

本地 GUI 客户端基于 PyQt/PySide + pyqtgraph，也可嵌入 Jupyter Lab。
"""

from .pipeline import GenomeDataPipeline
from .server import VisualizationServer
from .client import (
    ConceptViewer,
    PhaseSyncViewer,
    MemoryViewer,
    MainVisualizationWindow,
    jupyter_launch,
)

__all__ = [
    "GenomeDataPipeline",
    "VisualizationServer",
    "ConceptViewer",
    "PhaseSyncViewer",
    "MemoryViewer",
    "MainVisualizationWindow",
    "jupyter_launch",
]