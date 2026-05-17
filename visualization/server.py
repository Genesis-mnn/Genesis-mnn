# visualization/server.py
"""本地数据服务：封装 GenomeDataPipeline，在独立线程中周期性更新数据，并通知 GUI 客户端。

不依赖 Flask/WebSocket，仅使用 Python 线程与回调实现。
模仿 Nengo GUI 的服务端架构，提供 `VisualizationServer` 类。
"""

import threading
import time
from typing import Dict, List, Callable, Optional, Any
import numpy as np

from .pipeline import GenomeDataPipeline


class VisualizationServer:
    """本地可视化数据服务。

    在后台线程中持续调用 pipeline.step()，每隔 update_interval_ms 毫秒触发数据更新，
    并通过注册的回调函数将最新指标推送给 GUI 客户端。
    """

    def __init__(
        self,
        pipeline: GenomeDataPipeline,
        update_interval_ms: int = 100,  # 100 ms 刷新率
    ) -> None:
        """
        参数:
            pipeline: 已配置好的数据管道。
            update_interval_ms: 数据更新间隔（毫秒）。
        """
        self.pipeline = pipeline
        self.update_interval = update_interval_ms / 1000.0
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # 最新数据缓存
        self.latest_data: Dict[str, Any] = {
            'firing_rate': np.array([]),
            'phase_sync_matrix': np.eye(1),
            'syn_strength': np.array([]),
            'membrane': np.array([]),
            'memory': {},
        }

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """注册数据更新回调函数。回调接收一个包含最新指标的字典。

        参数:
            callback: 回调函数，签名为 callback(data: Dict[str, Any])。
        """
        self._callbacks.append(callback)

    def start(self) -> None:
        """启动后台数据更新线程。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止后台线程。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        """后台运行循环，定期调用 pipeline 并通知客户端。"""
        while self._running:
            try:
                # 1. 管道步进
                self.pipeline.step()

                # 2. 计算各项指标
                data = {
                    'firing_rate': self.pipeline.get_instant_firing_rate(),
                    'phase_sync_matrix': self.pipeline.compute_phase_sync_matrix(),
                    'syn_strength': self.pipeline.get_synaptic_strength_map(),
                    'membrane': self.pipeline.get_membrane_state(),
                    'memory': self.pipeline.get_memory_indices(),
                }
                self.latest_data = data

                # 3. 触发回调
                for cb in self._callbacks:
                    try:
                        cb(data)
                    except Exception:
                        pass

                time.sleep(self.update_interval)
            except Exception:
                time.sleep(self.update_interval)

    def get_latest_data(self) -> Dict[str, Any]:
        """同步获取最新数据快照（线程安全由 GIL 保证简单类型）。"""
        return self.latest_data.copy()