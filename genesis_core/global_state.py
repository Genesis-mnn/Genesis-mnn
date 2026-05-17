#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genesis 全局状态内源生成器 (global_state.py)

实现完全从网络内部活动中自发衍生的全局状态指标，
确保在零外部输入下调制场系统也能自主运行，实现公理四（存在先于任务）的完整闭环。

核心组件：
    - GlobalStateMonitor：从网络活动中计算全局状态指标，包含滑动平均平滑。
    - GlobalStateScheduler：封装监控器和调制场系统，提供统一的 step() 接口。
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any

# 导入核心基类和全局状态键常量
from .genesis_core import (
    MorphoNeuron,
    ResonanceSynapse,
    GLOBAL_REWARD_ERROR,
    GLOBAL_SATIETY,
    GLOBAL_NOVELTY,
    GLOBAL_STRESS,
    GLOBAL_ENERGY_DEFICIT,
)
# 导入调制场系统
from .modulator import ModulatorSystem


class GlobalStateMonitor:
    """全局状态监测器。

    完全从神经元和突触的内部状态中自主衍生五种全局状态指标，
    不依赖任何外部任务信号或预定义损失函数。
    对应白皮书公理四（存在先于任务）与公理五（价值内生）。

    参数:
        neurons: 网络中所有形态神经元实例的列表。
        connections: 网络中所有共鸣突触实例的列表（当前计算暂不需要，保留为未来扩展）。
        glia: 可选的胶质细胞网络（当前版本暂不直接使用）。
        optimal_rate: 最佳发放率，用于计算应激水平，默认 0.1（与健康度更新中的设定一致）。
        ema_alpha: 全局状态指标的 EMA 平滑系数，范围 (0, 1]。
    """

    def __init__(
        self,
        neurons: List[MorphoNeuron],
        connections: Optional[List[ResonanceSynapse]] = None,
        glia: Optional[Any] = None,
        optimal_rate: float = 0.1,
        ema_alpha: float = 0.1,
    ):
        if not (0 < ema_alpha <= 1.0):
            raise ValueError("ema_alpha 必须在 (0, 1] 区间内。")

        self.neurons = neurons
        self.connections = connections or []
        self.glia = glia
        self.optimal_rate = optimal_rate
        self.ema_alpha = ema_alpha

        # 用于计算新奇度的上一步全局发放率向量
        self._prev_firing_rates: Optional[torch.Tensor] = None

        # 各指标的滑动平均值（初始为 None，首次调用时直接赋值）
        self._ema_reward_error: Optional[float] = None
        self._ema_satiety: Optional[float] = None
        self._ema_novelty: Optional[float] = None
        self._ema_stress: Optional[float] = None
        self._ema_energy_deficit: Optional[float] = None

        # 新增：缓存最近一次 novelty（EMA 平滑值），供学习调度器使用
        self.last_novelty: float = 0.0

    def step(self) -> Dict[str, float]:
        """执行一步全局状态监测，计算并返回平滑后的全局状态指标字典。

        公理四：所有计算仅基于神经元与突触的内部状态。
        公理五：价值指标（奖赏误差等）完全从内部稳态偏差中衍生。

        返回:
            Dict[str, float]: 包含五个键值对的字典，键为调制原系统使用的常量，
                              值为平滑后的浮点数。在没有神经元时返回全零字典。
        """
        if not self.neurons:
            # 没有神经元的情况下，返回全零状态（标记网络未初始化）
            return {
                GLOBAL_REWARD_ERROR: 0.0,
                GLOBAL_SATIETY: 0.0,
                GLOBAL_NOVELTY: 0.0,
                GLOBAL_STRESS: 0.0,
                GLOBAL_ENERGY_DEFICIT: 0.0,
            }

        # 公理四、五：收集所有神经元的内部状态（健康度、发放率、目标健康度）
        all_h = []
        all_f = []
        all_rest = []
        for nrn in self.neurons:
            state = nrn.get_internal_state()
            # h: (batch, num_neurons)， firing_rate_avg: (batch, num_neurons)
            all_h.append(state['h'])
            all_f.append(state['firing_rate_avg'])
            # rest_h 是神经元参数，形状 (num_neurons,)
            all_rest.append(nrn.rest_h.data.detach())

        # 拼接所有神经元的数据，形成全局视图
        h_global = torch.cat(all_h, dim=1)  # (batch, total_N)
        f_global = torch.cat(all_f, dim=1)  # (batch, total_N)
        rest_global = torch.cat(all_rest, dim=0)  # (total_N,)

        # ---------- 1. reward_error: 平均目标健康度与平均当前健康度的差值 ----------
        mean_rest = rest_global.float().mean()
        mean_h = h_global.float().mean()
        reward_error_raw = (mean_rest - mean_h).item()

        # ---------- 2. satiety: 全局平均健康度 ----------
        satiety_raw = mean_h.item()

        # ---------- 3. stress: 全局平均发放率超出最佳发放率的程度 ----------
        # 公理四：发放率反映内在活动水平，正偏差表示过度活跃
        stress_raw = (f_global - self.optimal_rate).mean().item()

        # ---------- 4. energy_deficit: 健康度低于个体目标的总量 ----------
        # rest_global 形状 (total_N,)，需要广播到 (batch, total_N)
        deficit = torch.clamp(rest_global.unsqueeze(0) - h_global, min=0.0)
        energy_deficit_raw = deficit.mean().item()

        # ---------- 5. novelty: 基于全局发放率模式的变化程度 ----------
        # 当前全局发放率向量：跨批次求均值，得到每个神经元的平均发放率
        curr_f = f_global.mean(dim=0)  # (total_N,)
        if self._prev_firing_rates is not None:
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(
                curr_f.unsqueeze(0), self._prev_firing_rates.unsqueeze(0), dim=1
            ).item()
            # 处理数值异常（例如零向量导致的 NaN）
            if math.isnan(cos_sim):
                cos_sim = 0.0
            novelty_raw = 1.0 - max(cos_sim, 0.0)
        else:
            novelty_raw = 0.0  # 第一步无历史，新奇度定义为 0
        # 更新历史向量
        self._prev_firing_rates = curr_f.detach().clone()

        # ---------- EMA 平滑处理 ----------
        # 公理五：信号平滑避免调制场对瞬时波动的过度反应，保持价值信号的稳定性
        alpha = self.ema_alpha
        if self._ema_reward_error is None:
            self._ema_reward_error = reward_error_raw
            self._ema_satiety = satiety_raw
            self._ema_novelty = novelty_raw
            self._ema_stress = stress_raw
            self._ema_energy_deficit = energy_deficit_raw
        else:
            self._ema_reward_error = alpha * reward_error_raw + (1.0 - alpha) * self._ema_reward_error
            self._ema_satiety = alpha * satiety_raw + (1.0 - alpha) * self._ema_satiety
            self._ema_novelty = alpha * novelty_raw + (1.0 - alpha) * self._ema_novelty
            self._ema_stress = alpha * stress_raw + (1.0 - alpha) * self._ema_stress
            self._ema_energy_deficit = alpha * energy_deficit_raw + (1.0 - alpha) * self._ema_energy_deficit

        # 缓存当前和平滑后的新奇度，供外部（如学习调度器）访问
        self.last_novelty = self._ema_novelty

        return {
            GLOBAL_REWARD_ERROR: self._ema_reward_error,
            GLOBAL_SATIETY: self._ema_satiety,
            GLOBAL_NOVELTY: self._ema_novelty,
            GLOBAL_STRESS: self._ema_stress,
            GLOBAL_ENERGY_DEFICIT: self._ema_energy_deficit,
        }

    def get_current_novelty(self) -> float:
        """返回最近一次 step() 计算出的新奇度（EMA 平滑值）。"""
        return self.last_novelty

    def reset(self) -> None:
        """重置所有内部历史，以便从新的初始状态开始监测。"""
        self._prev_firing_rates = None
        self._ema_reward_error = None
        self._ema_satiety = None
        self._ema_novelty = None
        self._ema_stress = None
        self._ema_energy_deficit = None
        self.last_novelty = 0.0


class GlobalStateScheduler:
    """全局状态调度器。

    封装 GlobalStateMonitor 和 ModulatorSystem，
    提供统一的 step() 接口，使调制场完全由网络内部活动驱动。
    对应公理四（存在先于任务）与公理五（价值内生）的闭环实现。

    参数:
        monitor: GlobalStateMonitor 实例，负责从网络衍生全局状态。
        modulator_system: ModulatorSystem 实例，负责根据全局状态调控网络。
    """

    def __init__(
        self,
        monitor: GlobalStateMonitor,
        modulator_system: ModulatorSystem,
    ):
        self.monitor = monitor
        self.modulator_system = modulator_system

    def step(self) -> Dict[str, float]:
        """执行完整的“监测 → 调制”闭环一步。

        流程：
        1. 公理四：从网络内部活动衍生全局状态指标。
        2. 公理五：将内源全局状态传递给调制场系统，更新调制原浓度并施加到目标。

        返回:
            Dict[str, float]: 当前步的全局状态指标字典，供外部监视或记录。
        """
        # 1. 内源全局状态生成（公理四、五）
        global_state = self.monitor.step()

        # 2. 驱动调制场更新（公理五）
        self.modulator_system.step(global_state)

        return global_state


# ============================================================================
# 自包含测试与演示
# ============================================================================
if __name__ == "__main__":
    import sys
    import os

    # 确保当前目录在 path 中，以便导入同目录下的其他模块
    sys.path.insert(0, os.path.dirname(__file__))

    print("=== Genesis global_state.py 自包含演示 ===\n")

    # 创建简单的神经元层
    from neuron import LIFNeuron
    from synapse import STDPSynapse
    from modulator import create_default_modulators

    # 两个神经元群组，分别 4 个和 3 个神经元
    nrn1 = LIFNeuron(num_neurons=4, num_dendrites=1, tau=10.0, v_threshold=1.0)
    nrn2 = LIFNeuron(num_neurons=3, num_dendrites=1, tau=8.0, v_threshold=0.9)

    # 创建一些突触（仅用于结构完整性，全局监测当前无需它们）
    syn1 = STDPSynapse(num_synapses=6)
    syn2 = STDPSynapse(num_synapses=4)

    # 初始化调制场系统（使用默认五调制原）
    mod_sys = create_default_modulators()

    # 创建全局状态监测器
    monitor = GlobalStateMonitor(
        neurons=[nrn1, nrn2],
        connections=[syn1, syn2],
        optimal_rate=0.1,
        ema_alpha=0.2,
    )

    # 创建调度器
    scheduler = GlobalStateScheduler(monitor, mod_sys)

    # 注册一些调制目标到调制场（神经元和突触）
    # 为简单起见，我们给每个目标赋予一个空间位置
    mod_sys.register_target(nrn1, (0.0, 0.0, 0.0))
    mod_sys.register_target(nrn2, (1.0, 0.0, 0.0))
    mod_sys.register_target(syn1, (0.5, 0.0, 0.0))
    mod_sys.register_target(syn2, (0.5, 1.0, 0.0))

    print("开始模拟 5 个时间步...\n")
    for t in range(5):
        # 每个时间步，神经元接收随机输入以模拟内存活动（公理四：也可零输入）
        # 批次大小为 1
        x1 = torch.randn(1, 1, 4) * 0.5  # (batch, dendrites, neurons)
        x2 = torch.randn(1, 1, 3) * 0.5
        _ = nrn1(x1)
        _ = nrn2(x2)

        # 让突触也活动一下（模拟连接传播）
        pre_spk = torch.randint(0, 2, (1, 6)).float()
        post_spk = torch.randint(0, 2, (1, 6)).float()
        pre_r = torch.randn(1, 6, 1) * 0.1
        post_r = torch.randn(1, 6, 1) * 0.1
        _ = syn1(pre_spk, post_spk, pre_r, post_r)
        pre_spk2 = torch.randint(0, 2, (1, 4)).float()
        post_spk2 = torch.randint(0, 2, (1, 4)).float()
        pre_r2 = torch.randn(1, 4, 1) * 0.1
        post_r2 = torch.randn(1, 4, 1) * 0.1
        _ = syn2(pre_spk2, post_spk2, pre_r2, post_r2)

        # 调度器一步完成“监测 → 调制”
        global_state = scheduler.step()

        print(f"--- 步骤 {t} ---")
        for k, v in global_state.items():
            print(f"  {k}: {v:.4f}")
        # 演示 get_current_novelty 接口
        print(f"  [last_novelty via monitor]: {monitor.get_current_novelty():.4f}")

    print("\n演示完成。")
    print("观察：全局状态指标完全从网络内部活动中衍生，并实时驱动调制场更新。")
    print("满足公理四（存在先于任务）和公理五（价值内生）的完整闭环。")