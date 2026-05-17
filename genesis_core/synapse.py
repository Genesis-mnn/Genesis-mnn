# synapse.py — Genesis 框架预置连接点与可塑性规则
# 基于 ResonanceSynapse 基类实现 PlasticityRule 基类和 STDPSynapse。
# 严格遵循白皮书六条核心公理。

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import math

from .genesis_core import ResonanceSynapse, MODULATOR_REWARD, MODULATOR_FATIGUE, MODULATOR_STRESS


class PlasticityRule(ResonanceSynapse):
    """用户可自定义的可塑性基类。

    继承 `ResonanceSynapse`，保留全部动力学机制（囊泡池、动态阻抗、交往记忆等），
    并将耦合强度更新逻辑抽象为 `_apply_forward_plasticity` 方法。
    子类通过实现该方法可以定制完全局部的、在线的可塑性规则，
    且必须遵守公理二（共鸣取代权重）和公理三（演化优于训练）。

    Attributes:
        继承自 `ResonanceSynapse` 的全部缓冲区与参数。
    """

    def __init__(self,
                 num_synapses: int = 1,
                 pool_total: float = 1.0,
                 k_cyc: float = 0.01,
                 k_rp: float = 0.1,
                 p_base: float = 0.2,
                 ca_decay: float = 0.9,
                 facil_decay: float = 0.8,
                 facil_factor: float = 0.5,
                 memory_size: int = 10,
                 A_plus: float = 0.005,
                 A_minus: float = 0.005,
                 trace_decay: float = 0.9):
        """初始化可塑性基类。参数含义与 `ResonanceSynapse` 相同。"""
        super().__init__(num_synapses=num_synapses,
                         pool_total=pool_total,
                         k_cyc=k_cyc,
                         k_rp=k_rp,
                         p_base=p_base,
                         ca_decay=ca_decay,
                         facil_decay=facil_decay,
                         facil_factor=facil_factor,
                         memory_size=memory_size,
                         A_plus=A_plus,
                         A_minus=A_minus,
                         trace_decay=trace_decay)
        self._current_modulators: Dict[str, float] = {}  # 缓存当前时间步有效调制原

    def forward(self,
                pre_spike: torch.Tensor,
                post_spike: torch.Tensor,
                pre_r: torch.Tensor,
                post_r: torch.Tensor,
                modulator_concentrations: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """执行一个时间步的连接点信息处理，并应用子类定义的可塑性规则。

        参数:
            pre_spike: 连接点前脉冲 (batch, num_synapses)
            post_spike: 连接点后脉冲 (batch, num_synapses)
            pre_r: 连接点前谐振敏感度 (batch, num_synapses, r_dim) —— 用于公理二
            post_r: 连接点后谐振敏感度 (batch, num_synapses, r_dim)
            modulator_concentrations: 临时调制原浓度覆盖（用于测试/外部干预）

        返回:
            post_current: 连接点后电流 (batch, num_synapses)
        """
        # 本方法在父类 ResonanceSynapse.forward 的基础上增加了子类可塑性更新步骤，
        # 其余动力学流程（囊泡池、释放概率、残余钙、逆行信使等）与父类保持一致。
        bsz = pre_spike.shape[0]
        if self.strength.shape[0] != bsz:
            self.reset_state(bsz, pre_spike.device)

        # 合并调制原：外部临时覆盖优先，合并结果缓存供子类可塑性方法访问
        eff_modulators = {**self._modulator_concentrations,
                          **(modulator_concentrations or {})}
        self._current_modulators = eff_modulators

        # 1. 囊泡池恢复（三池模型，白皮书2.1节）
        d_Rcyc2RP = self.k_cyc * self.pool_Rcyc
        d_RP2RRP = self.k_rp * self.pool_RP
        self.pool_Rcyc = self.pool_Rcyc - d_Rcyc2RP
        self.pool_RP = self.pool_RP + d_Rcyc2RP - d_RP2RRP
        self.pool_RRP = self.pool_RRP + d_RP2RRP
        self.pool_Rcyc.clamp_(min=0.0, max=self.pool_total)
        self.pool_RP.clamp_(min=0.0, max=self.pool_total)
        self.pool_RRP.clamp_(min=0.0, max=self.pool_total)

        # 2. 释放概率动态计算（公理二核心实现依赖，白皮书2.2节）
        ca = self.resid_Ca
        K_ca, n = 0.5, 2
        p_ca = ca ** n / (ca ** n + K_ca ** n + 1e-8)
        p_fac = 1.0 + self.facil_factor * self.facil_trace

        mod_effect = 1.0
        if MODULATOR_REWARD in eff_modulators:
            mod_effect += 0.1 * eff_modulators[MODULATOR_REWARD]
        if MODULATOR_FATIGUE in eff_modulators:
            mod_effect -= 0.05 * eff_modulators[MODULATOR_FATIGUE]
        if MODULATOR_STRESS in eff_modulators:
            mod_effect += 0.2 * eff_modulators[MODULATOR_STRESS]

        p_release = self.p_base * p_ca * p_fac * mod_effect
        # 逆行信使调节释放概率（白皮书2.5节逆行）
        p_release = p_release * (1.0 - 0.1 * self.retrograde_signal)
        p_release = torch.clamp(p_release, 0.0, 1.0)

        # 3. 释放过程（消耗可释放池）
        release = pre_spike * p_release * self.pool_RRP
        self.pool_RRP = self.pool_RRP - release
        self.pool_Rcyc = self.pool_Rcyc + release
        self.pool_RRP.clamp_(min=0.0)

        # 4. 动态阻抗与后电流（公理二核心：阻抗 = |r_pre - r_post| 的一般化距离）
        impedance = torch.norm(pre_r - post_r, dim=-1)  # L2 距离，可扩展
        efficiency = 1.0 / (1.0 + impedance)
        post_current = release * efficiency * self.strength

        # 5. 残余钙更新
        ca_influx = 0.1 * pre_spike
        self.resid_Ca = self.ca_decay * self.resid_Ca + ca_influx

        # 6. 易化迹更新（短期可塑性）
        self.facil_trace = self.facil_decay * self.facil_trace + (1.0 - self.facil_decay) * pre_spike

        # 7. 逆行信使更新（双向可塑性的一部分，不依赖全局信号）
        self.retrograde_signal = 0.9 * self.retrograde_signal + 0.1 * post_spike

        # 8. 子类实现的可塑性更新（公理三：局部的、在线的、基于共鸣的演化）
        self._apply_forward_plasticity(pre_spike, post_spike)

        # 8.5 标记与捕获（白皮书2.6节，基类完整实现，确保子类继承）
        self._update_tag_and_capture(pre_spike, impedance, eff_modulators)

        # 9. 交往记忆队列更新（公理二：记录强共鸣事件）
        strong_event = pre_spike * (impedance < 0.1).float()
        if strong_event.any():
            timestamp = self._current_t
            batch_idx, syn_idx = torch.where(strong_event)
            ptr = self.memory_ptr[batch_idx, syn_idx]
            self.memory_buffer[batch_idx, syn_idx, ptr, 0] = timestamp
            self.memory_buffer[batch_idx, syn_idx, ptr, 1] = self.strength[batch_idx, syn_idx]
            self.memory_ptr[batch_idx, syn_idx] = (ptr + 1) % self.memory_size

        self._current_t += 1
        return post_current

    def _update_tag_and_capture(self,
                                 pre_spike: torch.Tensor,
                                 impedance: torch.Tensor,
                                 eff_modulators: Dict[str, float]) -> None:
        """标记与捕获机制：tag 衰减、强共鸣打标、奖励捕获增强强度、tag 消耗。

        对应于白皮书2.6节，实现突触标记与全局奖励信号的捕获。
        该方法从基类 ResonanceSynapse.forward() 中提取，供 PlasticityRule
        及其子类复用，确保 STDPSynapse 等预置连接点也能享有完整的标记与捕获能力。

        参数:
            pre_spike: 当前步连接点前脉冲 (batch, num_synapses)
            impedance: 动态阻抗 (batch, num_synapses)
            eff_modulators: 本步有效的调制原浓度字典
        """
        # 1. tag 衰减（遵循 trace_decay 动力学）
        self.tag = self.tag * self.trace_decay

        # 2. 强共鸣事件打标：当脉冲到来且阻抗极低时，留下可塑性标记
        strong_event = pre_spike * (impedance < 0.1).float()
        self.tag = self.tag + strong_event

        # 3. 奖励捕获：全局奖励信号出现时，将标记转化为强度提升
        reward = eff_modulators.get(MODULATOR_REWARD, 0.0)
        if reward > 0:
            delta = self.A_plus * reward * self.tag
            self.strength = self.strength + delta
            # 消耗 tag（强度提升后标记减弱）
            self.tag = self.tag - reward * self.A_plus * self.tag
            self.tag.clamp_(min=0.0)
            self.strength.clamp_(0.0, 1.0)

    def _apply_forward_plasticity(self, pre_spike: torch.Tensor, post_spike: torch.Tensor) -> None:
        """应用前向可塑性规则，更新耦合强度及相关状态。

        子类必须重写此方法。每个时间步调用，仅接收本步脉冲。
        实现应当基于局部信息（脉冲时序、迹、调制原浓度等）修改 `self.strength`，
        并维护相关的内部状态（例如 `self.pre_trace`、`self.post_trace` 等）。
        不得使用反向传播或全局损失梯度（公理三）。

        参数:
            pre_spike: 当前步连接点前脉冲 (batch, num_synapses)
            post_spike: 当前步连接点后脉冲 (batch, num_synapses)
        """
        raise NotImplementedError("子类必须实现 _apply_forward_plasticity 方法定义强度更新规则。")


class STDPSynapse(PlasticityRule):
    """经典脉冲时序依赖可塑性（STDP）连接点。

    实现标准的 STDP 学习规则：当连接点前脉冲在前、连接点后脉冲在后时，
    耦合强度增强（LTP）；反之减弱（LTD）。所有更新仅依赖局部脉冲时序信息。
    通过查询内部奖赏信号浓度实现三因子调制（公理五），无需外部损失函数。
    严格遵守公理二、公理三、公理五。
    推荐的、功能最完整的 STDP 标准实现

    Attributes:
        tau_pre: 连接点前迹时间常数（步数）
        tau_post: 连接点后迹时间常数（步数）
        lr: 基础学习率
        reward_scale: 奖赏信号对学习率的缩放强度
        pre_decay: 前迹衰减因子
        post_decay: 后迹衰减因子
    """

    def __init__(self,
                 num_synapses: int = 1,
                 pool_total: float = 1.0,
                 k_cyc: float = 0.01,
                 k_rp: float = 0.1,
                 p_base: float = 0.2,
                 ca_decay: float = 0.9,
                 facil_decay: float = 0.8,
                 facil_factor: float = 0.5,
                 memory_size: int = 10,
                 A_plus: float = 0.005,
                 A_minus: float = 0.005,
                 tau_pre: float = 20.0,
                 tau_post: float = 20.0,
                 lr: float = 1.0,
                 reward_scale: float = 1.0):
        """初始化 STDPSynapse。

        新增参数:
            tau_pre: 连接点前脉冲迹衰减时间常数（步数，>0）
            tau_post: 连接点后脉冲迹衰减时间常数（步数，>0）
            lr: 基础学习率，缩放 LTP/LTD 幅度
            reward_scale: 奖赏信号乘数的缩放系数（0 表示无调制）
        其他参数同 `PlasticityRule`。
        """
        # 将通用参数传递至基类，trace_decay 不再直接使用
        super().__init__(num_synapses=num_synapses,
                         pool_total=pool_total,
                         k_cyc=k_cyc,
                         k_rp=k_rp,
                         p_base=p_base,
                         ca_decay=ca_decay,
                         facil_decay=facil_decay,
                         facil_factor=facil_factor,
                         memory_size=memory_size,
                         A_plus=A_plus,
                         A_minus=A_minus)
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.lr = lr
        self.reward_scale = reward_scale

        # 计算离散时间步下的指数衰减系数
        self.pre_decay = math.exp(-1.0 / tau_pre) if tau_pre > 0 else 1.0
        self.post_decay = math.exp(-1.0 / tau_post) if tau_post > 0 else 1.0

    def _apply_forward_plasticity(self, pre_spike: torch.Tensor, post_spike: torch.Tensor) -> None:
        """STDP 规则的具体实现，包含基于奖赏信号的学习率动态调制。

        公理三：仅依赖连接点前后脉冲的时序信息，迹衰减与更新完全局域化。
        公理五：通过查询内部奖赏信号浓度（而非外部梯度）形成三因子调制。
        """
        # 迹衰减（独立时间常数）
        self.pre_trace = self.pre_trace * self.pre_decay
        self.post_trace = self.post_trace * self.post_decay

        # 公理五：从当前有效调制原中查询奖赏信号浓度，作为学习率乘数
        reward = self._current_modulators.get(MODULATOR_REWARD,
                                              self._modulator_concentrations.get(MODULATOR_REWARD, 0.0))
        mod_factor = 1.0 + self.reward_scale * reward

        # 动态学习率 = 基础学习率 × 奖赏乘数
        lr_eff = self.lr * mod_factor

        # LTP：当连接点后脉冲发生时，根据连接点前迹（代表前脉冲历史）增强
        ltp = lr_eff * self.A_plus * post_spike * self.pre_trace
        # LTD：当连接点前脉冲发生时，根据连接点后迹（代表后脉冲历史）减弱
        ltd = lr_eff * self.A_minus * pre_spike * self.post_trace

        self.strength = self.strength + ltp - ltd
        self.strength.clamp_(0.0, 1.0)

        # 将当前脉冲加入迹（用于后续时间步的配对）
        self.pre_trace = self.pre_trace + pre_spike
        self.post_trace = self.post_trace + post_spike