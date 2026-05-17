# learning.py — Genesis 框架学习与训练模块
# 提供替代梯度训练、局部可塑性突触层（STDPLinear）、脉冲网络基础构件
# （MNNBlock、SpikingMNNModel），以及 ANN→MNN 转换相关的辅助组件。
# 严格遵循白皮书六条核心公理，特别是公理三（演化优于训练）和公理五（价值内生）。
#
# 架构设计：
# - 替代梯度模块：提供多种平滑函数和 SpikeFunction，支持用户注册自定义梯度。
# - 脉冲神经元与局部可塑性：包含可微 LIF 神经元（SpikingLIFCell）和
#   基于 ResonanceSynapse 的局部可塑性突触层（STDPLinear），以及
#   MNN 基础构造块（MNNBlock, SpikingMNNModel）。
# - ANN→MNN 转换管线：将 PyTorch ANN 转换为脉冲 MNN，包含权重映射、BN 融合、
#   编码器/解码器包装。转换后的权重不参与梯度，可供后续局部可塑性演化。
# 所有类和方法均包含完整的类型注解和文档字符串。
#【公理合规说明】
#旧版 RL 组件（DQNAgent, A2CAgent, PPOAgent, ReplayBuffer 等）已移除，
#详见 v0.1 归档。未来将引入完全基于公理五的内源性学习系统。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import math

# 从框架核心导入
from .genesis_core import (
    MODULATOR_REWARD, MODULATOR_STRESS, MODULATOR_FATIGUE,
    MODULATOR_SATIETY, MODULATOR_NOVELTY,
    ResonanceSynapse
)
from .convert import ANNtoMNNConverter
from .encoding import RateEncoder


# ============================================================================
# 简单编码器与解码器（若 encoding.py 尚未就绪，提供最小可用实现）
# ============================================================================

class SpikeDecoder:
    """脉冲解码器：将脉冲序列解码为连续信号。

    提供均值、求和等解码方式。
    """

    @staticmethod
    def decode(spike_train: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """解码脉冲序列。

        参数:
            spike_train: (T, batch, *output_shape) 脉冲序列。
            method: 'mean'（平均发放率）或 'sum'（总发放数）。
        返回:
            decoded: (batch, *output_shape) 连续值。
        """
        if method == 'mean':
            return spike_train.mean(dim=0)
        elif method == 'sum':
            return spike_train.sum(dim=0)
        else:
            raise ValueError(f"不支持的解码方法: {method}")


# ============================================================================
# 替代梯度模块（SpikeFunction 与注册表）
# ============================================================================

# 全局替代梯度注册表
_SURROGATE_REGISTRY: Dict[str, Callable] = {}


def register_surrogate(name: str, fn: Callable) -> None:
    """注册用户自定义的替代梯度函数。

    参数:
        name: 函数名称，不得与内置名称重复。
        fn: 替代梯度函数，接受 (v - threshold) 张量并返回梯度张量。
    """
    _SURROGATE_REGISTRY[name] = fn


def get_surrogate(name: str) -> Callable:
    """根据名称获取替代梯度函数。"""
    if name not in _SURROGATE_REGISTRY:
        raise ValueError(f"未注册的替代梯度函数: {name}")
    return _SURROGATE_REGISTRY[name]


# 内置替代梯度（伪导数形式，用于反向传播）

def sigmoid_surrogate(x: torch.Tensor, beta: float = 4.0) -> torch.Tensor:
    """Sigmoid 型梯度：σ(beta*x) 的导数。"""
    s = torch.sigmoid(beta * x)
    return beta * s * (1.0 - s)


def atan_surrogate(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """反正切型梯度：atan(alpha*x) 的导数。"""
    return alpha / (1.0 + (alpha * x) ** 2) / math.pi


def fast_sigmoid_surrogate(x: torch.Tensor, beta: float = 4.0) -> torch.Tensor:
    """快速 S 型梯度：常用于 SNN，形式为 beta / (2 * (1 + |beta*x|)^2) 等。
    此处采用 beta / (1 + |beta*x|)^2 作为平滑近似梯度。
    """
    return beta / (1.0 + torch.abs(beta * x)) ** 2


# 注册内置函数
register_surrogate('sigmoid', sigmoid_surrogate)
register_surrogate('atan', atan_surrogate)
register_surrogate('fast_sigmoid', fast_sigmoid_surrogate)


class SpikeFunction(torch.autograd.Function):
    """脉冲发放函数：前向为阶跃，反向使用替代梯度。

    前向输出为 (v >= threshold).float()，反向时调用指定的替代梯度函数。
    推荐用法：通过 apply 方法以类的形式调用。

    参数:
        v: 膜电位张量 (batch, *shape)
        threshold: 发放阈值标量或与 v 同形张量
        surrogate_type: 替代梯度名称，支持 'sigmoid', 'atan', 'fast_sigmoid' 或自定义注册名。
        **kwargs: 传递给替代梯度函数的额外参数（如 beta, alpha）。
    """

    @staticmethod
    def forward(ctx, v: torch.Tensor, threshold: float = 1.0,
                surrogate_type: str = 'fast_sigmoid', **kwargs):
        ctx.save_for_backward(v)
        ctx.threshold = threshold
        ctx.surrogate_type = surrogate_type
        ctx.kwargs = kwargs
        # 前向：阶跃发放
        return (v >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        v, = ctx.saved_tensors
        x = v - ctx.threshold  # 差值决定梯度幅度

        surrogate_fn = get_surrogate(ctx.surrogate_type)
        grad_input = surrogate_fn(x, **ctx.kwargs)

        # 链式法则：grad_output 乘替代梯度
        return grad_output * grad_input, None, None, None


# ============================================================================
# 可微脉冲神经元与局部可塑性突触
# ============================================================================

class SpikingLIFCell(nn.Module):
    """可微 LIF 脉冲神经元单元。

    动力学：τ du/dt = -u + I，当 u >= v_threshold 时产生输出并重置。
    使用替代梯度使发放操作可微，梯度仅用于优化自身参数（τ, v_threshold）。
    满足公理三：替代梯度不用于演化结构连接。

    状态维护：内部维护膜电位 u（跨时间步），需通过 reset_state 初始化。
    """

    def __init__(self,
                 input_size: int,  # 仅作语义保留，实际由 current 形状决定
                 hidden_size: int,
                 tau: float = 2.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 surrogate_type: str = 'fast_sigmoid',
                 bias_dim: int = 1,
                 **surrogate_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.tau = nn.Parameter(torch.full((hidden_size,), tau))
        self.v_threshold = nn.Parameter(torch.full((hidden_size,), v_threshold))
        self.v_reset = nn.Parameter(torch.full((hidden_size,), v_reset))
        self.surrogate_type = surrogate_type
        self.surrogate_kwargs = surrogate_kwargs
        self.bias_dim = bias_dim

        # 谐振敏感度偏置（对应公理二：内部节律）
        self.bias_r = nn.Parameter(torch.randn(hidden_size, bias_dim) * 0.1)

        # 膜电位缓冲区（不可训练）
        self.register_buffer('u', torch.zeros(0))
        # 谐振敏感度缓冲区
        self.register_buffer('_r', torch.empty(0))
        # 内部调制原浓度存储
        self._modulator_concentrations: Dict[str, float] = {}

    def reset_state(self, batch_size: int):
        """重置膜电位为零，同时初始化谐振敏感度 r。"""
        self.u = torch.zeros(batch_size, self.hidden_size, device=self.tau.device)
        self._r = self.bias_r.unsqueeze(0).expand(batch_size, -1, -1).to(device=self.tau.device)

    def get_r(self) -> torch.Tensor:
        """返回当前的谐振敏感度 r，形状 (batch, hidden_size, bias_dim)。"""
        return self._r

    def apply_modulator(self, name: str, concentration: float):
        """接收调制原信号（公理五）。"""
        self._modulator_concentrations[name] = concentration

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        """单步前向传播。

        参数:
            current: 输入电流 (batch, hidden_size)

        返回:
            spike: 产生输出 (batch, hidden_size)
        """
        if self.u.numel() == 0 or self.u.size(0) != current.size(0):
            self.reset_state(current.size(0))

        # 时间常数约束为正
        tau_eff = torch.clamp(self.tau, min=0.1)
        du = (-self.u + current) / tau_eff.unsqueeze(0)
        self.u = self.u + du

        # 调制原对发放阈值的影响（公理五）
        effective_threshold = self.v_threshold.unsqueeze(0)
        if MODULATOR_STRESS in self._modulator_concentrations:
            effective_threshold = effective_threshold - 0.1 * self._modulator_concentrations[MODULATOR_STRESS]
        if MODULATOR_FATIGUE in self._modulator_concentrations:
            effective_threshold = effective_threshold + 0.1 * self._modulator_concentrations[MODULATOR_FATIGUE]

        spike = SpikeFunction.apply(self.u, effective_threshold, self.surrogate_type, **self.surrogate_kwargs)

        # 重置膜电位
        u_reset = self.v_reset.unsqueeze(0)
        self.u = (1.0 - spike) * self.u + spike * u_reset

        return spike


class STDPLinear(nn.Module):
    """带有局部 STDP 可塑性的全连接突触层（公理二：共鸣取代权重）。

    内部封装 ResonanceSynapse 实例，连接强度由前后神经元的谐振敏感度
    以及局部 STDP 规则共同决定。不再使用传统的标量权重矩阵。
    外部调制原（奖赏信号）可调节突触可塑性，实现三因子学习规则。
    满足公理二、三、五。

    参数:
        in_features: 输入特征数。
        out_features: 输出特征数。
        tau_pre, tau_post: 原始迹时间常数（已映射到 ResonanceSynapse 参数）。
        lr: STDP 基础学习率（与 A_plus/A_minus 相乘后传入 ResonanceSynapse）。
        A_plus, A_minus: 增强/减弱系数。
        reward_scale: 保留参数，实际调制由 ResonanceSynapse 内部调制原系统处理。
        **synapse_kwargs: 传递给 ResonanceSynapse 的额外参数（如 pool_total 等）。
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 tau_pre: float = 20.0,
                 tau_post: float = 20.0,
                 lr: float = 0.01,
                 A_plus: float = 0.005,
                 A_minus: float = 0.005,
                 reward_scale: float = 1.0,
                 **synapse_kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        num_synapses = in_features * out_features

        # 根据时间常数计算迹衰减（取平均）
        trace_decay = math.exp(-1.0 / ((tau_pre + tau_post) / 2))
        # 将基础学习率乘入 STDP 系数，以保持有效学习率一致
        eff_A_plus = lr * A_plus
        eff_A_minus = lr * A_minus

        # 构建 ResonanceSynapse 实例
        syn_kwargs = dict(
            A_plus=eff_A_plus,
            A_minus=eff_A_minus,
            trace_decay=trace_decay,
        )
        syn_kwargs.update(synapse_kwargs)
        self.synapse = ResonanceSynapse(num_synapses=num_synapses, **syn_kwargs)

        # 全连接索引缓冲区（与 utils.connect 中 'full' 规则一致）
        pre_idx = torch.arange(in_features).repeat_interleave(out_features)
        post_idx = torch.arange(out_features).repeat(in_features)
        self.register_buffer('pre_indices', pre_idx)
        self.register_buffer('post_indices', post_idx)

    def apply_modulator(self, name: str, concentration: float):
        """将调制原信号转发到内部的 ResonanceSynapse。"""
        self.synapse.apply_modulator(name, concentration)

    def compute_current(self,
                        pre_spike: torch.Tensor,
                        post_spike: torch.Tensor,
                        pre_r: torch.Tensor,
                        post_r: torch.Tensor) -> torch.Tensor:
        """基于共鸣连接点计算突触后电流。

        参数:
            pre_spike: (batch, in_features) 输入脉冲
            post_spike: (batch, out_features) 上一时间步的后突触脉冲（用于可塑性）
            pre_r: (batch, in_features, r_dim) 前层谐振敏感度
            post_r: (batch, out_features, r_dim) 后层谐振敏感度
        返回:
            current: (batch, out_features) 输出电流
        """
        bsz = pre_spike.size(0)

        # 将前/后层信号展开到每个突触
        pre_spike_exp = pre_spike[:, self.pre_indices]      # (batch, num_synapses)
        post_spike_exp = post_spike[:, self.post_indices]
        pre_r_exp = pre_r[:, self.pre_indices, :]           # (batch, num_synapses, r_dim)
        post_r_exp = post_r[:, self.post_indices, :]

        # 通过共鸣突触计算每个突触的电流（内部进行 STDP 更新等）
        syn_current = self.synapse(pre_spike_exp, post_spike_exp,
                                   pre_r_exp, post_r_exp)   # (batch, num_synapses)

        # 散射求和多突触到后神经元
        post_current = torch.zeros(bsz, self.out_features, device=pre_spike.device)
        post_current = post_current.index_add(1, self.post_indices, syn_current)
        return post_current

    def update_weights(self, post_spike: torch.Tensor):
        """
        权重更新已完全整合在 compute_current 中的 ResonanceSynapse 内部。
        此方法保留为空以兼容旧接口。
        """
        pass


class MNNBlock(nn.Module):
    """MNN 基本构造块：包含一个 STDPLinear 和一个 SpikingLIFCell。

    前向传播流程：
    1. STDPLinear 基于前脉冲、上一时刻的后脉冲以及谐振敏感度计算电流。
    2. SpikingLIFCell 产生当前脉冲。
    3. 返回当前脉冲和当前谐振敏感度，供下一层作为 pre 信息使用。

    满足公理二、三：突触权重完全由共鸣和局部 STDP 决定。
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 tau: float = 2.0,
                 v_threshold: float = 1.0,
                 surrogate_type: str = 'fast_sigmoid',
                 **stdp_kwargs):
        super().__init__()
        self.linear = STDPLinear(in_features, out_features, **stdp_kwargs)
        self.neuron = SpikingLIFCell(0, out_features, tau=tau,
                                     v_threshold=v_threshold, surrogate_type=surrogate_type)

        # 缓存上一时间步的后突触状态（用于突触可塑性计算）
        self.prev_post_spike: Optional[torch.Tensor] = None
        self.prev_post_r: Optional[torch.Tensor] = None

    def reset_state(self, batch_size: int, device: Optional[torch.device] = None):
        """重置神经元和突触的内部状态，并清空缓存。"""
        if device is None:
            device = self.neuron.tau.device
        self.neuron.reset_state(batch_size)
        self.linear.synapse.reset_state(batch_size, device)
        self.prev_post_spike = None
        self.prev_post_r = None

    def apply_modulator(self, name: str, concentration: float):
        self.linear.apply_modulator(name, concentration)
        self.neuron.apply_modulator(name, concentration)

    def forward(self,
                pre_spike: torch.Tensor,
                pre_r: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步前向传播。

        参数:
            pre_spike: (batch, in_features) 前层脉冲。
            pre_r: (batch, in_features, r_dim) 前层谐振敏感度；若为 None 则自动创建全零张量。

        返回:
            post_spike: (batch, out_features) 当前层脉冲。
            post_r: (batch, out_features, r_dim) 当前层谐振敏感度，供下一层使用。
        """
        bsz = pre_spike.size(0)

        # 若未传入 pre_r，则创建默认全零 r（例如网络的第一层）
        if pre_r is None:
            r_dim = self.neuron.bias_r.size(1)
            pre_r = torch.zeros(bsz, self.linear.in_features, r_dim, device=pre_spike.device)

        # 初始化或校验缓存的上一时刻 post 状态
        if self.prev_post_spike is None or self.prev_post_spike.size(0) != bsz:
            self.prev_post_spike = torch.zeros(bsz, self.linear.out_features, device=pre_spike.device)
        if self.prev_post_r is None or self.prev_post_r.size(0) != bsz:
            # 初始时取当前神经元 r 作为上一时刻的近似
            self.prev_post_r = self.neuron.get_r().detach()

        # 获取当前后神经元的谐振敏感度（用于电流计算中的阻抗）
        post_r = self.neuron.get_r()

        # 计算突触后电流（内部同时进行基于上一时刻 post 脉冲的局部 STDP 更新）
        current = self.linear.compute_current(pre_spike, self.prev_post_spike, pre_r, post_r)

        # 神经元产生当前脉冲
        post_spike = self.neuron(current)

        # 更新缓存，供下一时间步使用
        self.prev_post_spike = post_spike.detach()
        self.prev_post_r = post_r.detach()

        return post_spike, self.prev_post_r


# ============================================================================
# 用于强化学习的完整脉冲网络模型 (SpikingMNNModel)
# ============================================================================

class SpikingMNNModel(nn.Module):
    """面向强化学习的脉冲 MNN 模型。

    包含：速率编码器 -> 多个 MNNBlock -> 脉冲解码器。
    可处理连续状态输入，输出连续值（如 Q 值或策略 logits）。
    内部突触通过局部 STDP 演化，神经元参数可通过替代梯度优化。

    参数:
        input_dim: 状态空间维度。
        hidden_dims: 隐藏层维度列表。
        output_dim: 输出维度（动作数或值）。
        T: 时间步数。
        tau, v_threshold: 神经元参数。
        surrogate_type: 替代梯度类型。
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 T: int = 8,
                 tau: float = 2.0,
                 v_threshold: float = 1.0,
                 surrogate_type: str = 'fast_sigmoid',
                 **stdp_kwargs):
        super().__init__()
        self.T = T
        self.encoder = RateEncoder(T)
        self.decoder = SpikeDecoder()

        blocks = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            blocks.append(MNNBlock(in_dim, h_dim, tau=tau, v_threshold=v_threshold,
                                   surrogate_type=surrogate_type, **stdp_kwargs))
            in_dim = h_dim
        blocks.append(MNNBlock(in_dim, output_dim, tau=tau, v_threshold=v_threshold,
                               surrogate_type=surrogate_type, **stdp_kwargs))
        self.blocks = nn.ModuleList(blocks)

    def apply_modulator(self, name: str, concentration: float):
        """向所有子模块传递调制原浓度。"""
        for block in self.blocks:
            block.apply_modulator(name, concentration)

    def reset_state(self, batch_size: int):
        """重置所有脉冲神经元以及突触的内部状态。"""
        for block in self.blocks:
            block.reset_state(batch_size)

    def forward(self, x_cont: torch.Tensor) -> torch.Tensor:
        """前向传播：编码 → 脉冲动力学 → 解码。

        参数:
            x_cont: (batch, input_dim) 连续值状态
        返回:
            out_cont: (batch, output_dim) 解码后的连续信号
        """
        spikes = self.encoder.encode(x_cont)  # (T, batch, input_dim)
        batch_size = x_cont.size(0)
        self.reset_state(batch_size)

        output_spikes = []
        for t in range(self.T):
            x = spikes[t]
            r = None  # 第一层的 pre_r 未知，由 block 内部生成默认值
            for block in self.blocks:
                x, r = block(x, r)  # 返回 post_spike, post_r
            output_spikes.append(x)

        out_seq = torch.stack(output_spikes, dim=0)  # (T, batch, output_dim)
        out_cont = self.decoder.decode(out_seq, method='mean')
        return out_cont


# ============================================================================
# 内源性学习调度器 (IntrinsicLearningScheduler)
# ============================================================================

class IntrinsicLearningScheduler:
    """内源性学习调度器（公理五：价值内生）。

    负责统一调度所有星形胶质细胞的内源性学习步，与 v0.1 的 GlobalStateScheduler
    协同工作。它从全局状态调度器获取当前新奇度，并依次调用每个 Astrocyte 的
    intrinsic_learning_step，最后汇总并返回学习统计信息。
    """

    def __init__(self, glial_network: 'GlialNetwork', global_state_scheduler: 'GlobalStateScheduler'):
        """
        参数:
            glial_network: GlialNetwork 实例，包含所有星形胶质细胞。
            global_state_scheduler: GlobalStateScheduler 实例，用于获取全局新奇度。
        """
        self.glial_network = glial_network
        self.global_state_scheduler = global_state_scheduler

    def step(self) -> Dict[str, int]:
        """执行一次内源性学习步，在所有胶质细胞中应用基于预测误差的学习规则。

        返回:
            统计字典，包含 'pruned_synapses' 和 'enhanced_synapses'。
        """
        # 从全局状态调度器的 monitor 获取当前新奇度
        novelty = self.global_state_scheduler.monitor.get_current_novelty()

        total_pruned = 0
        total_enhanced = 0

        for astro in self.glial_network.astrocytes:
            pruned, enhanced = astro.intrinsic_learning_step(novelty)
            total_pruned += pruned
            total_enhanced += enhanced

        return {
            'pruned_synapses': total_pruned,
            'enhanced_synapses': total_enhanced
        }