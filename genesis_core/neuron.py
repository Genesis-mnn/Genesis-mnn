# neuron.py — Genesis 框架预置神经元类型
# 基于 MorphoNeuron 基类实现六种经典发放模型。
# 严格遵循白皮书六条核心公理。

# 注意：IF/LIF/PLIF/QIF/EIF 等经典模型默认使用透明树突处理（简单平均输入），
# 以保持与标准教材动力学一致。若需使用完整树突棘可塑性及多室计算，
# 请直接使用 MorphoNeuron 基类并配置 num_dendrites > 1。

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .genesis_core import MorphoNeuron, MODULATOR_STRESS, MODULATOR_FATIGUE


class IFNeuron(MorphoNeuron):
    """无泄漏积分-发放神经元 (Integrate-and-Fire without leak)。

    仅累积输入，无泄漏项。膜电位持续累加内部/外部电流，
    达到阈值后发放并重置。
    """

    def __init__(self,
                 num_neurons: int = 1,
                 num_dendrites: int = 1,
                 v_reset: float = 0.0,
                 v_threshold: float = 1.0,
                 rest_h: float = 0.7,
                 refractory_period: int = 2,
                 axon_delay: int = 1,
                 bias_dim: int = 1):
        super().__init__(num_neurons=num_neurons,
                         num_dendrites=num_dendrites,
                         tau_u=1e9,
                         rest_h=rest_h,
                         threshold=v_threshold,
                         refractory_period=refractory_period,
                         axon_delay=axon_delay,
                         bias_dim=bias_dim)
        self.register_buffer('v_reset', torch.full((num_neurons,), v_reset))

    def forward(self,
                x: Optional[torch.Tensor] = None,
                modulator_concentrations: Optional[Dict[str, float]] = None) -> torch.Tensor:
        # 允许输入形状 (batch, N)，内部扩展为 (batch, 1, N)
        if x is not None and x.dim() == 2:
            x = x.unsqueeze(1)
        return super().forward(x, modulator_concentrations)

    def _update_dendrites(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        """简化树突整合：直接平均树突输入，去除复杂棘动态。"""
        batch_size = self._u.shape[0]
        if x is None:
            return torch.zeros(batch_size, self.num_neurons, device=self._u.device)
        return x.mean(dim=1)

    def _update_soma(self,
                     v_dend: torch.Tensor,
                     modulators: Dict[str, float]) -> torch.Tensor:
        # 公理一：健康度回归驱动内在电流，维持稳态
        # 公理四：自发活动噪声保障零输入下不自闭
        # 公理五：调制原通过改变内在电流间接影响发放行为
        self._refractory_counter = torch.clamp(self._refractory_counter - 1, min=0)
        in_ref = (self._refractory_counter > 0).float()
        not_ref = 1.0 - in_ref

        # 内在电流（健康度维持 + 基态噪声）
        I_homeo = -0.1 * (self._h - self.rest_h.unsqueeze(0))
        I_noise = 0.02 * torch.randn_like(self._u) * (1.0 + self._r.norm(dim=-1))
        I_int = I_homeo + I_noise
        if MODULATOR_STRESS in modulators:
            I_int += 0.1 * modulators[MODULATOR_STRESS]
        if MODULATOR_FATIGUE in modulators:
            I_int -= 0.05 * modulators[MODULATOR_FATIGUE]

        I_total = v_dend + I_int
        # 无泄漏积分：du = I_total
        du = I_total
        self._u = not_ref * (self._u + du) + in_ref * self._u

        spike = not_ref * (self._u >= self.threshold.unsqueeze(0)).float()
        self._u = (1.0 - spike) * self._u + spike * self.v_reset
        self._refractory_counter = torch.where(
            spike > 0,
            torch.full_like(self._refractory_counter, self.refractory_period),
            self._refractory_counter
        )
        return spike


class LIFNeuron(MorphoNeuron):
    """泄漏积分-发放神经元 (Leaky Integrate-and-Fire)。

    膜电位向静息电位指数衰减，支持不应期和阈值发放。
    参数：v_threshold, v_reset, tau (固定), R。
    """

    def __init__(self,
                 num_neurons: int = 1,
                 num_dendrites: int = 1,
                 tau: float = 10.0,
                 R: float = 1.0,
                 u_rest: float = 0.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 rest_h: float = 0.7,
                 refractory_period: int = 2,
                 axon_delay: int = 1,
                 bias_dim: int = 1):
        super().__init__(num_neurons=num_neurons,
                         num_dendrites=num_dendrites,
                         tau_u=tau,
                         rest_h=rest_h,
                         threshold=v_threshold,
                         refractory_period=refractory_period,
                         axon_delay=axon_delay,
                         bias_dim=bias_dim)
        self.register_buffer('tau', torch.full((num_neurons,), tau))  # 固定时间常数
        self.R = nn.Parameter(torch.full((num_neurons,), R))
        self.register_buffer('u_rest', torch.full((num_neurons,), u_rest))
        self.register_buffer('v_reset', torch.full((num_neurons,), v_reset))

    def forward(self,
                x: Optional[torch.Tensor] = None,
                modulator_concentrations: Optional[Dict[str, float]] = None) -> torch.Tensor:
        if x is not None and x.dim() == 2:
            x = x.unsqueeze(1)
        return super().forward(x, modulator_concentrations)

    def _update_dendrites(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = self._u.shape[0]
        if x is None:
            return torch.zeros(batch_size, self.num_neurons, device=self._u.device)
        return x.mean(dim=1)

    def _update_soma(self,
                     v_dend: torch.Tensor,
                     modulators: Dict[str, float]) -> torch.Tensor:
        self._refractory_counter = torch.clamp(self._refractory_counter - 1, min=0)
        in_ref = (self._refractory_counter > 0).float()
        not_ref = 1.0 - in_ref

        # 公理一、四：内在电流维持稳态与自持活动
        I_homeo = -0.1 * (self._h - self.rest_h.unsqueeze(0))
        I_noise = 0.02 * torch.randn_like(self._u) * (1.0 + self._r.norm(dim=-1))
        I_int = I_homeo + I_noise
        # 公理五：调制原作用于内部状态
        if MODULATOR_STRESS in modulators:
            I_int += 0.1 * modulators[MODULATOR_STRESS]
        if MODULATOR_FATIGUE in modulators:
            I_int -= 0.05 * modulators[MODULATOR_FATIGUE]

        I_total = v_dend + I_int
        # 泄漏积分-发放方程：τ du/dt = -(u - u_rest) + R I
        du = (-(self._u - self.u_rest) + self.R.unsqueeze(0) * I_total) / self.tau.unsqueeze(0)
        self._u = not_ref * (self._u + du) + in_ref * self._u

        spike = not_ref * (self._u >= self.threshold.unsqueeze(0)).float()
        self._u = (1.0 - spike) * self._u + spike * self.v_reset
        self._refractory_counter = torch.where(
            spike > 0,
            torch.full_like(self._refractory_counter, self.refractory_period),
            self._refractory_counter
        )
        return spike


class PLIFNeuron(MorphoNeuron):
    """参数化泄漏积分-发放神经元 (Parametric LIF)。

    膜时间常数 τ 为可学习参数，可通过梯度优化。
    其他参数同经典 LIF。
    """

    def __init__(self,
                 num_neurons: int = 1,
                 num_dendrites: int = 1,
                 tau: float = 10.0,
                 R: float = 1.0,
                 u_rest: float = 0.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 rest_h: float = 0.7,
                 refractory_period: int = 2,
                 axon_delay: int = 1,
                 bias_dim: int = 1):
        super().__init__(num_neurons=num_neurons,
                         num_dendrites=num_dendrites,
                         tau_u=tau,
                         rest_h=rest_h,
                         threshold=v_threshold,
                         refractory_period=refractory_period,
                         axon_delay=axon_delay,
                         bias_dim=bias_dim)
        self.tau = nn.Parameter(torch.full((num_neurons,), tau))  # 可学习时间常数
        self.R = nn.Parameter(torch.full((num_neurons,), R))
        self.register_buffer('u_rest', torch.full((num_neurons,), u_rest))
        self.register_buffer('v_reset', torch.full((num_neurons,), v_reset))

    def forward(self,
                x: Optional[torch.Tensor] = None,
                modulator_concentrations: Optional[Dict[str, float]] = None) -> torch.Tensor:
        if x is not None and x.dim() == 2:
            x = x.unsqueeze(1)
        return super().forward(x, modulator_concentrations)

    def _update_dendrites(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = self._u.shape[0]
        if x is None:
            return torch.zeros(batch_size, self.num_neurons, device=self._u.device)
        return x.mean(dim=1)

    def _update_soma(self,
                     v_dend: torch.Tensor,
                     modulators: Dict[str, float]) -> torch.Tensor:
        self._refractory_counter = torch.clamp(self._refractory_counter - 1, min=0)
        in_ref = (self._refractory_counter > 0).float()
        not_ref = 1.0 - in_ref

        I_homeo = -0.1 * (self._h - self.rest_h.unsqueeze(0))
        I_noise = 0.02 * torch.randn_like(self._u) * (1.0 + self._r.norm(dim=-1))
        I_int = I_homeo + I_noise
        if MODULATOR_STRESS in modulators:
            I_int += 0.1 * modulators[MODULATOR_STRESS]
        if MODULATOR_FATIGUE in modulators:
            I_int -= 0.05 * modulators[MODULATOR_FATIGUE]

        I_total = v_dend + I_int
        du = (-(self._u - self.u_rest) + self.R.unsqueeze(0) * I_total) / self.tau.unsqueeze(0)
        self._u = not_ref * (self._u + du) + in_ref * self._u

        spike = not_ref * (self._u >= self.threshold.unsqueeze(0)).float()
        self._u = (1.0 - spike) * self._u + spike * self.v_reset
        self._refractory_counter = torch.where(
            spike > 0,
            torch.full_like(self._refractory_counter, self.refractory_period),
            self._refractory_counter
        )
        return spike


class QIFNeuron(MorphoNeuron):
    """二次积分-发放神经元 (Quadratic Integrate-and-Fire)。

    动力学包含二次非线性项 a*(u - u_rest)*(u - u_c)，
    可表现 I 类兴奋性。参数：a, u_rest, u_c, tau, R。
    """

    def __init__(self,
                 num_neurons: int = 1,
                 num_dendrites: int = 1,
                 a: float = 1.0,
                 u_rest: float = 0.0,
                 u_c: float = 1.0,
                 tau: float = 10.0,
                 R: float = 1.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 rest_h: float = 0.7,
                 refractory_period: int = 2,
                 axon_delay: int = 1,
                 bias_dim: int = 1):
        super().__init__(num_neurons=num_neurons,
                         num_dendrites=num_dendrites,
                         tau_u=tau,
                         rest_h=rest_h,
                         threshold=v_threshold,
                         refractory_period=refractory_period,
                         axon_delay=axon_delay,
                         bias_dim=bias_dim)
        self.a = nn.Parameter(torch.full((num_neurons,), a))
        self.u_rest = nn.Parameter(torch.full((num_neurons,), u_rest))
        self.u_c = nn.Parameter(torch.full((num_neurons,), u_c))
        self.tau = nn.Parameter(torch.full((num_neurons,), tau))
        self.R = nn.Parameter(torch.full((num_neurons,), R))
        self.register_buffer('v_reset', torch.full((num_neurons,), v_reset))

    def forward(self,
                x: Optional[torch.Tensor] = None,
                modulator_concentrations: Optional[Dict[str, float]] = None) -> torch.Tensor:
        if x is not None and x.dim() == 2:
            x = x.unsqueeze(1)
        return super().forward(x, modulator_concentrations)

    def _update_dendrites(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = self._u.shape[0]
        if x is None:
            return torch.zeros(batch_size, self.num_neurons, device=self._u.device)
        return x.mean(dim=1)

    def _update_soma(self,
                     v_dend: torch.Tensor,
                     modulators: Dict[str, float]) -> torch.Tensor:
        self._refractory_counter = torch.clamp(self._refractory_counter - 1, min=0)
        in_ref = (self._refractory_counter > 0).float()
        not_ref = 1.0 - in_ref

        I_homeo = -0.1 * (self._h - self.rest_h.unsqueeze(0))
        I_noise = 0.02 * torch.randn_like(self._u) * (1.0 + self._r.norm(dim=-1))
        I_int = I_homeo + I_noise
        if MODULATOR_STRESS in modulators:
            I_int += 0.1 * modulators[MODULATOR_STRESS]
        if MODULATOR_FATIGUE in modulators:
            I_int -= 0.05 * modulators[MODULATOR_FATIGUE]

        I_total = v_dend + I_int
        # 二次非线性项：a*(u - u_rest)*(u - u_c)
        quad_term = (self.a.unsqueeze(0) *
                     (self._u - self.u_rest.unsqueeze(0)) *
                     (self._u - self.u_c.unsqueeze(0)))
        du = (quad_term + self.R.unsqueeze(0) * I_total) / self.tau.unsqueeze(0)
        self._u = not_ref * (self._u + du) + in_ref * self._u

        spike = not_ref * (self._u >= self.threshold.unsqueeze(0)).float()
        self._u = (1.0 - spike) * self._u + spike * self.v_reset
        self._refractory_counter = torch.where(
            spike > 0,
            torch.full_like(self._refractory_counter, self.refractory_period),
            self._refractory_counter
        )
        return spike


class EIFNeuron(MorphoNeuron):
    """指数积分-发放神经元 (Exponential Integrate-and-Fire)。

    在软阈值附近引入指数项，模拟动作电位的快速上升。
    参数：tau, R, rest, delta_T, theta_rh。
    """

    def __init__(self,
                 num_neurons: int = 1,
                 num_dendrites: int = 1,
                 tau: float = 10.0,
                 R: float = 1.0,
                 rest: float = 0.0,
                 delta_T: float = 1.0,
                 theta_rh: float = 0.8,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 rest_h: float = 0.7,
                 refractory_period: int = 2,
                 axon_delay: int = 1,
                 bias_dim: int = 1):
        super().__init__(num_neurons=num_neurons,
                         num_dendrites=num_dendrites,
                         tau_u=tau,
                         rest_h=rest_h,
                         threshold=v_threshold,
                         refractory_period=refractory_period,
                         axon_delay=axon_delay,
                         bias_dim=bias_dim)
        self.tau = nn.Parameter(torch.full((num_neurons,), tau))
        self.R = nn.Parameter(torch.full((num_neurons,), R))
        self.rest = nn.Parameter(torch.full((num_neurons,), rest))
        self.delta_T = nn.Parameter(torch.full((num_neurons,), delta_T))
        self.theta_rh = nn.Parameter(torch.full((num_neurons,), theta_rh))
        self.register_buffer('v_reset', torch.full((num_neurons,), v_reset))

    def forward(self,
                x: Optional[torch.Tensor] = None,
                modulator_concentrations: Optional[Dict[str, float]] = None) -> torch.Tensor:
        if x is not None and x.dim() == 2:
            x = x.unsqueeze(1)
        return super().forward(x, modulator_concentrations)

    def _update_dendrites(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = self._u.shape[0]
        if x is None:
            return torch.zeros(batch_size, self.num_neurons, device=self._u.device)
        return x.mean(dim=1)

    def _update_soma(self,
                     v_dend: torch.Tensor,
                     modulators: Dict[str, float]) -> torch.Tensor:
        self._refractory_counter = torch.clamp(self._refractory_counter - 1, min=0)
        in_ref = (self._refractory_counter > 0).float()
        not_ref = 1.0 - in_ref

        I_homeo = -0.1 * (self._h - self.rest_h.unsqueeze(0))
        I_noise = 0.02 * torch.randn_like(self._u) * (1.0 + self._r.norm(dim=-1))
        I_int = I_homeo + I_noise
        if MODULATOR_STRESS in modulators:
            I_int += 0.1 * modulators[MODULATOR_STRESS]
        if MODULATOR_FATIGUE in modulators:
            I_int -= 0.05 * modulators[MODULATOR_FATIGUE]

        I_total = v_dend + I_int
        # 指数项：delta_T * exp((u - theta_rh) / delta_T)
        exp_arg = torch.clamp((self._u - self.theta_rh.unsqueeze(0)) / self.delta_T.unsqueeze(0),
                              max=20.0, min=-20.0)
        exp_term = self.delta_T.unsqueeze(0) * torch.exp(exp_arg)
        du = (-(self._u - self.rest.unsqueeze(0)) + exp_term + self.R.unsqueeze(0) * I_total) / self.tau.unsqueeze(0)
        self._u = not_ref * (self._u + du) + in_ref * self._u

        spike = not_ref * (self._u >= self.threshold.unsqueeze(0)).float()
        self._u = (1.0 - spike) * self._u + spike * self.v_reset
        self._refractory_counter = torch.where(
            spike > 0,
            torch.full_like(self._refractory_counter, self.refractory_period),
            self._refractory_counter
        )
        return spike


class IzhikevichNeuron(MorphoNeuron):
    """二阶非线性神经元 (Izhikevich 模型)。

    包含膜电位 v 和恢复变量 w，可产生多种发放模式。
    参数：a, b, c, d。发放阈值 v_threshold 默认为 30。
    """

    def __init__(self,
                 num_neurons: int = 1,
                 num_dendrites: int = 1,
                 a: float = 0.02,
                 b: float = 0.2,
                 c: float = -65.0,
                 d: float = 8.0,
                 v_threshold: float = 30.0,
                 rest_h: float = 0.7,
                 homeo_gain: float = 1.0,
                 noise_std: float = 0.5,
                 refractory_period: int = 0,
                 axon_delay: int = 1,
                 bias_dim: int = 1):
        # 提前注册恢复变量 w，保证 reset_state 中可访问
        self.register_buffer('_w', torch.empty(0))
        super().__init__(num_neurons=num_neurons,
                         num_dendrites=num_dendrites,
                         tau_u=1.0,
                         rest_h=rest_h,
                         threshold=v_threshold,
                         refractory_period=refractory_period,
                         axon_delay=axon_delay,
                         bias_dim=bias_dim)
        self.a = nn.Parameter(torch.full((num_neurons,), a))
        self.b = nn.Parameter(torch.full((num_neurons,), b))
        self.c = nn.Parameter(torch.full((num_neurons,), c))
        self.d = nn.Parameter(torch.full((num_neurons,), d))
        self.homeo_gain = homeo_gain
        self.noise_std = noise_std

    def reset_state(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        """重置全部状态，包括恢复变量 w。"""
        super().reset_state(batch_size, device)
        self._w = torch.zeros(batch_size, self.num_neurons, device=self._u.device)

    def forward(self,
                x: Optional[torch.Tensor] = None,
                modulator_concentrations: Optional[Dict[str, float]] = None) -> torch.Tensor:
        if x is not None and x.dim() == 2:
            x = x.unsqueeze(1)
        return super().forward(x, modulator_concentrations)

    def _update_dendrites(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = self._u.shape[0]
        if x is None:
            return torch.zeros(batch_size, self.num_neurons, device=self._u.device)
        return x.mean(dim=1)

    def _update_soma(self,
                     v_dend: torch.Tensor,
                     modulators: Dict[str, float]) -> torch.Tensor:
        # 公理一：健康度回归（系数可调，适配 Izhikevich 模型尺度）
        I_homeo = -self.homeo_gain * (self._h - self.rest_h.unsqueeze(0))
        I_noise = self.noise_std * torch.randn_like(self._u) * (1.0 + self._r.norm(dim=-1))
        I_int = I_homeo + I_noise
        # 公理五：调制原通过内在电流作用于膜电位
        if MODULATOR_STRESS in modulators:
            I_int += 0.1 * modulators[MODULATOR_STRESS]
        if MODULATOR_FATIGUE in modulators:
            I_int -= 0.05 * modulators[MODULATOR_FATIGUE]

        I_total = v_dend + I_int

        # Izhikevich 二阶动力学
        dv = 0.04 * self._u ** 2 + 5 * self._u + 140 - self._w + I_total
        dw = self.a.unsqueeze(0) * (self.b.unsqueeze(0) * self._u - self._w)
        self._u = self._u + dv
        self._w = self._w + dw

        # 发放与重置
        spike = (self._u >= self.threshold.unsqueeze(0)).float()
        self._u = (1.0 - spike) * self._u + spike * self.c.unsqueeze(0)
        self._w = self._w + spike * self.d.unsqueeze(0)

        return spike

    def get_internal_state(self) -> Dict[str, torch.Tensor]:
        """返回包含恢复变量 w 的完整内部状态快照。"""
        state = super().get_internal_state()
        state['w'] = self._w.clone()
        return state