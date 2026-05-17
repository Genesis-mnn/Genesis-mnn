# encoding.py
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    """编码器基类，定义脉冲编码的统一接口。

    对应 Genesis 白皮书 10.3 节。所有自定义编码器需继承本类并实现
    ``encode`` 方法。本类不包含可训练参数，仅保存时间窗口长度 T。

    参数:
        T (int): 时间窗口长度（仿真步数）。
    """

    def __init__(self, T: int):
        super().__init__()
        if not isinstance(T, int) or T <= 0:
            raise ValueError("T 必须为正整数。")
        self.T = T

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """将实数值张量转换为脉冲序列。

        参数:
            x (torch.Tensor): 输入张量，形状为 ``(batch, *features)``。
                各元素通常应归一化到 [0,1] 区间，具体范围由子类定义。

        返回:
            torch.Tensor: 脉冲序列，形状 ``(T, batch, *features)``，
                元素为 0 或 1。
        """
        raise NotImplementedError


class RateEncoder(BaseEncoder):
    """速率编码器：基于泊松过程生成脉冲序列。

    将输入强度映射为每时间步的发放概率，通过伯努利采样生成脉冲。
    对应白皮书 10.3 节“速率编码”。

    参数:
        T (int): 时间窗口长度。
        f_max (float): 最大发放率（每时间步的最大发放概率），默认 1.0。
            x=1 时对应发放概率 f_max。
    """

    def __init__(self, T: int, f_max: float = 1.0):
        super().__init__(T)
        self.f_max = f_max

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """执行速率编码。

        参数:
            x (torch.Tensor): 输入强度，形状 ``(batch, *features)``。
                建议将值归一化至 [0,1]。

        返回:
            torch.Tensor: 脉冲序列，形状 ``(T, batch, *features)``。
        """
        # 输入合法性检查：确保输入强度在 [0,1] 区间
        x = torch.clamp(x, 0.0, 1.0)
        # 将输入映射为发放概率，钳位保证概率在 [0, 1]
        p = torch.clamp(x * self.f_max, 0.0, 1.0)  # (batch, *features)

        # 为每个时间步生成随机数并比较
        rand = torch.rand((self.T,) + p.shape, device=x.device, dtype=x.dtype)
        spikes = (rand < p.unsqueeze(0)).float()  # (T, batch, *features)
        return spikes


class TemporalEncoder(BaseEncoder):
    """时间编码器 (TTFS)：将输入强度映射为精确脉冲延迟。

    强度越大，延迟越小。每个特征至多产生一个脉冲。
    支持自定义强度-延迟映射曲线。
    对应白皮书 10.3 节“时间编码”。

    参数:
        T (int): 时间窗口长度（仿真步数）。
        max_t (float): 最大延迟时间（与 T 同单位）。默认 10.0。
        mapping_fn (Optional[Callable[[torch.Tensor], torch.Tensor]]): 自定义强度->延迟映射函数。
            接受 torch.Tensor 并返回同形状的延迟张量（浮点）。
            若未提供，默认线性映射：``delay = (1 - x) * max_t``，
            其中 x 应位于 [0,1]。
    """

    def __init__(self, T: int, max_t: float = 10.0, mapping_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__(T)
        self.max_t = max_t
        if mapping_fn is None:
            # 默认映射：x ∈ [0,1] -> 延迟 ∈ [0, max_t]
            self.mapping_fn = lambda x: (1.0 - x) * max_t
        else:
            self.mapping_fn = mapping_fn

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """执行时间编码。

        参数:
            x (torch.Tensor): 输入强度，形状 ``(batch, *features)``。
                若使用默认映射，值应归一化至 [0,1]。

        返回:
            torch.Tensor: 脉冲序列，形状 ``(T, batch, *features)``。
                若延迟超出 T-1，对应位置在整个窗口内均不放发（全零）。
        """
        # 注意：若使用默认映射，输入 x 应位于 [0,1]；自定义映射时请按需调整。
        # 计算浮点延迟
        delay = self.mapping_fn(x)  # shape = input_shape
        delay_int = torch.round(delay).long()

        # 钳位到 [0, T]；延迟 >= T 的将通过 one-hot 截断实现无脉冲
        T = self.T
        delay_clamped = torch.clamp(delay_int, 0, T)

        # one-hot 编码，类别数 T+1，再切除最后一个类别（索引 T）
        one_hot = F.one_hot(delay_clamped, num_classes=T + 1)  # (*features, T+1)
        spikes = one_hot[..., :T].movedim(-1, 0).float()        # (T, *features)
        return spikes


class PhaseEncoder(BaseEncoder):
    """相位编码器：将输入映射为背景节律上的发放窗口位置。

    利用周期性振荡，在输入决定的相位区间内产生输出。
    对应白皮书 10.3 节“相位编码”。

    参数:
        T (int): 时间窗口长度（仿真步数）。
        T_osc (int): 振荡周期（步数），决定背景节律频率。
        phi (float): 相位偏移（弧度），用于整体旋转窗口位置。默认 0。
        phase_width (float): 相位窗口宽度（弧度），在此范围内产生脉冲。
            默认 π/4 (45°)。
        scale (float): 将输入 x 映射到相位偏移的缩放因子。
            默认 2π，即 x∈[0,1] 时覆盖整个周期。
    """

    def __init__(self, T: int, T_osc: int, phi: float = 0.0,
                 phase_width: float = math.pi / 4, scale: float = 2 * math.pi):
        super().__init__(T)
        if T_osc <= 0:
            raise ValueError("T_osc 必须为正整数。")
        self.T_osc = T_osc
        self.phi = phi
        self.phase_width = phase_width
        self.scale = scale

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """执行相位编码。

        参数:
            x (torch.Tensor): 输入强度，形状 ``(batch, *features)``。
                使用默认 scale 时，x ∈ [0,1] 映射到 [0, 2π] 的相位偏移。

        返回:
            torch.Tensor: 脉冲序列，形状 ``(T, batch, *features)``。
                无输入时仍可能根据背景节律产生脉冲（符合公理四）。
        """
        # 注意：输入 x 应位于 [0,1]（默认 scale=2π）；若修改 scale 请相应调整输入范围。
        T = self.T
        T_osc = self.T_osc
        phi = self.phi
        half_width = self.phase_width / 2.0
        scale = self.scale
        device = x.device
        input_shape = x.shape

        # 时间步索引及当前振荡相位（不包含固定偏移 phi）
        t = torch.arange(T, device=device, dtype=x.dtype)
        current_phase = (2 * math.pi * (t % T_osc) / T_osc) % (2 * math.pi)  # (T,)

        # 输入决定的发放窗口中心相位
        window_center = (phi + x * scale) % (2 * math.pi)  # input_shape

        # 广播计算每个时间步与窗口中心的角距离
        # 将 current_phase 变形为 (T, 1, 1, ...) 以便广播
        cp = current_phase.view(T, *([1] * len(input_shape)))
        wc = window_center  # input_shape (缺少的时间维度会在前面自动补 1)
        diff = torch.abs(cp - wc)
        diff = torch.min(diff, 2 * math.pi - diff)  # 环形距离

        spikes = (diff < half_width).float()  # (T, *input_shape)
        return spikes


class Decoder:
    """解码器：从脉冲序列中重建原始信号。

    提供三种逆映射方法，用于生命体闭环交互。
    对应白皮书 10.3 节“解码接口”。
    """

    @staticmethod
    def decode_rate(spikes: torch.Tensor, f_max: float = 1.0) -> torch.Tensor:
        """速率解码：从发放率估计输入强度。

        参数:
            spikes (torch.Tensor): 脉冲序列，形状 ``(T, *features)``。
            f_max (float): 编码时使用的最大发放率。默认 1.0。

        返回:
            torch.Tensor: 重建的输入强度，形状 ``(*features)``。
        """
        rate = spikes.mean(dim=0)  # 沿时间平均
        x_hat = rate / f_max
        return x_hat

    @staticmethod
    def decode_temporal(spikes: torch.Tensor, max_t: float = 10.0,
                        mapping_fn_inv: Optional[Callable] = None) -> torch.Tensor:
        """时间解码：从脉冲延迟重建输入强度。

        针对 TTFS 编码（每特征至多一个脉冲）。
        若无脉冲，默认将延迟视为 max_t。

        参数:
            spikes (torch.Tensor): 脉冲序列，形状 ``(T, *features)``。
            max_t (float): 编码时的最大延迟，用于逆映射与默认值。
            mapping_fn_inv (Callable, 可选): 自定义延迟->强度逆映射。
                若未提供，使用默认：``x = 1 - delay / max_t``。

        返回:
            torch.Tensor: 重建的输入强度，形状 ``(*features)``。
        """
        # 检测是否存在脉冲
        has_spike = spikes.sum(dim=0) > 0  # (*features)
        delay = spikes.argmax(dim=0)       # 首次发放时间步（全零时返回0）
        # 无脉冲元素设置为 max_t
        delay = torch.where(has_spike, delay,
                            torch.full_like(delay, max_t, dtype=delay.dtype))

        if mapping_fn_inv is not None:
            x_hat = mapping_fn_inv(delay.float())
        else:
            x_hat = 1.0 - delay.float() / max_t
        return x_hat

    @staticmethod
    def decode_phase(spikes: torch.Tensor, T_osc: int,
                     phi: float = 0.0, scale: float = 2 * math.pi) -> torch.Tensor:
        """相位解码：从发放相位重建输入强度。

        通过环形平均估计脉冲相位的中心，反推输入。
        对应编码公式：window_center = (phi + x * scale) % 2π。

        参数:
            spikes (torch.Tensor): 脉冲序列，形状 ``(T, *features)``。
            T_osc (int): 振荡周期，需与编码时一致。
            phi (float): 编码时使用的固定相位偏移。
            scale (float): 编码时使用的相位缩放因子。

        返回:
            torch.Tensor: 重建的输入强度，形状 ``(*features)``。
        """
        T = spikes.shape[0]
        device = spikes.device
        input_shape = spikes.shape[1:]

        # 各时间步的振荡相位
        t = torch.arange(T, device=device, dtype=spikes.dtype)
        phase = (2 * math.pi * (t % T_osc) / T_osc) % (2 * math.pi)  # (T,)

        # 环形平均：以脉冲为权重计算平均相位矢量
        phase_v = phase.view(T, *([1] * len(input_shape)))  # (T, 1, 1, ...)
        cos_sum = (spikes * torch.cos(phase_v)).sum(dim=0)  # (*input_shape)
        sin_sum = (spikes * torch.sin(phase_v)).sum(dim=0)
        mean_phase = torch.atan2(sin_sum, cos_sum) % (2 * math.pi)

        x_hat = (mean_phase - phi) / scale

        # 处理无脉冲特征：总发放数为 0 的位置返回 NaN，表示无法解码
        spike_count = spikes.sum(dim=0)
        no_spike = spike_count == 0
        if no_spike.any():
            x_hat = x_hat.masked_fill(no_spike, float('nan'))
        return x_hat