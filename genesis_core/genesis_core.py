# genesis_core.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Type
import math

# ============================================================================
# 通用常量定义
# ============================================================================

# 调制原常量（中文名称，符合术语中立宪章）
MODULATOR_REWARD = "奖赏信号"
MODULATOR_SATIETY = "满足度因子"
MODULATOR_NOVELTY = "新奇响应子"
MODULATOR_STRESS = "应激激活子"
MODULATOR_FATIGUE = "疲劳累积子"

# 全局状态键常量（用于全局状态监控与调制原系统）
GLOBAL_REWARD_ERROR = "reward_error"
GLOBAL_SATIETY = "satiety"
GLOBAL_NOVELTY = "novelty"
GLOBAL_STRESS = "stress"
GLOBAL_ENERGY_DEFICIT = "energy_deficit"

# 精度映射字典（用于混合精度配置，对应白皮书第九章）
_PRECISION_MAP: Dict[str, torch.dtype] = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


# ============================================================================
# 形态神经元基类 (MorphoNeuron)
# ============================================================================

class MorphoNeuron(nn.Module):
    """形态神经元基类。

    实现多室模型：树突棘、树突干、胞体、轴突。
    内部状态三元组 (u, h, r) 共同构成完备的动力学内核。
    对应白皮书第一章及核心公理。
    """

    def __init__(self,
                 num_neurons: int = 1,
                 num_dendrites: int = 8,
                 tau_u: float = 10.0,
                 rest_h: float = 0.7,
                 threshold: float = 1.0,
                 refractory_period: int = 2,
                 axon_delay: int = 1,
                 bias_dim: int = 1,
                 noise_scale: float = 0.02):
        """
        参数:
            num_neurons: 神经元数量。
            num_dendrites: 每个神经元的树突棘数量。
            tau_u: 膜电位时间常数（响应时间常数）。
            rest_h: 健康度自然回归目标（基准健康度）。
            threshold: 胞体发放阈值。
            refractory_period: 不应期步数。
            axon_delay: 轴突传导延迟步数。
            bias_dim: 共鸣偏好向量的维度，决定节律空间的维数。
            noise_scale: 自持活动噪声强度。
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.num_dendrites = num_dendrites
        self.axon_delay = axon_delay
        self.refractory_period = refractory_period
        self.r_dim = bias_dim
        self.noise_scale = noise_scale

        # 可学习性格参数（对应白皮书1.6节）
        self.tau_u = nn.Parameter(torch.full((num_neurons,), tau_u))
        self.rest_h = nn.Parameter(torch.full((num_neurons,), rest_h))
        self.threshold = nn.Parameter(torch.full((num_neurons,), threshold))
        self.bias = nn.Parameter(torch.randn(num_neurons, bias_dim) * 0.1)  # 共鸣偏好向量

        # 简化参数自适应网络维度
        self.gene_dim = 2

        # 状态缓冲区（将采用 register_buffer 管理，便于迁移与持久化）
        # 占位，实际形状由 reset_state 根据批次大小确定
        self.register_buffer('_u', torch.empty(0))
        self.register_buffer('_h', torch.empty(0))
        self.register_buffer('_r', torch.empty(0))
        self.register_buffer('_refractory_counter', torch.empty(0))
        self.register_buffer('_firing_rate_avg', torch.empty(0))
        self.register_buffer('_gene_state', torch.empty(0))
        self.register_buffer('_spine_ca', torch.empty(0))
        self.register_buffer('_spine_rho', torch.empty(0))
        self.register_buffer('_spine_g', torch.empty(0))
        self.register_buffer('_axon_queue', torch.empty(0))
        self.register_buffer('_axon_write_idx', torch.tensor(0, dtype=torch.long))
        self._current_t: int = 0

        # 胶质细胞注册列表
        self.astrocyte_ids: List[int] = []

        # 内部调制原浓度字典（外部通过 apply_modulator 设置）
        self._modulator_concentrations: Dict[str, float] = {}

        # 演化历史哈希器（运行时注入，不可序列化）
        self._evolution_hasher = None

        # 创建默认单样本状态，方便零输入自持活动演示
        self.reset_state(1)

    def reset_state(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        """重置全部内部状态到初始值，并适配给定的批次大小。

        参数:
            batch_size: 仿真批次大小（样本数）。
            device: 张量设备，若不提供则沿用参数设备。
        """
        if device is None:
            device = self.tau_u.device
        N = self.num_neurons
        D = self.num_dendrites

        self.register_buffer('_u', torch.zeros(batch_size, N, device=device))
        self.register_buffer('_h', self.rest_h.data.unsqueeze(0).expand(batch_size, -1).clone())
        # 谐振敏感度初始化为偏好向量方向（对应公理二）
        self.register_buffer('_r', self.bias.unsqueeze(0).expand(batch_size, -1, -1).clone())
        self.register_buffer('_refractory_counter', torch.zeros(batch_size, N, device=device))
        self.register_buffer('_firing_rate_avg', torch.zeros(batch_size, N, device=device))
        self.register_buffer('_gene_state', torch.zeros(batch_size, N, self.gene_dim, device=device))

        # 树突棘状态：钙浓度、第二类输入通道/第一类输入通道比值、传导权重（对应白皮书1.1节）
        self.register_buffer('_spine_ca', torch.zeros(batch_size, D, N, device=device))
        self.register_buffer('_spine_rho', torch.ones(batch_size, D, N, device=device) * 0.5)
        self.register_buffer('_spine_g', torch.ones(D, N, device=device) / D)  # 等权重传导

        # 轴突延迟队列（环形缓冲，对应白皮书1.4节）
        queue_size = self.axon_delay + 1
        self.register_buffer('_axon_queue', torch.zeros(queue_size, batch_size, N, device=device))
        self.register_buffer('_axon_write_idx', torch.tensor(0, device=device, dtype=torch.long))

        self._current_t = 0

        # 清空调制原浓度，避免跨批次状态残留
        self._modulator_concentrations.clear()

    def forward(self,
                x: Optional[torch.Tensor] = None,
                modulator_concentrations: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """执行一个时间步的完整多室更新，并返回轴突输出脉冲。

        参数:
            x: 每个树突棘的输入信号，形状 (batch, num_dendrites, num_neurons)。
               若为 None，表示零外部输入（用于验证公理四：存在先于任务）。
            modulator_concentrations: 可选的调制原浓度覆写字典。
        返回:
            out_spike: 经过轴突传导延迟后的输出脉冲，形状 (batch, num_neurons)。
        """
        # ----- 批次与设备管理 -----
        if x is not None:
            bsz, dev = x.shape[0], x.device
        else:
            if self._u.numel() == 0:
                raise RuntimeError("状态未初始化，请先调用 reset_state 或提供有效输入。")
            bsz, dev = self._u.shape[0], self._u.device

        if self._u.shape[0] != bsz:
            self.reset_state(bsz, dev)

        # ----- 调制原浓度合并（外部传入优先，否则使用内部存储） -----
        eff_modulators = {**self._modulator_concentrations,
                          **(modulator_concentrations or {})}

        # ----- 1. 树突棘与树突干处理（对应白皮书1.1、1.2节） -----
        v_dend = self._update_dendrites(x)

        # ----- 2. 胞体整合与发放（对应白皮书1.3节，包含不应期、内在电流） -----
        spike = self._update_soma(v_dend, eff_modulators)

        # ----- 3. 轴突传导延迟（对应白皮书1.4节） -----
        out_spike = self._update_axon(spike)

        # ----- 4. 内禀状态变量更新（健康度、敏感度、参数自适应网络） -----
        self._update_health(spike)
        self._update_resonance()
        self._update_gene(spike)

        # ----- 5. 树突棘动态重塑（修复三） -----
        self.update_dendritic_spines()

        self._current_t += 1

        return out_spike

    # ------------------------------------------------------------------------
    # 子更新函数
    # ------------------------------------------------------------------------

    def _update_dendrites(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        """树突棘非线性计算、树突干被动传导与主动增强。"""
        if x is None:
            x = torch.zeros(self._u.shape[0], self.num_dendrites, self.num_neurons,
                            device=self._u.device)
        # 树突棘局部非线性计算（独立处理每个输入通道）
        # 局部电位由输入信号、第二类输入通道/第一类输入通道比值、局部钙浓度共同决定
        v_spine = torch.tanh(x * (1.0 + self._spine_rho)) * torch.sigmoid(self._spine_ca)
        # 树突棘状态更新
        ca_decay, ca_influx = 0.05, 0.1
        self._spine_ca = self._spine_ca - ca_decay * self._spine_ca + ca_influx * torch.abs(x)
        # rho 缓慢向钙依赖的目标值移动
        target_rho = torch.sigmoid(self._spine_ca)
        self._spine_rho = self._spine_rho + 0.01 * (target_rho - self._spine_rho)

        # 树突干被动传导
        v_dend = torch.sum(self._spine_g.unsqueeze(0) * v_spine, dim=1)
        # 主动增强：强同步输入时放大
        dend_boost_threshold = 0.5
        boost = (v_dend.abs() > dend_boost_threshold).float()
        v_dend = v_dend * (1.0 + boost * 0.5)
        return v_dend

    def _update_soma(self, v_dend: torch.Tensor,
                     modulators: Dict[str, float]) -> torch.Tensor:
        """胞体膜电位更新、阈值发放、不应期处理（内在性优先）。"""
        # 不应期计数器衰减
        self._refractory_counter = torch.clamp(self._refractory_counter - 1, min=0)
        in_ref = (self._refractory_counter > 0).float()
        not_ref = 1.0 - in_ref

        # 内在电流（对应公理一、四）
        # 健康度回归驱动：当 h 低于目标时去极化，促使发放以恢复稳态
        I_homeo = -0.1 * (self._h - self.rest_h.unsqueeze(0))
        # 自发活动噪声（保障零输入下不自闭）
        I_noise = self.noise_scale * torch.randn_like(self._u) * (1.0 + self._r.norm(dim=-1))
        I_int = I_homeo + I_noise

        # 全局调制原对内状态的作用（公理五：价值内生）
        if MODULATOR_STRESS in modulators:
            I_int = I_int + 0.1 * modulators[MODULATOR_STRESS]
        if MODULATOR_FATIGUE in modulators:
            I_int = I_int - 0.05 * modulators[MODULATOR_FATIGUE]
        # 局部胶质反馈调制原（由星形胶质细胞通过 apply_modulator 注入）
        if "local_glial_bias" in modulators:
            I_int = I_int + 0.1 * modulators["local_glial_bias"]

        # 膜电位更新（参数自适应网络调整有效时间常数）
        tau_eff = self.tau_u.unsqueeze(0) + 0.1 * self._gene_state[..., 0]
        tau_eff = torch.clamp(tau_eff, min=1.0)
        du = (-self._u + v_dend + I_int) / tau_eff
        self._u = not_ref * (self._u + du) + in_ref * (self._u * 0.9)  # 不应期缓慢衰减

        # 发放阈值（参数自适应网络调整）
        thresh_eff = self.threshold.unsqueeze(0) + 0.1 * self._gene_state[..., 1]
        spike = not_ref * (self._u >= thresh_eff).float()

        # 发放后重置膜电位并进入不应期
        u_reset = 0.0
        self._u = spike * u_reset + (1.0 - spike) * self._u
        self._refractory_counter = torch.where(
            spike > 0,
            torch.full_like(self._refractory_counter, self.refractory_period),
            self._refractory_counter
        )

        # 注入演化历史哈希更新（在膜电位等状态变更后）
        if self._evolution_hasher is not None:
            bytes_of_state = self._u.clone().detach().cpu().numpy().tobytes()
            self._evolution_hasher.update(bytes_of_state)

        return spike

    def _update_axon(self, spike: torch.Tensor) -> torch.Tensor:
        """轴突传导延迟逻辑（环形队列）。"""
        if self.axon_delay == 0:
            return spike  # 无延迟直接传出

        write_idx = self._axon_write_idx.item()
        self._axon_queue[write_idx] = spike
        # 读取位置：下一槽即为最老脉冲，实现 FIFO 延迟
        read_idx = (write_idx + 1) % (self.axon_delay + 1)
        delayed_spike = self._axon_queue[read_idx]
        self._axon_write_idx = torch.tensor(
            (write_idx + 1) % (self.axon_delay + 1),
            device=self._axon_write_idx.device,
            dtype=torch.long
        )
        # 传导可靠性（概率性丢失）
        reliability = 0.98
        dropout_mask = torch.rand_like(delayed_spike) < reliability
        return delayed_spike * dropout_mask.float()

    def _update_health(self, spike: torch.Tensor) -> None:
        """健康度 h 的回归机制（公理一强制要求）。"""
        alpha_f = 0.1
        self._firing_rate_avg = (1.0 - alpha_f) * self._firing_rate_avg + alpha_f * spike

        optimal_rate = 0.1
        tau_h = 50.0
        # 活动偏离最佳值时消耗健康度
        penalty = 0.1 * torch.abs(self._firing_rate_avg - optimal_rate)
        dh = (self.rest_h.unsqueeze(0) - self._h) / tau_h - penalty
        self._h = self._h + dh
        self._h.clamp_(0.0, 1.0)

        # 注入演化历史哈希更新（在健康度等内部状态变更后）
        if self._evolution_hasher is not None:
            bytes_of_state = self._u.clone().detach().cpu().numpy().tobytes()
            self._evolution_hasher.update(bytes_of_state)

    def _update_resonance(self) -> None:
        """谐振敏感度 r 的更新（方向由 bias 决定，幅度受健康度调制）。"""
        base_dir = self.bias.unsqueeze(0)  # (1, N, r_dim)
        # 幅度：健康度 sigmoid 变换 + 活动影响
        mag = torch.sigmoid((self._h - 0.5) * 5.0)
        mag = mag * (1.0 + 0.02 * self._firing_rate_avg)
        self._r = base_dir * mag.unsqueeze(-1)

        # 注入演化历史哈希更新（在谐振敏感度变更后）
        if self._evolution_hasher is not None:
            bytes_of_state = self._u.clone().detach().cpu().numpy().tobytes()
            self._evolution_hasher.update(bytes_of_state)

    def _update_gene(self, spike: torch.Tensor) -> None:
        """简化参数自适应网络：根据长期活动缓慢调整参数（对应白皮书1.3节）。"""
        decay = 0.001
        target = self._firing_rate_avg.unsqueeze(-1) - 0.1  # 偏差向量
        target = torch.cat([target, target], dim=-1)  # 两个参数自适应维度
        self._gene_state = (1.0 - decay) * self._gene_state + decay * target

    # ------------------------------------------------------------------------
    # 树突棘动态重塑（修复三）
    # ------------------------------------------------------------------------
    def update_dendritic_spines(self):
        """根据树突棘钙浓度平均值动态调整树突棘数量（接口级实现）。

        当整个神经元群的平均钙浓度高于 0.8 时增加一个树突棘，
        低于 0.2 时减少一个树突棘（最少保持 1 个）。
        当前版本通过全局 num_dendrites 调整并重塑缓冲区。
        """
        if self._spine_ca.numel() == 0:
            return
        # 计算所有树突棘钙浓度的平均值（标量）
        avg_ca = self._spine_ca.mean()
        if avg_ca > 0.8 and self.num_dendrites < 16:
            self._add_dendrite()
        elif avg_ca < 0.2 and self.num_dendrites > 1:
            self._remove_dendrite()

    def _add_dendrite(self):
        """增加一个树突棘并扩展所有相关缓冲区。"""
        D = self.num_dendrites
        B, _, N = self._spine_ca.shape
        device = self._spine_ca.device
        dtype = self._spine_ca.dtype

        # 扩展 _spine_ca (batch, D, N) -> (batch, D+1, N)
        new_ca = torch.zeros(B, 1, N, device=device, dtype=dtype)
        self.register_buffer('_spine_ca', torch.cat([self._spine_ca, new_ca], dim=1))
        # 扩展 _spine_rho
        new_rho = torch.full((B, 1, N), 0.5, device=device, dtype=dtype)
        self.register_buffer('_spine_rho', torch.cat([self._spine_rho, new_rho], dim=1))
        # 扩展 _spine_g (D, N) -> (D+1, N)，新权重初始为 0
        new_g = torch.zeros(1, N, device=device, dtype=dtype)
        self.register_buffer('_spine_g', torch.cat([self._spine_g, new_g], dim=0))

        self.num_dendrites += 1

    def _remove_dendrite(self):
        """减少一个树突棘并截断所有相关缓冲区。"""
        if self.num_dendrites <= 1:
            return
        # 截取前 num_dendrites-1 个树突棘
        new_D = self.num_dendrites - 1
        self.register_buffer('_spine_ca', self._spine_ca[:, :new_D, :].clone())
        self.register_buffer('_spine_rho', self._spine_rho[:, :new_D, :].clone())
        self.register_buffer('_spine_g', self._spine_g[:new_D, :].clone())
        self.num_dendrites = new_D

    # ------------------------------------------------------------------------
    # 接口钩子
    # ------------------------------------------------------------------------

    def get_internal_state(self) -> Dict[str, torch.Tensor]:
        """返回神经元当前内部状态的完整快照（用于监视、保存、调试）。"""
        return {
            'u': self._u.clone(),
            'h': self._h.clone(),
            'r': self._r.clone(),
            'refractory_counter': self._refractory_counter.clone(),
            'firing_rate_avg': self._firing_rate_avg.clone(),
            'gene_state': self._gene_state.clone(),
            'spine_ca': self._spine_ca.clone(),
            'spine_rho': self._spine_rho.clone(),
            'spine_g': self._spine_g.clone(),
            'axon_queue': self._axon_queue.clone(),
            'axon_write_idx': self._axon_write_idx.clone(),
            'noise_scale': torch.tensor(self.noise_scale, device=self._u.device),
        }

    def apply_modulator(self, name: str, concentration: float) -> None:
        """接收全局调制原信号，存储浓度供动力学方程使用（公理五）。

        注意：调制原不直接控制行为，而是通过影响内部状态变量(u,h,r)间接作用。
        """
        # 仅记录浓度，实际效应在 forward 中施加，保证一致性与时间精度
        self._modulator_concentrations[name] = concentration

    def register_astrocyte(self, astrocyte_id: int) -> None:
        """注册一个与之关联的星形胶质细胞ID。"""
        if astrocyte_id not in self.astrocyte_ids:
            self.astrocyte_ids.append(astrocyte_id)

    def inject_evolution_hasher(self, hasher) -> None:
        """注入演化历史哈希器，用于实时累积网络状态变化。

        参数:
            hasher: 一个包含 update 和 get_hash 方法的哈希器实例。
        """
        self._evolution_hasher = hasher


# ============================================================================
# 共鸣连接点基类 (ResonanceSynapse)
# ============================================================================

class ResonanceSynapse(nn.Module):
    """共鸣连接点基类。

    实现囊泡池三池模型、动态阻抗、双向可塑性、连接点标记与捕获。
    对应白皮书第二章及公理二、三。
        .. note::
        基类内建的可塑性逻辑（包括STDP）是 **简化默认行为**，主要用于原型验证和继承扩展。
        对于需要完整功能的应用，建议使用 :class:`synapse.STDPSynapse`。
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
        """
        参数:
            num_synapses: 连接点数量。
            pool_total: 囊泡总量（归一化）。
            k_cyc: 回收池向预备池转换速率。
            k_rp: 预备池向可释放池填充速率。
            p_base: 基础释放概率。
            ca_decay: 残余钙衰减系数。
            facil_decay: 易化迹衰减系数。
            facil_factor: 易化对释放概率的贡献因子。
            memory_size: 交往记忆队列容量。
            A_plus: STDP增强系数。
            A_minus: STDP减弱系数。
            trace_decay: STDP迹衰减系数。
        """
        super().__init__()
        self.num_synapses = num_synapses
        self.pool_total = pool_total
        self.k_cyc = k_cyc
        self.k_rp = k_rp
        self.p_base = p_base
        self.ca_decay = ca_decay
        self.facil_decay = facil_decay
        self.facil_factor = facil_factor
        self.memory_size = memory_size
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.trace_decay = trace_decay

        # 内部状态缓冲区
        self.register_buffer('pool_RP', torch.empty(0))
        self.register_buffer('pool_RRP', torch.empty(0))
        self.register_buffer('pool_Rcyc', torch.empty(0))
        self.register_buffer('resid_Ca', torch.empty(0))
        self.register_buffer('facil_trace', torch.empty(0))
        self.register_buffer('pre_trace', torch.empty(0))
        self.register_buffer('post_trace', torch.empty(0))
        self.register_buffer('strength', torch.empty(0))
        self.register_buffer('tag', torch.empty(0))
        self.register_buffer('retrograde_signal', torch.empty(0))
        self.register_buffer('prune_flag', torch.empty(0, dtype=torch.bool))
        self.register_buffer('protected', torch.empty(0, dtype=torch.bool))
        self.register_buffer('memory_buffer', torch.empty(0))
        self.register_buffer('memory_ptr', torch.empty(0, dtype=torch.long))
        self._current_t: int = 0

        # 调制原内部存储
        self._modulator_concentrations: Dict[str, float] = {}

        # 演化历史哈希器（运行时注入，不可序列化）
        self._evolution_hasher = None

        # 创建默认单样本状态
        self.reset_state(1)

    def reset_state(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        """重置全部内部状态到初始值。"""
        if device is None:
            if self.strength.numel() > 0:
                device = self.strength.device
            else:
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = torch.device('cpu')
        N = self.num_synapses

        self.register_buffer('pool_RP', torch.full((batch_size, N), self.pool_total * 0.2, device=device))
        self.register_buffer('pool_RRP', torch.full((batch_size, N), self.pool_total * 0.1, device=device))
        self.register_buffer('pool_Rcyc', torch.full((batch_size, N), self.pool_total * 0.7, device=device))
        self.register_buffer('resid_Ca', torch.zeros(batch_size, N, device=device))
        self.register_buffer('facil_trace', torch.zeros(batch_size, N, device=device))
        self.register_buffer('pre_trace', torch.zeros(batch_size, N, device=device))
        self.register_buffer('post_trace', torch.zeros(batch_size, N, device=device))
        self.register_buffer('strength', torch.ones(batch_size, N, device=device) * 0.5)
        self.register_buffer('tag', torch.zeros(batch_size, N, device=device))
        self.register_buffer('retrograde_signal', torch.zeros(batch_size, N, device=device))
        self.register_buffer('prune_flag', torch.zeros(batch_size, N, device=device, dtype=torch.bool))
        self.register_buffer('protected', torch.zeros(batch_size, N, device=device, dtype=torch.bool))
        self.register_buffer('memory_buffer', torch.zeros(batch_size, N, self.memory_size, 2, device=device))
        self.register_buffer('memory_ptr', torch.zeros(batch_size, N, device=device, dtype=torch.long))
        self._current_t = 0

        # 清空调制原浓度，避免跨批次状态残留
        self._modulator_concentrations.clear()

    def forward(self,
                pre_spike: torch.Tensor,
                post_spike: torch.Tensor,
                pre_r: torch.Tensor,
                post_r: torch.Tensor,
                modulator_concentrations: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """执行一个时间步的连接点信息处理，返回连接点后电流。

        参数:
            pre_spike: 连接点前脉冲，形状 (batch, num_synapses)
            post_spike: 连接点后脉冲，形状 (batch, num_synapses)
            pre_r: 连接点前谐振敏感度向量，形状 (batch, num_synapses, r_dim)
            post_r: 连接点后谐振敏感度向量，形状 (batch, num_synapses, r_dim)
            modulator_concentrations: 调制原浓度覆写。
        返回:
            post_current: 连接点后电流，形状 (batch, num_synapses)
        """
        bsz = pre_spike.shape[0]
        if self.strength.shape[0] != bsz:
            self.reset_state(bsz, pre_spike.device)

        eff_modulators = {**self._modulator_concentrations,
                          **(modulator_concentrations or {})}

        # 1. 囊泡池恢复（对应白皮书2.1节三池模型）
        d_Rcyc2RP = self.k_cyc * self.pool_Rcyc
        d_RP2RRP = self.k_rp * self.pool_RP
        self.pool_Rcyc = self.pool_Rcyc - d_Rcyc2RP
        self.pool_RP = self.pool_RP + d_Rcyc2RP - d_RP2RRP
        self.pool_RRP = self.pool_RRP + d_RP2RRP
        self.pool_Rcyc.clamp_(min=0.0, max=self.pool_total)
        self.pool_RP.clamp_(min=0.0, max=self.pool_total)
        self.pool_RRP.clamp_(min=0.0, max=self.pool_total)

        # 2. 释放概率动态计算（残余钙+易化+全局调制原，对应白皮书2.2节）
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
        # 局部胶质反馈调制原（由星形胶质细胞通过 apply_modulator 注入）
        if "local_glial_bias" in eff_modulators:
            mod_effect += 0.2 * eff_modulators["local_glial_bias"]

        p_release = self.p_base * p_ca * p_fac * mod_effect
        # 逆行信使调节释放概率（对应白皮书2.5节逆行）
        p_release = p_release * (1.0 - 0.1 * self.retrograde_signal)
        p_release = torch.clamp(p_release, 0.0, 1.0)

        # 3. 释放过程（消耗可释放池，增加回收池）
        release = pre_spike * p_release * self.pool_RRP
        self.pool_RRP = self.pool_RRP - release
        self.pool_Rcyc = self.pool_Rcyc + release
        self.pool_RRP.clamp_(min=0.0)

        # 4. 动态阻抗计算（|r_pre - r_post| 的一般化距离，对应公理二；文档声明可扩展）
        # 默认采用 L2 距离，可替换为任意 dist(r_pre, r_post) 度量
        impedance = torch.norm(pre_r - post_r, dim=-1)  # (batch, N)
        # 传递效率：失谐越小，效率越高
        efficiency = 1.0 / (1.0 + impedance)
        post_current = release * efficiency * self.strength

        # 5. 残余钙更新
        ca_influx = 0.1 * pre_spike
        self.resid_Ca = self.ca_decay * self.resid_Ca + ca_influx

        # 6. 易化迹更新
        self.facil_trace = self.facil_decay * self.facil_trace + (1.0 - self.facil_decay) * pre_spike

        # 7. 双向可塑性 - 顺向 STDP（对应白皮书2.5节）
        self.pre_trace *= self.trace_decay
        self.post_trace *= self.trace_decay
        ltp = self.A_plus * pre_spike * self.post_trace
        ltd = self.A_minus * post_spike * self.pre_trace
        self.strength = self.strength + ltp - ltd
        self.strength.clamp_(0.0, 1.0)
        # 增加当前脉冲到迹
        self.pre_trace = self.pre_trace + pre_spike
        self.post_trace = self.post_trace + post_spike

        # 8. 逆行信使更新
        self.retrograde_signal = 0.9 * self.retrograde_signal + 0.1 * post_spike

        # 9. 连接点标记与捕获（对应白皮书2.6节）
        strong_event = pre_spike * (impedance < 0.1).float()
        self.tag = self.tag + strong_event * 0.1
        self.tag = self.tag * 0.95  # 衰减
        if MODULATOR_REWARD in eff_modulators:
            reward = eff_modulators[MODULATOR_REWARD]
            # 奖赏信号捕获标记，促成持久结构改变
            strength_boost = self.tag * reward * 0.01
            self.strength = self.strength + strength_boost
            self.tag = self.tag - strength_boost * 10  # 消耗标记

        # 10. 交往记忆队列更新（对应白皮书2.4节）
        if strong_event.any():
            timestamp = self._current_t
            batch_idx, syn_idx = torch.where(strong_event)
            ptr = self.memory_ptr[batch_idx, syn_idx]
            self.memory_buffer[batch_idx, syn_idx, ptr, 0] = timestamp
            self.memory_buffer[batch_idx, syn_idx, ptr, 1] = self.strength[batch_idx, syn_idx]
            self.memory_ptr[batch_idx, syn_idx] = (ptr + 1) % self.memory_size

        # 注入演化历史哈希更新（在连接点强度等状态变更之后）
        if self._evolution_hasher is not None:
            bytes_of_state = self.strength.clone().detach().cpu().numpy().tobytes()
            self._evolution_hasher.update(bytes_of_state)

        self._current_t += 1
        return post_current

    # ------------------------------------------------------------------------
    # 接口钩子
    # ------------------------------------------------------------------------

    def get_synaptic_state(self) -> Dict[str, torch.Tensor]:
        """返回连接点当前内部状态的完整快照。"""
        return {
            'pool_RP': self.pool_RP.clone(),
            'pool_RRP': self.pool_RRP.clone(),
            'pool_Rcyc': self.pool_Rcyc.clone(),
            'resid_Ca': self.resid_Ca.clone(),
            'facil_trace': self.facil_trace.clone(),
            'pre_trace': self.pre_trace.clone(),
            'post_trace': self.post_trace.clone(),
            'strength': self.strength.clone(),
            'tag': self.tag.clone(),
            'retrograde_signal': self.retrograde_signal.clone(),
            'prune_flag': self.prune_flag.clone(),
            'protected': self.protected.clone(),
            'memory_buffer': self.memory_buffer.clone(),
            'memory_ptr': self.memory_ptr.clone(),
        }

    def apply_modulator(self, name: str, concentration: float) -> None:
        """接收全局调制原信号，存储浓度供动力学方程使用（公理五）。"""
        self._modulator_concentrations[name] = concentration

    def mark_for_pruning(self) -> None:
        """标记该连接点为可修剪状态（由小胶质细胞后续处理）。"""
        self.prune_flag = torch.ones_like(self.prune_flag)

    def protect_from_pruning(self) -> None:
        """保护该连接点免于修剪。"""
        self.prune_flag = torch.zeros_like(self.prune_flag)
        self.protected = torch.ones_like(self.protected)

    def inject_evolution_hasher(self, hasher) -> None:
        """注入演化历史哈希器，用于实时累积连接点状态变化。

        参数:
            hasher: 一个包含 update 和 get_hash 方法的哈希器实例。
        """
        self._evolution_hasher = hasher


# ============================================================================
# 预置神经元类型映射（延迟导入，避免循环依赖）
# ============================================================================

_NEURON_TYPE_MAP = None


def get_neuron_type_map() -> Dict[str, Type[MorphoNeuron]]:
    """返回 NEURON_TYPE_MAP，通过延迟导入避免循环依赖。

    首次调用时从 .neuron 模块导入所需类，并缓存结果。
    后续调用直接返回缓存，确保在模块完全加载后执行导入。

    注意：如果 utils.py 和 life.py 原来直接导入 NEURON_TYPE_MAP，
    需要改为调用 genesis_core.get_neuron_type_map() 以获取该映射表。
    """
    global _NEURON_TYPE_MAP
    if _NEURON_TYPE_MAP is None:
        from .neuron import (
            IFNeuron, LIFNeuron, PLIFNeuron, QIFNeuron, EIFNeuron, IzhikevichNeuron,
        )
        _NEURON_TYPE_MAP = {
            # 简写键（用于 create_layer 等便捷接口）
            'IF': IFNeuron,
            'LIF': LIFNeuron,
            'PLIF': PLIFNeuron,
            'QIF': QIFNeuron,
            'EIF': EIFNeuron,
            'Izhikevich': IzhikevichNeuron,
            'MorphoNeuron': MorphoNeuron,
            # 类名键（用于 .life 文件加载时通过类名查找）
            'IFNeuron': IFNeuron,
            'LIFNeuron': LIFNeuron,
            'PLIFNeuron': PLIFNeuron,
            'QIFNeuron': QIFNeuron,
            'EIFNeuron': EIFNeuron,
            'IzhikevichNeuron': IzhikevichNeuron,
        }
    return _NEURON_TYPE_MAP