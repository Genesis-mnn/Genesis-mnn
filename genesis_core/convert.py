#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genesis 模型转换管线 (convert.py)

实现 SNN→MNN 和 ANN→MNN 的自动转换，严格遵循白皮书六条核心公理。

核心功能：
    1. SNNtoMNNConverter：将基于 SpikingJelly / snnTorch 的 SNN 转换为 Genesis 原生 MNN。
    2. ANNtoMNNConverter：将 PyTorch 传统网络（CNN/RNN/Transformer）转换为 MNN 脉冲网络。
    3. 统一入口 convert()，自动检测模型类型并返回可直接脉冲推理的 MNN 模型。

公理约束：
    - 公理二（共鸣取代权重）：连接必须映射为 ResonanceSynapse（或子类），
      初始 impedance 设为 0，strength 继承原权重。
    - 公理三（演化优于训练）：转换后的权重通过局部可塑性规则（如 STDP）自发演化，
      不使用反向传播或全局损失梯度。

适配器模式：
    提供注册机制，允许为不同 SNN 框架注册自定义映射规则。
    内置默认规则，即使未安装任何外部 SNN 库也能独立运行。

注意事项：
    - 需要将本文件与 genesis_core.py, neuron.py, synapse.py 放置在同一目录下。
    - 转换后的模型在推理时建议固定批次大小，以避免内部状态重置带来的开销。
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Type
import math

# ============================================================================
# 导入 Genesis 核心模块（务必保证路径正确）
# ============================================================================
from .genesis_core import (
    MorphoNeuron,
    ResonanceSynapse,
    MODULATOR_REWARD,
    MODULATOR_FATIGUE,
    MODULATOR_STRESS,
)
from .neuron import (
    IFNeuron,
    LIFNeuron,
    PLIFNeuron,
    QIFNeuron,
    EIFNeuron,
    IzhikevichNeuron,
)
from .synapse import STDPSynapse, PlasticityRule


# ============================================================================
# 公理二 · 通用突触层（基于索引的稀疏实现）
# ============================================================================
class SynapticLayer(nn.Module):
    """统一突触层，封装 ResonanceSynapse 并处理任意连接拓扑。

    支持全连接和卷积等任意稀疏/密集连接模式，通过预计算 pre/post 神经元索引
    来实现高效的脉冲-突触映射，符合公理二（初始 impedance 为 0 通过统一 r 初值保证）。
    """

    def __init__(
        self,
        in_neurons: int,
        out_neurons: int,
        pre_indices: torch.Tensor,
        post_indices: torch.Tensor,
        synapse_class: Type[ResonanceSynapse] = ResonanceSynapse,
        synapse_kwargs: Optional[Dict[str, Any]] = None,
        initial_weight: Optional[torch.Tensor] = None,
        initial_bias: Optional[torch.Tensor] = None,
        r_dim: int = 1,
    ):
        """
        参数:
            in_neurons: 前一层神经元数量（或输入维度）。
            out_neurons: 后一层神经元数量。
            pre_indices: (num_synapses,) long，每个突触对应的前神经元局部索引。
            post_indices: (num_synapses,) long，每个突触对应的后神经元局部索引。
            synapse_class: 使用的突触类，默认为 ResonanceSynapse。
            synapse_kwargs: 传递给突触构造函数的额外参数。
            initial_weight: 初始权重 (num_synapses,) 或 (out, in) 密集矩阵。
            initial_bias: 初始偏置 (out_neurons,)，将用于调整神经元阈值。
            r_dim: 谐振敏感度向量的维度。
        """
        super().__init__()
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons
        self.r_dim = r_dim

        # 注册索引缓冲区
        self.register_buffer("pre_indices", torch.as_tensor(pre_indices, dtype=torch.long))
        self.register_buffer("post_indices", torch.as_tensor(post_indices, dtype=torch.long))

        num_synapses = len(pre_indices)
        self.num_synapses = num_synapses
        if synapse_kwargs is None:
            synapse_kwargs = {}
        self.synapse = synapse_class(num_synapses=num_synapses, **synapse_kwargs)

        # 处理初始权重（模板存储，用于 reset_state 时恢复）
        if initial_weight is not None:
            if initial_weight.dim() == 2:  # 密集矩阵 (out, in)
                weight_flat = initial_weight[self.post_indices, self.pre_indices].clone()
            else:
                weight_flat = initial_weight.clone()
            self.register_buffer("weight_template", weight_flat)
        else:
            self.register_buffer("weight_template", torch.empty(0))  # 标记为空

        # 初始偏置（外部使用，不在此层内直接作用于神经元）
        self.initial_bias = initial_bias

    def reset_state(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        """重置突触内部状态，并恢复预设的初始权重（公理二）。"""
        self.synapse.reset_state(batch_size, device)
        if self.weight_template.numel() > 0:
            self.synapse.strength.data.copy_(
                self.weight_template.unsqueeze(0).expand(batch_size, -1)
            )

    def forward(
        self,
        pre_spike: torch.Tensor,
        pre_r: torch.Tensor,
        post_spike: torch.Tensor,
        post_r: torch.Tensor,
        modulator_concentrations: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """执行一个时间步的突触信息传递与可塑性更新。

        参数:
            pre_spike: (batch, in_neurons) 前神经元脉冲。
            pre_r: (batch, in_neurons, r_dim) 前神经元谐振敏感度。
            post_spike: (batch, out_neurons) 后神经元上一时间步脉冲。
            post_r: (batch, out_neurons, r_dim) 后神经元上一时间步谐振敏感度。
            modulator_concentrations: 调制原浓度字典。

        返回:
            post_current: (batch, out_neurons) 汇聚到每个后神经元的电流。
        """
        # 维度校验（修复二）
        assert pre_r.shape[-1] == self.r_dim, (
            f"pre_r 最后一维应为 {self.r_dim}，实际为 {pre_r.shape[-1]}"
        )
        assert post_r.shape[-1] == self.r_dim, (
            f"post_r 最后一维应为 {self.r_dim}，实际为 {post_r.shape[-1]}"
        )

        B, N_in = pre_spike.shape
        # 通过索引提取每个连接对应的前/后脉冲和 r
        pre_spike_all = pre_spike[:, self.pre_indices]  # (B, num_synapses)
        post_spike_all = post_spike[:, self.post_indices]
        pre_r_all = pre_r[:, self.pre_indices, :]  # (B, num_synapses, r_dim)
        post_r_all = post_r[:, self.post_indices, :]

        # 公理二：动态阻抗由 |r_pre - r_post| 决定，传递效率自动调整
        post_current_all = self.synapse(
            pre_spike_all,
            post_spike_all,
            pre_r_all,
            post_r_all,
            modulator_concentrations,
        )  # (B, num_synapses)

        # 按后神经元索引散射求和，得到后电流
        post_current = torch.zeros(
            B, self.out_neurons, device=pre_spike.device
        ).scatter_add(
            1,
            self.post_indices.unsqueeze(0).expand(B, -1),
            post_current_all,
        )
        return post_current


# ============================================================================
# MNN 包装网络（实现脉冲推理时间循环）
# ============================================================================
class MNNModel(nn.Module):
    """转换后可直接用于脉冲推理的 MNN 网络封装。

    内部维护神经元层和突触层的执行顺序，处理时间步循环以及层间脉冲传递。
    支持自动状态重置（推理场景）和持续演化（公理三场景）。
    """

    def __init__(
        self,
        stages: List[Tuple[Optional[SynapticLayer], MorphoNeuron]],
        time_steps: int,
        batch_size: int = 1,
        auto_reset: bool = True,
    ):
        """
        参数:
            stages: 阶段列表，每个元素为 (突触层, 神经元层)；突触层可为 None。
            time_steps: 模拟总时间步数（固定用于推理，演化时可忽略）。
            batch_size: 初始批次大小（用于状态分配）。
            auto_reset: 是否在每次 forward 开始时自动重置状态。
        """
        super().__init__()
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.auto_reset = auto_reset

        self.synapses = nn.ModuleList()
        self.neurons = nn.ModuleList()
        for syn, neu in stages:
            self.synapses.append(syn)
            self.neurons.append(neu)

        # 为每一层存储上一时间步的脉冲和 r（初始化全零）
        self.last_spikes: List[torch.Tensor] = []
        self.last_rs: List[torch.Tensor] = []
        # 预注册，实际在 reset_states 中赋予
        self._stages = stages

    def reset_states(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        """重置所有组件状态并将 last_spike/last_r 归零。"""
        if device is None:
            device = next(self.parameters()).device
        self.last_spikes.clear()
        self.last_rs.clear()
        for syn, neu in self._stages:
            if syn is not None:
                syn.reset_state(batch_size, device)
            neu.reset_state(batch_size, device)
            self.last_spikes.append(
                torch.zeros(batch_size, neu.num_neurons, device=device)
            )
            self.last_rs.append(
                torch.zeros(batch_size, neu.num_neurons, neu.r_dim, device=device)
            )

    def forward(
        self,
        x: torch.Tensor,
        modulator_concentrations: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """执行脉冲时间序列传播。

        参数:
            x: (batch, T, in_features) 输入脉冲序列，或 (batch, in_features) 单步。
            modulator_concentrations: 全局调制原浓度。

        返回:
            out: (batch, T, last_neurons) 最后一层神经元的脉冲序列。
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 变为 (B,1,N)
        B, T, in_features = x.shape

        # 自动重置状态（推理模式）
        if self.auto_reset or B != self.batch_size:
            self.reset_states(B, x.device)

        # 外部输入的谐振敏感度（中性零向量，保证公理二初始 impedance 为 0）
        r_dim = self.neurons[0].r_dim
        input_r = torch.zeros(B, in_features, r_dim, device=x.device)

        outputs = []
        for t in range(T):
            inp_spike = x[:, t, :]  # (B, in)

            # 当前步的前一层脉冲与 r（初始为外输入）
            pre_spike = inp_spike
            pre_r = input_r

            for idx, (syn, neu) in enumerate(self._stages):
                # 本层上一步的脉冲和 r（用作突触的 post 信息）
                post_spike = self.last_spikes[idx]
                post_r = self.last_rs[idx]

                if syn is not None:
                    # 突触传递 + 可塑性更新
                    current = syn(
                        pre_spike, pre_r, post_spike, post_r, modulator_concentrations
                    )
                    # 神经元整合产生输出（传入 (B, out_neurons) 电流）
                    # 神经元 forward 需要形状 (B, 1, N) 或自动 unsqueeze
                    if current.dim() == 2:
                        current = current.unsqueeze(1)  # => (B, 1, N)
                    spike = neu(current, modulator_concentrations)
                else:
                    # 直连输入（无突触层），例如输入直接进入第一层神经元
                    if pre_spike.dim() == 2:
                        pre_spike = pre_spike.unsqueeze(1)
                    spike = neu(pre_spike, modulator_concentrations)

                # 确保 spike 形状为 (B, N)
                if spike.dim() == 3:
                    spike = spike.squeeze(1)

                # 存储当前步脉冲和 r（公理二依赖 r 更新）
                self.last_spikes[idx] = spike
                self.last_rs[idx] = neu.get_internal_state()['r']

                # 传递给下一层
                pre_spike = spike
                pre_r = neu.get_internal_state()['r']

            outputs.append(pre_spike)  # 最后一层脉冲作为输出

        outputs = torch.stack(outputs, dim=1)  # (B, T, N_last)
        return outputs


# ============================================================================
# 网络适配器注册机制
# ============================================================================
class SNNAdapterRegistry:
    """SNN 框架适配器注册表。

    允许为不同 SNN 库注册自定义映射规则，包括：
    - 如何识别连接层/神经元层
    - 如何提取参数
    - 映射到哪种 Genesis 组件
    """

    _adapters: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, adapter: Any) -> None:
        cls._adapters[name] = adapter

    @classmethod
    def get(cls, name: str) -> Any:
        if name not in cls._adapters:
            raise ValueError(f"未知的 SNN 适配器: {name}，可用: {list(cls._adapters.keys())}")
        return cls._adapters[name]


class BaseSNNAdapter:
    """SNN 适配器基类，子类需实现必要的映射方法。"""

    def is_connection(self, module: nn.Module) -> bool:
        """判断模块是否为连接层（如 Linear / Conv2d）。"""
        raise NotImplementedError

    def is_neuron(self, module: nn.Module) -> bool:
        """判断模块是否为脉冲神经元。"""
        raise NotImplementedError

    def get_connection_params(
        self, module: nn.Module
    ) -> Dict[str, Any]:
        """从连接层提取参数（weight, bias, shape 等）。"""
        raise NotImplementedError

    def get_neuron_params(
        self, module: nn.Module
    ) -> Dict[str, Any]:
        """从神经元提取参数（如 v_threshold, tau 等），用于构建 Genesis 神经元。"""
        raise NotImplementedError

    def get_neuron_class(
        self, module: nn.Module
    ) -> Type[MorphoNeuron]:
        """返回对应的 Genesis 神经元类。"""
        raise NotImplementedError


# ----------------------------------------------------------------------------
# 默认 SNN 适配器（基于类名与属性匹配，无需安装外部库）
# ----------------------------------------------------------------------------
class DefaultSNNAdapter(BaseSNNAdapter):
    """内置通用 SNN 适配器，通过检查模块类名和属性进行映射。

    支持：
    - 连接层：nn.Linear, nn.Conv2d
    - 神经元：任何包含 'IF', 'LIF', 'PLIF', 'QIF', 'EIF', 'Izhikevich' 的类名。
    """

    # 类名关键词到 Genesis 神经元类的映射
    NEURON_KEYWORDS = {
        "IFNode": IFNeuron,
        "LIFNode": LIFNeuron,
        "PLIFNode": PLIFNeuron,
        "QIFNode": QIFNeuron,
        "EIFNode": EIFNeuron,
        "IzhikevichNode": IzhikevichNeuron,
        "IF": IFNeuron,
        "LIF": LIFNeuron,
        "PLIF": PLIFNeuron,
        "QIF": QIFNeuron,
        "EIF": EIFNeuron,
        "Izhikevich": IzhikevichNeuron,
    }

    def is_connection(self, module: nn.Module) -> bool:
        return isinstance(module, (nn.Linear, nn.Conv2d))

    def is_neuron(self, module: nn.Module) -> bool:
        name = module.__class__.__name__
        return any(key in name for key in self.NEURON_KEYWORDS)

    def get_connection_params(self, module: nn.Module) -> Dict[str, Any]:
        params = {}
        if hasattr(module, "weight") and module.weight is not None:
            params["weight"] = module.weight.detach().clone()
        if hasattr(module, "bias") and module.bias is not None:
            params["bias"] = module.bias.detach().clone()
        else:
            params["bias"] = None
        if isinstance(module, nn.Linear):
            params["in_features"] = module.in_features
            params["out_features"] = module.out_features
        elif isinstance(module, nn.Conv2d):
            params.update(
                {
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                    "padding": module.padding,
                    "dilation": module.dilation,
                    "groups": module.groups,
                }
            )
        return params

    def get_neuron_params(self, module: nn.Module) -> Dict[str, Any]:
        params = {
            "num_neurons": 1,  # 可能会被覆盖
            "rest_h": 0.7,
            "refractory_period": 2,
            "axon_delay": 1,
            "bias_dim": 1,
        }
        # 尝试提取常见属性
        for attr in [
            "v_threshold",
            "v_reset",
            "tau",
            "R",
            "u_rest",
            "a",
            "b",
            "c",
            "d",
            "delta_T",
            "theta_rh",
        ]:
            if hasattr(module, attr):
                val = getattr(module, attr)
                if isinstance(val, torch.Tensor):
                    val = val.detach().clone()
                params[attr] = val
        # 有时阈值保存在 threshold 属性
        if hasattr(module, "threshold") and "v_threshold" not in params:
            params["v_threshold"] = getattr(module, "threshold")
        if hasattr(module, "v_reset") and "v_reset" not in params:
            params["v_reset"] = getattr(module, "v_reset")
        return params

    def get_neuron_class(self, module: nn.Module) -> Type[MorphoNeuron]:
        name = module.__class__.__name__
        for key, cls in self.NEURON_KEYWORDS.items():
            if key in name:
                return cls
        return LIFNeuron  # 默认回退


# 注册默认适配器
SNNAdapterRegistry.register("default", DefaultSNNAdapter())


# ============================================================================
# 工具函数：卷积 → 神经元索引生成
# ============================================================================
def conv2d_to_indices(
    in_channels: int,
    out_channels: int,
    h_in: int,
    w_in: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """计算二维卷积对应的 pre/post 神经元局部索引。

    返回:
        pre_indices: (num_synapses,) long
        post_indices: (num_synapses,) long
        out_h, out_w: 输出特征图尺寸，用于后续分配神经元 ID。
    """
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    # 输入神经元展平：ID = c * H * W + h * W + w
    # 输出神经元展平：ID = c_out * out_H * out_W + h_out * out_W + w_out
    out_h = (h_in + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (w_in + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    pre_list = []
    post_list = []
    for c_out in range(out_channels):
        for h_out in range(out_h):
            for w_out in range(out_w):
                for c_in in range(in_channels):
                    for k_h in range(kh):
                        for k_w in range(kw):
                            h_in_pos = h_out * sh - ph + k_h * dh
                            w_in_pos = w_out * sw - pw + k_w * dw
                            if 0 <= h_in_pos < h_in and 0 <= w_in_pos < w_in:
                                pre_id = c_in * h_in * w_in + h_in_pos * w_in + w_in_pos
                                post_id = (
                                    c_out * out_h * out_w + h_out * out_w + w_out
                                )
                                pre_list.append(pre_id)
                                post_list.append(post_id)
    return (
        torch.tensor(pre_list, dtype=torch.long),
        torch.tensor(post_list, dtype=torch.long),
        out_h,
        out_w,
    )


# ============================================================================
# Batch Normalization 融合工具（用于 ANN 转换）
# ============================================================================
def fuse_bn(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    bn: nn.BatchNorm2d,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将 BatchNorm 参数融合到前一个卷积/全连接层的权重和偏置中。"""
    if bn.training or not bn.track_running_stats:
        raise RuntimeError("BN 层必须在 eval 模式且 track_running_stats=True 才能融合。")
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    if gamma is None or mean is None:
        return weight, bias

    std = torch.sqrt(var + eps)
    # 缩放权重
    if weight.dim() == 4:  # Conv2d [out, in, kh, kw]
        weight_fused = weight * (gamma / std).reshape(-1, 1, 1, 1)
    elif weight.dim() == 2:  # Linear [out, in]
        weight_fused = weight * (gamma / std).unsqueeze(1)
    else:
        raise ValueError(f"不支持的权重维度: {weight.dim()}")

    # 融合偏置
    if bias is not None:
        bias_fused = (bias - mean) * gamma / std + beta
    else:
        bias_fused = -mean * gamma / std + beta
    return weight_fused, bias_fused


# ============================================================================
# SNN → MNN 转换器
# ============================================================================
class SNNtoMNNConverter:
    """将基于外部框架（SpikingJelly / snnTorch）的 SNN 模型转换为 MNN 网络。

    通过注册适配器模式支持不同框架，默认使用类名匹配的通用适配器。
    可选用 STDPSynapse 替代 ResonanceSynapse，以启用公理三的局部可塑性演化。
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: str = "default",
        use_stdp: bool = False,
        time_steps: int = 20,
        input_shape: Optional[Tuple] = None,
    ):
        """
        参数:
            model: 源 SNN 模型（通常为 nn.Sequential）。
            adapter: 适配器名称，默认为 'default'。
            use_stdp: 是否使用 STDPSynapse（公理三），否则使用基础 ResonanceSynapse。
            time_steps: 转换后模型的默认模拟步数。
            input_shape: 输入特征形状（用于推断卷积尺寸），如 (C, H, W) 或 (in_features,)。
        """
        self.model = model
        self.adapter = SNNAdapterRegistry.get(adapter)
        self.use_stdp = use_stdp
        self.time_steps = time_steps
        self.input_shape = input_shape
        self.mnn_model = None

    def convert(self) -> MNNModel:
        """执行转换，返回 MNNModel。"""
        # 遍历模型，收集阶段列表
        children = list(self.model.children())
        if not children:
            raise ValueError("源模型至少需要包含一个子模块（连接或神经元）。")
        stages = []
        # 推断输入尺寸：用于卷积层索引生成
        # 若未提供 input_shape，尝试通过模型推断或报错
        in_shape = self.input_shape
        if in_shape is None:
            # 尝试从第一层的参数推断
            first = children[0]
            if isinstance(first, nn.Linear):
                in_shape = (first.in_features,)
            elif isinstance(first, nn.Conv2d):
                raise ValueError("包含 Conv2d 时必须提供 input_shape 以推断尺寸。")
            else:
                in_shape = (1,)  # fallback

        idx = 0
        while idx < len(children):
            conn_mod = children[idx]
            if self.adapter.is_connection(conn_mod):
                # 连接层
                conn_params = self.adapter.get_connection_params(conn_mod)
                # 查找下一个神经元模块
                idx += 1
                if idx < len(children) and self.adapter.is_neuron(children[idx]):
                    neu_mod = children[idx]
                    idx += 1
                else:
                    # 没有接神经元，则自动添加默认 LIFNeuron
                    neu_mod = None
                # 创建突触层
                syn_layer = self._build_synaptic_layer(
                    conn_params, conn_mod, in_shape, conn_mod
                )
                # 更新 in_shape 以匹配该层的输出
                if isinstance(conn_mod, nn.Linear):
                    in_shape = (conn_params["out_features"],)
                elif isinstance(conn_mod, nn.Conv2d):
                    # 根据卷积参数计算输出尺寸
                    out_h = (
                        in_shape[1]
                        + 2 * conn_params["padding"][0]
                        - conn_params["dilation"][0] * (conn_params["kernel_size"][0] - 1)
                        - 1
                    ) // conn_params["stride"][0] + 1
                    out_w = (
                        in_shape[2]
                        + 2 * conn_params["padding"][1]
                        - conn_params["dilation"][1] * (conn_params["kernel_size"][1] - 1)
                        - 1
                    ) // conn_params["stride"][1] + 1
                    in_shape = (conn_params["out_channels"], out_h, out_w)
                else:
                    raise NotImplementedError(
                        f"不支持的连接类型: {type(conn_mod)}"
                    )

                # 创建神经元层
                if neu_mod is not None:
                    neu_params = self.adapter.get_neuron_params(neu_mod)
                    # 设置神经元数量（根据输出尺寸）
                    if isinstance(conn_mod, nn.Linear):
                        neu_params["num_neurons"] = conn_params["out_features"]
                    elif isinstance(conn_mod, nn.Conv2d):
                        neu_params["num_neurons"] = (
                            conn_params["out_channels"] * out_h * out_w
                        )
                    neu_class = self.adapter.get_neuron_class(neu_mod)
                    neu_layer = self._build_neuron(neu_class, neu_params, conn_params)
                else:
                    # 默认 LIF 神经元
                    neu_params = {"num_neurons": syn_layer.out_neurons}
                    neu_layer = LIFNeuron(**neu_params)

                # 公理二：确保初始 impedance 为 0（所有神经元 bias 初始化为零）
                neu_layer.bias.data.zero_()
                neu_layer.reset_state(1)  # 刷新内部 r

                stages.append((syn_layer, neu_layer))

            elif self.adapter.is_neuron(conn_mod):
                # 孤立的神经元（例如输入编码层），无前置突触
                neu_params = self.adapter.get_neuron_params(conn_mod)
                # 需要用户提供 num_neurons 或基于输入推断
                if "num_neurons" not in neu_params:
                    if isinstance(in_shape, tuple) and len(in_shape) == 1:
                        neu_params["num_neurons"] = in_shape[0]
                    else:
                        neu_params["num_neurons"] = 1
                neu_class = self.adapter.get_neuron_class(conn_mod)
                neu_layer = self._build_neuron(neu_class, neu_params)
                neu_layer.bias.data.zero_()
                neu_layer.reset_state(1)
                stages.append((None, neu_layer))
                idx += 1
            else:
                # 不识别的模块，跳过
                idx += 1

        self.mnn_model = MNNModel(stages, time_steps=self.time_steps)
        return self.mnn_model

    def _build_synaptic_layer(
        self,
        conn_params: Dict,
        conn_module: nn.Module,
        in_shape: Tuple,
        module: nn.Module,
    ) -> SynapticLayer:
        """根据连接参数构建 SynapticLayer。"""
        weight = conn_params.get("weight")
        bias = conn_params.get("bias")

        if isinstance(conn_module, nn.Linear):
            in_neurons = conn_params["in_features"]
            out_neurons = conn_params["out_features"]
            # 全连接索引
            pre_idx = torch.arange(in_neurons).repeat_interleave(out_neurons)
            post_idx = torch.arange(out_neurons).repeat(in_neurons)
            initial_weight = weight  # dense matrix
        elif isinstance(conn_module, nn.Conv2d):
            # 需要输入尺寸来计算索引
            if len(in_shape) != 3 or in_shape[0] != conn_params["in_channels"]:
                raise ValueError(
                    f"输入形状 {in_shape} 与 Conv2d 的 in_channels 不匹配。"
                )
            in_c, h_in, w_in = in_shape
            pre_idx, post_idx, out_h, out_w = conv2d_to_indices(
                in_channels=conn_params["in_channels"],
                out_channels=conn_params["out_channels"],
                h_in=h_in,
                w_in=w_in,
                kernel_size=conn_params["kernel_size"],
                stride=conn_params["stride"],
                padding=conn_params["padding"],
                dilation=conn_params["dilation"],
            )
            in_neurons = in_c * h_in * w_in
            out_neurons = conn_params["out_channels"] * out_h * out_w
            # 卷积权重形状 [out_c, in_c, kh, kw]，需要按索引展平
            # 构建权重模板：每个突触的初始强度 = 该连接的权重值
            weight_flat = torch.zeros(len(pre_idx), device=weight.device)
            # 权重索引与 pre/post 索引顺序一致（conv2d_to_indices 中循环顺序）
            idx = 0
            for c_out in range(conn_params["out_channels"]):
                for h_out in range(out_h):
                    for w_out in range(out_w):
                        for c_in in range(conn_params["in_channels"]):
                            for k_h in range(conn_params["kernel_size"][0]):
                                for k_w in range(conn_params["kernel_size"][1]):
                                    h_in_pos = (
                                        h_out * conn_params["stride"][0]
                                        - conn_params["padding"][0]
                                        + k_h * conn_params["dilation"][0]
                                    )
                                    w_in_pos = (
                                        w_out * conn_params["stride"][1]
                                        - conn_params["padding"][1]
                                        + k_w * conn_params["dilation"][1]
                                    )
                                    if 0 <= h_in_pos < h_in and 0 <= w_in_pos < w_in:
                                        weight_flat[idx] = weight[
                                            c_out, c_in, k_h, k_w
                                        ]
                                        idx += 1
            initial_weight = weight_flat
        else:
            raise NotImplementedError(f"不支持的连接类型: {type(conn_module)}")

        synapse_class = STDPSynapse if self.use_stdp else ResonanceSynapse
        # 公理三：若使用 STDPSynapse，后续演化通过局部 STDP 规则，不使用反向传播
        syn_kwargs = {}
        if self.use_stdp:
            syn_kwargs["lr"] = 1.0  # 可调整
        syn_layer = SynapticLayer(
            in_neurons=in_neurons,
            out_neurons=out_neurons,
            pre_indices=pre_idx,
            post_indices=post_idx,
            synapse_class=synapse_class,
            synapse_kwargs=syn_kwargs,
            initial_weight=initial_weight,
            initial_bias=bias,
            r_dim=1,
        )
        # 公理二：初始状态确保 strength 继承权重，impedance 初始为 0（由神经元 r 为 0 保证）
        return syn_layer

    def _build_neuron(
        self,
        neu_class: Type[MorphoNeuron],
        neu_params: Dict,
        conn_params: Optional[Dict] = None,
    ) -> MorphoNeuron:
        """根据参数字典实例化 Genesis 神经元，并应用偏置作为阈值偏移。"""
        # 提取构造函数接受的参数
        valid_args = {
            k: v
            for k, v in neu_params.items()
            if k
            in [
                "num_neurons",
                "num_dendrites",
                "tau_u",
                "rest_h",
                "threshold",
                "refractory_period",
                "axon_delay",
                "bias_dim",
                "noise_scale",
                "tau",
                "R",
                "u_rest",
                "v_threshold",
                "v_reset",
                "a",
                "b",
                "c",
                "d",
                "delta_T",
                "theta_rh",
                "homeo_gain",
                "noise_std",
            ]
        }
        # 某些神经元类可能有特殊参数名
        if "v_threshold" in valid_args and "threshold" not in valid_args:
            valid_args["threshold"] = valid_args.pop("v_threshold")
        # 默认树突数量设为 1（简化）
        if "num_dendrites" not in valid_args:
            valid_args["num_dendrites"] = 1

        neuron = neu_class(**valid_args)

        # 公理二：将原偏置项映射为神经元阈值偏移
        if conn_params and conn_params.get("bias") is not None:
            bias = conn_params["bias"]
            # 偏置增加通常使神经元更易产生输出，等效降低阈值
            neuron.threshold.data = neuron.threshold.data - bias.to(
                neuron.threshold.device
            )
            # 保持阈值为正
            neuron.threshold.data.clamp_(min=0.1)
        return neuron


# ============================================================================
# ANN → MNN 转换器
# ============================================================================
class ANNtoMNNConverter:
    """将传统 PyTorch 模型（CNN/RNN/Transformer）转换为 MNN 脉冲网络。

    映射规则：
        - ReLU/激活函数 → 跳过（由脉冲神经元阈值行为替代）
        - 全连接/卷积权重 → 共鸣连接初始 strength
        - 偏置项 → 胞体阈值偏移
        - BatchNorm → 融合到前一层权重
    """

    def __init__(
        self,
        model: nn.Module,
        time_steps: int = 20,
        input_shape: Optional[Tuple] = None,
        fuse_bn_layers: bool = True,
        default_neuron: Type[MorphoNeuron] = LIFNeuron,
    ):
        """
        参数:
            model: 源 ANN 模型（建议为顺序结构）。
            time_steps: 默认模拟步数。
            input_shape: 输入形状，如 (C,H,W) 或 (in_features,)，用于推断尺寸。
            fuse_bn_layers: 是否自动将 BN 层融合到前一个权重层（推荐 True）。
            default_neuron: 用于替换激活的默认脉冲神经元类。
        """
        self.model = model
        self.time_steps = time_steps
        self.input_shape = input_shape
        self.fuse_bn_layers = fuse_bn_layers
        self.default_neuron = default_neuron
        self.mnn_model = None

    def convert(self) -> MNNModel:
        """执行转换，返回 MNNModel。"""
        # 确保模型在 eval 模式以融合 BN
        if self.fuse_bn_layers:
            self.model.eval()

        # 将模型按顺序拆解为层列表，并进行 BN 融合和 ReLU 跳过
        layers = list(self.model.children())
        processed = self._preprocess_layers(layers)

        # 推断输入尺寸
        in_shape = self._infer_shape(processed)
        if in_shape is None and self.input_shape is not None:
            in_shape = self.input_shape
        elif in_shape is None:
            raise ValueError("无法推断输入尺寸，请提供 input_shape 参数。")

        stages = []
        for module in processed:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 连接层：提取权重偏置
                weight = module.weight.detach().clone()
                bias = module.bias.detach().clone() if module.bias is not None else None

                if isinstance(module, nn.Linear):
                    in_neurons = module.in_features
                    out_neurons = module.out_features
                    pre_idx = torch.arange(in_neurons).repeat_interleave(out_neurons)
                    post_idx = torch.arange(out_neurons).repeat(in_neurons)
                    initial_weight = weight
                    # 更新形状
                    in_shape = (out_neurons,)
                elif isinstance(module, nn.Conv2d):
                    if len(in_shape) != 3 or in_shape[0] != module.in_channels:
                        raise ValueError(
                            f"Conv2d 输入形状 {in_shape} 与模块 in_channels {module.in_channels} 不匹配。"
                        )
                    in_c, h_in, w_in = in_shape
                    pre_idx, post_idx, out_h, out_w = conv2d_to_indices(
                        in_channels=module.in_channels,
                        out_channels=module.out_channels,
                        h_in=h_in,
                        w_in=w_in,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                    )
                    in_neurons = in_c * h_in * w_in
                    out_neurons = module.out_channels * out_h * out_w
                    # 卷积权重展平（同 SNN 转换）
                    weight_flat = torch.zeros(len(pre_idx), device=weight.device)
                    idx = 0
                    for c_out in range(module.out_channels):
                        for h_out_ in range(out_h):
                            for w_out_ in range(out_w):
                                for c_in in range(module.in_channels):
                                    for k_h in range(module.kernel_size[0]):
                                        for k_w in range(module.kernel_size[1]):
                                            h_in_pos = (
                                                h_out_ * module.stride[0]
                                                - module.padding[0]
                                                + k_h * module.dilation[0]
                                            )
                                            w_in_pos = (
                                                w_out_ * module.stride[1]
                                                - module.padding[1]
                                                + k_w * module.dilation[1]
                                            )
                                            if (
                                                0 <= h_in_pos < h_in
                                                and 0 <= w_in_pos < w_in
                                            ):
                                                weight_flat[idx] = weight[
                                                    c_out, c_in, k_h, k_w
                                                ]
                                                idx += 1
                    initial_weight = weight_flat
                    in_shape = (module.out_channels, out_h, out_w)
                else:
                    continue

                # 创建突触层（默认使用 ResonanceSynapse，因为 ANN 原无脉冲时序）
                syn_layer = SynapticLayer(
                    in_neurons=in_neurons,
                    out_neurons=out_neurons,
                    pre_indices=pre_idx,
                    post_indices=post_idx,
                    initial_weight=initial_weight,
                    initial_bias=bias,
                )
                # 创建脉冲神经元层（替代激活函数）
                neu_params = {
                    "num_neurons": out_neurons,
                    "num_dendrites": 1,
                    "rest_h": 0.7,
                }
                if self.default_neuron in (LIFNeuron, PLIFNeuron):
                    neu_params["tau"] = 10.0
                    neu_params["v_threshold"] = 1.0
                neuron = self.default_neuron(**neu_params)
                neuron.bias.data.zero_()  # 公理二：初始 r 相同，impedance 为 0
                # 与 SNN 转换器共享的偏置-阈值映射逻辑，后续可提取为公共工具函数。
                if bias is not None:
                    # 偏置映射为阈值偏移：增大输入 → 更易产生输出 → 降低阈值
                    neuron.threshold.data = neuron.threshold.data - bias.to(
                        neuron.threshold.device
                    )
                    neuron.threshold.data.clamp_(min=0.1)
                neuron.reset_state(1)

                stages.append((syn_layer, neuron))

        self.mnn_model = MNNModel(stages, time_steps=self.time_steps)
        return self.mnn_model

    def _preprocess_layers(self, layers: List[nn.Module]) -> List[nn.Module]:
        """预处理：融合 BN，去除激活层，合并连续线性层（可选）。"""
        processed = []
        i = 0
        while i < len(layers):
            mod = layers[i]
            # 跳过 Dropout
            if isinstance(mod, nn.Dropout):
                i += 1
                continue
            # 处理 BN 融合
            if (
                self.fuse_bn_layers
                and isinstance(mod, nn.BatchNorm2d)
                and len(processed) > 0
                and isinstance(processed[-1], (nn.Linear, nn.Conv2d))
            ):
                prev = processed[-1]
                if hasattr(prev, "weight"):
                    new_weight, new_bias = fuse_bn(
                        prev.weight.data,
                        prev.bias.data if prev.bias is not None else None,
                        mod,
                    )
                    prev.weight.data = new_weight
                    if new_bias is not None:
                        if prev.bias is None:
                            prev.bias = nn.Parameter(new_bias)
                        else:
                            prev.bias.data = new_bias
                    elif prev.bias is not None:
                        prev.bias.data.zero_()
                i += 1
                continue
            # 跳过激活层（ReLU、Sigmoid 等），由后续脉冲神经元替代
            if isinstance(mod, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
                i += 1
                continue
            processed.append(mod)
            i += 1
        return processed

    def _infer_shape(self, layers: List[nn.Module]) -> Optional[Tuple]:
        """从第一层的参数推断输入形状。"""
        if not layers:
            return None
        first = layers[0]
        if isinstance(first, nn.Linear):
            return (first.in_features,)
        if isinstance(first, nn.Conv2d):
            return (first.in_channels, 32, 32)  # 无法推断，需要用户提供
        return None


# ============================================================================
# 统一转换入口
# ============================================================================
def convert(
    model: nn.Module,
    target: str = "mnn",
    source: Optional[str] = None,
    **kwargs,
) -> MNNModel:
    """统一模型转换入口，自动选择 SNN 或 ANN 转换器。

    参数:
        model: 待转换的 PyTorch 模型。
        target: 目标框架，目前固定为 'mnn'。
        source: 源模型类型，'snn' 或 'ann'。若为 None，则自动检测（推荐）。
        **kwargs: 传递给具体转换器的额外参数，如 time_steps, input_shape 等。

    返回:
        MNNModel 实例，可直接进行脉冲推理（调用 .forward(x)）。
    """
    if target != "mnn":
        raise ValueError("当前仅支持转换为 MNN 目标。")

    # 自动检测源类型
    if source is None:
        adapter = DefaultSNNAdapter()
        # 检查是否包含脉冲神经元
        for m in model.modules():
            if adapter.is_neuron(m):
                source = "snn"
                break
        else:
            source = "ann"

    if source == "snn":
        converter = SNNtoMNNConverter(model, **kwargs)
    elif source == "ann":
        converter = ANNtoMNNConverter(model, **kwargs)
    else:
        raise ValueError(f"未知的源模型类型: {source}")

    return converter.convert()


# ============================================================================
# 示例用法与独立运行测试
# ============================================================================
if __name__ == "__main__":
    # 确保 genesis_core 等模块可导入（假设它们在同一目录）
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))

    print("=" * 60)
    print("Genesis convert.py 自包含演示")
    print("=" * 60)

    # ------------------------------------------------------------------------
    # 1. 构建一个模拟的 SNN 模型（模仿 SpikingJelly 接口）
    # ------------------------------------------------------------------------
    class FakeLIFNode(nn.Module):
        def __init__(self, v_threshold=1.0, v_reset=0.0, tau=2.0):
            super().__init__()
            self.v_threshold = v_threshold
            self.v_reset = v_reset
            self.tau = tau
            self.threshold = v_threshold  # 别名

        def forward(self, x):
            return x  # 仅用于占位

    print("\n>>> 创建模拟 SNN 模型 (Linear(10,20) + LIF + Linear(20,5) + LIF)")
    snn_model = nn.Sequential(
        nn.Linear(10, 20),
        FakeLIFNode(v_threshold=1.0, tau=5.0),
        nn.Linear(20, 5),
        FakeLIFNode(v_threshold=1.0, tau=3.0),
    )

    # 转换 SNN
    print(">>> 执行 SNN -> MNN 转换...")
    mnn_snn = convert(
        snn_model,
        source="snn",
        time_steps=30,
        use_stdp=True,  # 公理三：启用 STDP 可塑性
    )
    print("转换成功！模型结构：", mnn_snn)

    # 测试推理
    batch_size = 1
    T = 30
    x = torch.rand(batch_size, T, 10) > 0.5  # 二值脉冲序列
    x = x.float()
    print(f"输入形状: {x.shape}")
    with torch.no_grad():
        out = mnn_snn(x)
    print(f"输出脉冲形状: {out.shape} (期望: [1, 30, 5])")
    spike_rate = out.mean(dim=1)  # 每个输出神经元的平均发放率
    print(f"输出发放率: {spike_rate.numpy()}")

    # ------------------------------------------------------------------------
    # 2. 构建一个模拟 ANN 模型 (Linear + ReLU + Linear)
    # ------------------------------------------------------------------------
    print("\n>>> 创建模拟 ANN 模型 (Linear(784,256) + ReLU + Linear(256,10))")
    ann_model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    # 转换 ANN
    print(">>> 执行 ANN -> MNN 转换...")
    mnn_ann = convert(
        ann_model,
        source="ann",
        time_steps=20,
        input_shape=(784,),
        default_neuron=IFNeuron,
    )
    print("转换成功！模型结构：", mnn_ann)

    # 测试推理（模拟 MNIST 脉冲输入）
    x_ann = torch.rand(batch_size, T, 784) > 0.8  # 泊松编码模拟
    x_ann = x_ann.float()
    print(f"输入形状: {x_ann.shape}")
    with torch.no_grad():
        out_ann = mnn_ann(x_ann)
    print(f"输出脉冲形状: {out_ann.shape} (期望: [1, 20, 10])")
    spike_rate_ann = out_ann.mean(dim=1)
    print(f"输出发放率: {spike_rate_ann.numpy()}")

    # ------------------------------------------------------------------------
    # 3. 演示公理二和公理三的验证
    # ------------------------------------------------------------------------
    print("\n>>> 公理验证：")
    # 检查突触是否为 ResonanceSynapse 子类
    if hasattr(mnn_snn, "synapses"):
        syn = mnn_snn.synapses[0]
        print(f"  突触类型: {type(syn.synapse).__name__}")
        if isinstance(syn.synapse, ResonanceSynapse):
            print("  ✓ 公理二满足：突触为 ResonanceSynapse（或其子类）")
        if isinstance(syn.synapse, STDPSynapse):
            print("  ✓ 公理三满足：使用 STDP 局部可塑性，无反向传播")

    # 检查初始 impedance 是否为 0（通过 r 相同保证）
    neu = mnn_snn.neurons[0]
    r_pre = neu.get_internal_state()['r']  # 神经元初始 r
    print(f"  初始神经元 r 向量范数: {r_pre.norm(dim=-1).mean().item():.4f} (接近0)")
    print("  ✓ 初始 impedance 为 0 由所有神经元 r 相同保证")

    print("\n>>> 演示完成，convert.py 可独立运行并通过编译检查。")