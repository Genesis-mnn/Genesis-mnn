# dendritic_spine_sleep.py — Genesis v0.3 树突棘睡眠重塑模块
#
# 在睡眠状态下，树突棘的重塑规则切换到更激进的模式：
# - 新生阈值从默认的 0.8 降低到 0.5（更容易新生）。
# - 回缩阈值从默认的 0.2 降低到 0.1（更容易回缩）。
# - 只有被胶质细胞标记为“保护”的突触对应的树突棘免于回缩。
#
# 严格遵循 Genesis v2.6 白皮书核心公理及术语中立宪章。

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple

from .genesis_core import MorphoNeuron, ResonanceSynapse


class DendriticSpineSleepRemodeling:
    """睡眠期间树突棘选择性重塑模块。

    在睡眠状态下被 ``SleepWakeCycle`` 调用，对所有神经元的树突棘执行
    特异性的新生与回缩，让网络在睡眠中进行更自由的结构探索。

    理论依据（白皮书 v2.6 第一章及第十五章）：
    睡眠期间树突棘的消除和巩固具有高度选择性——与学习相关的树突棘被加强，
    无关的被清除。本模块通过降低新生/回缩阈值并引入保护机制实现这一逻辑。

    构造参数:
        neurons: 形态神经元列表。每个实例的 ``num_neurons`` 应为 1，
                 以精确控制单个神经元的树突棘重塑。
        connections: 突触连接列表，每个元素为 (synapse, pre_neuron, post_neuron)。
                     用于获取每个树突棘对应的保护标志 (``protect_flag``)。
                     若为 None，则所有树突棘均视为未保护，回缩逻辑正常执行但
                     不享受豁免。
        spine_birth_threshold: 树突棘新生钙浓度阈值，默认 0.5（睡眠模式）。
        spine_retraction_threshold: 树突棘回缩钙浓度阈值，默认 0.1（睡眠模式）。
        max_total_spines: 单个神经元允许的最大树突棘数量，默认 32。
        min_total_spines: 单个神经元允许的最小树突棘数量，默认 1。
        protection_mode: 保护策略。'synapse' 表示从突触的 ``protected`` 标志获取；
                         'none' 表示不启用保护（所有树突棘均可回缩）。
    """

    def __init__(
        self,
        neurons: List[MorphoNeuron],
        connections: Optional[List[Tuple[ResonanceSynapse, MorphoNeuron, MorphoNeuron]]] = None,
        spine_birth_threshold: float = 0.5,
        spine_retraction_threshold: float = 0.1,
        max_total_spines: int = 32,
        min_total_spines: int = 1,
        protection_mode: str = 'synapse',
    ) -> None:
        # 基本参数验证
        for nrn in neurons:
            if nrn.num_neurons != 1:
                raise ValueError(
                    f"DendriticSpineSleepRemodeling 要求每个神经元实例的 num_neurons=1，"
                    f"但发现实例含 {nrn.num_neurons} 个神经元。请将层拆分为单个神经元实例。"
                )
        self.neurons = neurons
        self.connections = connections or []
        self.spine_birth_threshold = spine_birth_threshold
        self.spine_retraction_threshold = spine_retraction_threshold
        self.max_total_spines = max_total_spines
        self.min_total_spines = min_total_spines
        self.protection_mode = protection_mode

        # 内部统计（每步更新）
        self._last_new_spines: int = 0
        self._last_retracted_spines: int = 0
        self._last_net_change: int = 0

        # 预构建神经元到传入保护标志的映射（可选优化）
        self._protection_cache: Dict[int, torch.Tensor] = {}
        if self.protection_mode == 'synapse' and self.connections:
            self._build_protection_cache()

    def _build_protection_cache(self) -> None:
        """遍历 connections 为每个神经元构建树突棘保护标志向量。

        映射规则（假设）：
        每个连接中的突触（可能包含多个子突触）对应于后神经元的连续树突棘索引。
        对于每个后神经元，按连接出现的顺序，将其保护标志张量展平并截断/填充
        至当前树突棘数量，存入缓存。

        若连接结构复杂导致映射不正确，可重写本方法。
        """
        protection_dict: Dict[int, List[float]] = {id(n): [] for n in self.neurons}
        for syn, pre_nrn, post_nrn in self.connections:
            post_id = id(post_nrn)
            if post_id not in protection_dict:
                continue
            # 获取该突触实例的保护标志：形状可能为 (batch, num_synapses) 或 (batch,) 或标量
            try:
                protected = syn.protected  # 直接访问缓冲区
                if protected.ndim == 2:
                    # 形状 (batch, num_synapses)：沿批次维度取逻辑或，即任意样本保护则保护
                    prot_flag = protected.any(dim=0).float().tolist()
                elif protected.ndim == 1:
                    # 形状 (batch,)：取逻辑或
                    prot_flag = [float(protected.any().item())]
                else:
                    prot_flag = [float(protected.item())] if protected.numel() == 1 else [0.0]
            except AttributeError:
                prot_flag = [0.0] * syn.num_synapses
            protection_dict[post_id].extend(prot_flag)

        self._protection_cache.clear()
        for nrn in self.neurons:
            nid = id(nrn)
            flags = protection_dict.get(nid, [])
            num_spines = nrn.num_dendrites
            if len(flags) >= num_spines:
                # 取前 num_spines 个
                flags = flags[:num_spines]
            else:
                # 不足部分填 0（未保护）
                flags.extend([0.0] * (num_spines - len(flags)))
            self._protection_cache[nid] = torch.tensor(flags, dtype=torch.bool)

    def _get_protection_mask(self, neuron: MorphoNeuron) -> torch.Tensor:
        """返回指定神经元当前的树突棘保护掩码，形状 (num_dendrites,)，bool 类型。"""
        if self.protection_mode == 'synapse' and id(neuron) in self._protection_cache:
            # 缓存可能过期（树突棘数量变化），重新对齐长度
            mask = self._protection_cache[id(neuron)]
            num_spines = neuron.num_dendrites
            if mask.shape[0] > num_spines:
                mask = mask[:num_spines]
            elif mask.shape[0] < num_spines:
                # 新生的树突棘默认未保护
                pad = torch.zeros(num_spines - mask.shape[0], dtype=torch.bool)
                mask = torch.cat([mask, pad], dim=0)
            return mask
        else:
            # 无保护信息或模式为 'none'，全部未保护
            return torch.zeros(neuron.num_dendrites, dtype=torch.bool)

    def remodel_step(self) -> None:
        """执行一步树突棘睡眠重塑。

        该方法应由 ``SleepWakeCycle`` 在睡眠状态下周期性调用。
        对每个神经元独立决策：
        1. 计算各树突棘的平均钙浓度（取批次均值）。
        2. 获取当前保护掩码。
        3. 标记回缩候选：钙浓度低于回缩阈值且未被保护。
        4. 标记新生候选：钙浓度高于新生阈值。
        5. 执行删除与新增操作，更新树突棘数量与缓冲区。
        6. 记录本步新生的树突棘数、回缩的树突棘数及净变化。

        异常:
            RuntimeError: 若神经元状态未初始化（树突棘缓冲区为空）。
        """
        total_new = 0
        total_retracted = 0

        for nrn in self.neurons:
            if nrn._spine_ca.numel() == 0:
                raise RuntimeError("神经元状态未初始化，请先调用 reset_state。")

            # 计算当前批次的平均钙浓度（取 batch 平均，形状 (num_dendrites,)）
            spine_ca_batch = nrn._spine_ca
            if spine_ca_batch.ndim == 3:
                ca_avg = spine_ca_batch.mean(dim=0).mean(dim=1)  # (num_dendrites,)
            else:
                ca_avg = spine_ca_batch.mean(dim=0)  # fallback

            # 获取保护掩码
            protected_mask = self._get_protection_mask(nrn)

            # 回缩候选：钙浓度 < 回缩阈值 且 未保护
            retraction_candidates = (ca_avg < self.spine_retraction_threshold) & (~protected_mask)
            spines_to_remove = torch.where(retraction_candidates)[0]

            # 新生候选：钙浓度 > 新生阈值（只要有至少一个高钙树突棘且未达上限）
            can_birth = (ca_avg > self.spine_birth_threshold).any().item()
            current_dendrites = nrn.num_dendrites
            allow_birth = current_dendrites < self.max_total_spines

            # 执行回缩
            if spines_to_remove.numel() > 0:
                # 保证回缩后不低于最小树突棘数
                num_to_remove = min(spines_to_remove.numel(), current_dendrites - self.min_total_spines)
                if num_to_remove > 0:
                    # 按索引从大到小排序后删除（避免索引偏移）
                    to_remove = spines_to_remove[-num_to_remove:].sort(descending=True)[0]
                    self._delete_spines(nrn, to_remove)
                    total_retracted += num_to_remove

            # 执行新生（至少有一个高钙树突棘且未满）
            if can_birth and allow_birth:
                self._add_spine(nrn)
                total_new += 1

        # 更新统计信息
        self._last_new_spines = total_new
        self._last_retracted_spines = total_retracted
        self._last_net_change = total_new - total_retracted

    def _delete_spines(self, neuron: MorphoNeuron, indices: torch.Tensor) -> None:
        """从神经元中移除指定索引的树突棘。

        参数:
            neuron: 目标神经元实例。
            indices: 待删除的树突棘索引（已排序降序）。
        """
        if indices.numel() == 0:
            return
        device = neuron._spine_ca.device
        keep_mask = torch.ones(neuron.num_dendrites, dtype=torch.bool, device=device)
        keep_mask[indices] = False
        new_num = keep_mask.sum().item()

        # 更新 _spine_ca: (batch, D, N) -> (batch, new_D, N)
        neuron.register_buffer('_spine_ca', neuron._spine_ca[:, keep_mask, :].clone())
        # 更新 _spine_rho
        neuron.register_buffer('_spine_rho', neuron._spine_rho[:, keep_mask, :].clone())
        # 更新 _spine_g: (D, N) -> (new_D, N)
        neuron.register_buffer('_spine_g', neuron._spine_g[keep_mask, :].clone())
        neuron.num_dendrites = new_num

    def _add_spine(self, neuron: MorphoNeuron) -> None:
        """向神经元添加一个树突棘（追加在末尾）。"""
        if neuron.num_dendrites >= self.max_total_spines:
            return
        D = neuron.num_dendrites
        B, _, N = neuron._spine_ca.shape
        device = neuron._spine_ca.device
        dtype = neuron._spine_ca.dtype

        new_ca = torch.zeros(B, 1, N, device=device, dtype=dtype)
        new_rho = torch.full((B, 1, N), 0.5, device=device, dtype=dtype)
        new_g = torch.zeros(1, N, device=device, dtype=dtype)

        neuron.register_buffer('_spine_ca', torch.cat([neuron._spine_ca, new_ca], dim=1))
        neuron.register_buffer('_spine_rho', torch.cat([neuron._spine_rho, new_rho], dim=1))
        neuron.register_buffer('_spine_g', torch.cat([neuron._spine_g, new_g], dim=0))
        neuron.num_dendrites += 1

    def get_remodel_statistics(self) -> Dict[str, int]:
        """返回最近一次 ``remodel_step()`` 的重塑统计信息。

        返回值字典包含以下键：
            - 'new_spines': 本步新生的树突棘总数。
            - 'retracted_spines': 本步回缩的树突棘总数。
            - 'net_change': 净变化（新生 - 回缩）。
        """
        return {
            'new_spines': self._last_new_spines,
            'retracted_spines': self._last_retracted_spines,
            'net_change': self._last_net_change,
        }

    def update_protection_from_glia(self) -> None:
        """基于当前 connections 的最新保护状态刷新内部保护缓存。

        应在小胶质细胞更新保护标志后调用，以确保回缩豁免实时生效。
        """
        if self.protection_mode == 'synapse' and self.connections:
            self._build_protection_cache()