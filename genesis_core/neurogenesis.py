# genesis_core/neurogenesis.py
import math
import random
from typing import Dict, List, Optional, Tuple, Type, Any
import torch

from .genesis_core import MorphoNeuron, ResonanceSynapse
from .neuron import LIFNeuron, IFNeuron, PLIFNeuron, QIFNeuron, EIFNeuron, IzhikevichNeuron
from .synapse import STDPSynapse, PlasticityRule
from .glia import Astrocyte, Microglia, GlialNetwork


# ============================================================================
# 1. NeurogenesisPool —— 神经干细胞池
# ============================================================================

class NeurogenesisPool:
    """神经干细胞池，按概率生成新的形态神经元。"""

    def __init__(
        self,
        neuron_class: Type[MorphoNeuron] = LIFNeuron,
        param_variability: float = 0.1,
    ):
        if not issubclass(neuron_class, MorphoNeuron):
            raise TypeError("neuron_class 必须是 MorphoNeuron 的子类。")
        self.neuron_class = neuron_class
        self.param_variability = param_variability

    def spawn_neuron(self, **extra_kwargs: Any) -> MorphoNeuron:
        kwargs: Dict[str, Any] = {
            'num_neurons': 1,
            'num_dendrites': 1,
            'rest_h': 0.8,
        }
        kwargs.update(extra_kwargs)

        cls = self.neuron_class
        if cls in (LIFNeuron, PLIFNeuron, QIFNeuron, EIFNeuron):
            kwargs.setdefault('tau', 10.0)
            kwargs.setdefault('v_threshold', 1.0)
        elif cls == IFNeuron:
            kwargs.setdefault('v_threshold', 1.0)
        elif cls == IzhikevichNeuron:
            kwargs.setdefault('v_threshold', 30.0)

        neuron = cls(**kwargs)

        variability = self.param_variability
        if hasattr(neuron, 'tau') and isinstance(neuron.tau, torch.Tensor):
            with torch.no_grad():
                factor = 1.0 + random.uniform(-variability, variability)
                neuron.tau.data *= factor

        thr_attr = 'threshold'
        if hasattr(neuron, thr_attr):
            with torch.no_grad():
                factor = 1.0 + random.uniform(-variability, variability)
                getattr(neuron, thr_attr).data *= factor

        if hasattr(neuron, 'noise_scale'):
            neuron.noise_scale *= (1.0 + random.uniform(-variability, variability))

        if hasattr(neuron, 'bias') and isinstance(neuron.bias, torch.Tensor):
            r_dim = neuron.bias.shape[-1]
            new_bias = torch.zeros(1, r_dim)
            new_bias[0, 0] = 0.5
            with torch.no_grad():
                neuron.bias.data = new_bias.to(neuron.bias.device)

        neuron.reset_state(1)
        return neuron


# ============================================================================
# 2. 神经元凋亡检测
# ============================================================================

def check_apoptosis(
    neurons: List[MorphoNeuron],
    connections: List[Tuple[ResonanceSynapse, MorphoNeuron, MorphoNeuron]],
    silent_counters: Dict[int, int],
    silence_window: int = 10000,
) -> List[MorphoNeuron]:
    """检测所有满足凋亡条件的神经元，返回神经元实例列表。"""
    candidates = []
    syn_strengths: Dict[int, List[float]] = {id(n): [] for n in neurons}
    for syn, pre, post in connections:
        if id(post) in syn_strengths:
            s = syn.get_synaptic_state()['strength']
            val = s[0, 0].item() if s.numel() > 0 else 0.0
            syn_strengths[id(post)].append(val)
        if id(pre) in syn_strengths:
            s = syn.get_synaptic_state()['strength']
            val = s[0, 0].item() if s.numel() > 0 else 0.0
            syn_strengths[id(pre)].append(val)

    for nrn in neurons:
        nid = id(nrn)
        state = nrn.get_internal_state()
        h = state['h'][0, 0].item()

        if h < 0.05:
            candidates.append(nrn)
            continue

        if silent_counters.get(nid, 0) >= silence_window:
            candidates.append(nrn)
            continue

        strengths = syn_strengths.get(nid, [])
        if len(strengths) > 0 and all(s < 1e-8 for s in strengths):
            candidates.append(nrn)
            continue

    return candidates


# ============================================================================
# 3. NeurogenesisScheduler —— 生死循环全局调度器
# ============================================================================

class NeurogenesisScheduler:
    """统一管理神经发生与凋亡的调度器。"""

    def __init__(
        self,
        neurons: List[MorphoNeuron],
        connections: List[Tuple[ResonanceSynapse, MorphoNeuron, MorphoNeuron]],
        glia: GlialNetwork,
        p_neurogenesis_base: float = 0.0001,
        apoptosis_delay: int = 100,
        silence_window: int = 10000,
        param_variability: float = 0.1,
        neuron_class: Type[MorphoNeuron] = LIFNeuron,
    ):
        if not neurons:
            raise ValueError("初始神经元列表不能为空。")
        self._neurons = neurons   # 外部引用，直接操作
        self._connections = connections
        self.glia = glia
        self.p_base = p_neurogenesis_base
        self.apoptosis_delay = apoptosis_delay
        self.silence_window = silence_window

        self.pool = NeurogenesisPool(neuron_class=neuron_class, param_variability=param_variability)

        self.apoptosis_queue: Dict[int, int] = {}
        self.silent_counters: Dict[int, int] = {id(n): 0 for n in neurons}

        existing_synapses = {micro.synapse for micro in glia.microglias}
        for syn, _, _ in connections:
            if syn not in existing_synapses:
                micro = Microglia(synapse=syn)
                glia.microglias.append(micro)
                existing_synapses.add(syn)

    def record_spikes(self, spikes: Dict[int, float]) -> None:
        for nid in self.silent_counters:
            s = spikes.get(nid, 0.0)
            if s == 0.0:
                self.silent_counters[nid] += 1
            else:
                self.silent_counters[nid] = 0

    def step(self) -> Dict[str, Any]:
        p = self._compute_p_neurogenesis()

        candidates = check_apoptosis(
            self._neurons, self._connections, self.silent_counters, self.silence_window
        )
        for nrn in candidates:
            nid = id(nrn)
            if nid not in self.apoptosis_queue:
                self.apoptosis_queue[nid] = self.apoptosis_delay

        executed = []
        for nid in list(self.apoptosis_queue.keys()):
            self.apoptosis_queue[nid] -= 1
            if self.apoptosis_queue[nid] <= 0:
                nrn = self._find_neuron_by_id(nid)
                if nrn is not None:
                    self._execute_apoptosis(nrn)
                executed.append(nid)
        for nid in executed:
            del self.apoptosis_queue[nid]

        if random.random() < p:
            self._perform_neurogenesis()

        stats = self.get_statistics()
        stats['p_neurogenesis'] = p
        return stats

    def _find_neuron_by_id(self, nid: int) -> Optional[MorphoNeuron]:
        for n in self._neurons:
            if id(n) == nid:
                return n
        return None

    def get_statistics(self) -> Dict[str, Any]:
        n_neurons = len(self._neurons)
        n_conns = len(self._connections)
        queue_len = len(self.apoptosis_queue)
        if n_neurons > 0:
            mean_h = 0.0
            for nrn in self._neurons:
                state = nrn.get_internal_state()
                mean_h += state['h'][0, 0].item()
            mean_h /= n_neurons
        else:
            mean_h = 0.0
        return {
            'active_neurons': n_neurons,
            'synaptic_connections': n_conns,
            'apoptosis_queue_length': queue_len,
            'global_mean_health': mean_h,
        }

    def _compute_p_neurogenesis(self) -> float:
        stats = self.get_statistics()
        mean_h = stats['global_mean_health']
        if mean_h < 0.4:
            factor = 1.0 + (0.4 - mean_h) * 10.0
            factor = min(5.0, factor)
        elif mean_h > 0.8:
            factor = 1.0 - (mean_h - 0.8) * 4.5
            factor = max(0.1, factor)
        else:
            factor = 1.0
        return self.p_base * factor

    def _execute_apoptosis(self, neuron: MorphoNeuron) -> None:
        nid = id(neuron)
        if neuron not in self._neurons:
            return
        self._neurons.remove(neuron)
        if nid in self.silent_counters:
            del self.silent_counters[nid]

        to_remove = []
        affected_synapses = set()
        for i, (syn, pre, post) in enumerate(self._connections):
            if pre is neuron or post is neuron:
                to_remove.append(i)
                affected_synapses.add(syn)

        for i in reversed(to_remove):
            del self._connections[i]

        self.glia.microglias = [
            micro for micro in self.glia.microglias
            if micro.synapse not in affected_synapses
        ]

        for astro in self.glia.astrocytes:
            astro.neuron_refs = [
                ref for ref in astro.neuron_refs
                if ref['obj'] is not neuron
            ]
            astro.synapse_refs = [
                ref for ref in astro.synapse_refs
                if ref['obj'] not in affected_synapses
            ]

    def _perform_neurogenesis(self) -> None:
        neuron = self.pool.spawn_neuron()
        self._neurons.append(neuron)
        nid = id(neuron)
        self.silent_counters[nid] = 0

        existing_neurons = [n for n in self._neurons if n is not neuron]
        if not existing_neurons:
            return

        num_connections = random.randint(5, min(10, len(existing_neurons)))
        selected = random.sample(existing_neurons, num_connections)

        new_synapses = []
        for target in selected:
            if random.random() < 0.5:
                pre, post = target, neuron
            else:
                pre, post = neuron, target

            syn = STDPSynapse(num_synapses=1)
            with torch.no_grad():
                syn.strength.data.fill_(0.01)
            self._connections.append((syn, pre, post))
            new_synapses.append(syn)

        if self.glia.astrocytes:
            astro = random.choice(self.glia.astrocytes)
        else:
            astro = Astrocyte(
                astrocyte_id=0,
                neuron_groups=[],
                synapse_groups=[],
                position=(0.0, 0.0, 0.0),
            )
            self.glia.astrocytes.append(astro)

        neuron.register_astrocyte(astro.id)
        astro.neuron_refs.append({
            'obj': neuron,
            'indices': torch.tensor([0], dtype=torch.long),
            'orig_threshold': neuron.threshold.data.clone(),
            'orig_rest_h': neuron.rest_h.data.clone(),
        })

        for syn in new_synapses:
            micro = Microglia(synapse=syn)
            self.glia.microglias.append(micro)

        for syn in new_synapses:
            astro.synapse_refs.append({
                'obj': syn,
                'indices': torch.tensor([0], dtype=torch.long),
            })