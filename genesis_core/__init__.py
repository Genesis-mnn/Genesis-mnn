# genesis_core/__init__.py
# Genesis 框架公共 API 聚合并入口
# 遵循术语中立宪章，版本 0.2.0

__version__ = "0.2.0"

# =============================================================================
# 基础核心 —— 形态神经元、共鸣突触、全局常量
# =============================================================================
from .genesis_core import (
    MorphoNeuron,
    ResonanceSynapse,
    MODULATOR_REWARD,
    MODULATOR_SATIETY,
    MODULATOR_NOVELTY,
    MODULATOR_STRESS,
    MODULATOR_FATIGUE,
    GLOBAL_REWARD_ERROR,
    GLOBAL_SATIETY,
    GLOBAL_NOVELTY,
    GLOBAL_STRESS,
    GLOBAL_ENERGY_DEFICIT,
    get_neuron_type_map,
)

# =============================================================================
# 预置神经元类型
# =============================================================================
from .neuron import (
    IFNeuron,
    LIFNeuron,
    PLIFNeuron,
    QIFNeuron,
    EIFNeuron,
    IzhikevichNeuron,
)

# =============================================================================
# 预置突触及可塑性规则
# =============================================================================
from .synapse import (
    PlasticityRule,
    STDPSynapse,
)

# =============================================================================
# 胶质细胞网络
# =============================================================================
from .glia import (
    Astrocyte,
    Microglia,
    GlialNetwork,
)

# =============================================================================
# 调制场系统及预置调制原
# =============================================================================
from .modulator import (
    Modulator,
    RewardSignal,
    SatietyFactor,
    NoveltyResponder,
    StressActivator,
    FatigueAccumulator,
    ModulatorSystem,
    create_default_modulators,
)

# =============================================================================
# 全局状态生成与调度
# =============================================================================
from .global_state import (
    GlobalStateMonitor,
    GlobalStateScheduler,
)

# =============================================================================
# 脉冲编码与解码
# =============================================================================
from .encoding import (
    BaseEncoder,
    RateEncoder,
    TemporalEncoder,
    PhaseEncoder,
    Decoder,
)

# =============================================================================
# 模型转换工具 (SNN/ANN → MNN)
# =============================================================================
from .convert import (
    SNNtoMNNConverter,
    ANNtoMNNConverter,
    convert,
    SynapticLayer,
    MNNModel,
    SNNAdapterRegistry,
    BaseSNNAdapter,
    DefaultSNNAdapter,
    fuse_bn,
    conv2d_to_indices,
)

# =============================================================================
# 监视与可视化
# =============================================================================
from .monitor import (
    BaseMonitor,
    SpikeMonitor,
    MembraneMonitor,
    WeightMonitor,
    PlasticityMonitor,
    ModulatorMonitor,
    replay_membrane,
    replay_modulator_field,
)

# =============================================================================
# 量化与模型压缩
# =============================================================================
from .quantize import (
    PrecisionConfig,
    QuantConfig,
    PRECISION_MAP,
    compute_quant_params,
    quantize_tensor,
    dequantize_tensor,
    fake_quantize,
    weight_rescale,
    ThresholdCenteredQuantize,
    FakeQuantize,
    QuantMorphoNeuron,
    QuantResonanceSynapse,
    quantize_model_ptq,
    QATModel,
    prepare_qat,
    convert_qat,
    quantize_model,
    save_quantized_model,
    load_quantized_model,
    apply_precision,
)

# =============================================================================
# 网络拓扑与集群管理
# =============================================================================
from .utils import (
    Layer,
    Assembly,
    Column,
    BrainRegion,
    Connection,
    create_assembly,
    create_layer,
    create_column,
    create_brain_region,
    connect,
    generate_sparse_connectivity,
    load_neuromorphic_dataset,
    create_custom_dataloader,
)

# =============================================================================
# .life 文件持久化
# =============================================================================
from .life import (
    GenesisNetwork,
    EvolutionHasher,
    LifeSaver,
    LifeLoader,
    LifeRegistry,
    SynapticConnection,
    IntegrityError,
    LifeFormatError,
)

# =============================================================================
# 统一运行器
# =============================================================================
from .simulator import (
    GenesisSimulator,
    create_default_simulator,
)

# =============================================================================
# 硬件抽象层 (HAL)
# =============================================================================
from .hal import (
    GenesisBackend,
    CPUBackend,
    CUDABackend,
    AscendBackend,
    ThirdPartyBackend,
    register_third_party_backend,
    get_available_backends,
    set_active_backend,
    get_active_backend,
)

# =============================================================================
# 学习与替代梯度训练
# =============================================================================
from .learning import (
    SpikeFunction,
    SpikingLIFCell,
    STDPLinear,
    MNNBlock,
    SpikingMNNModel,
    IntrinsicLearningScheduler,
    register_surrogate,
    get_surrogate,
)

# =============================================================================
# 神经发生与凋亡
# =============================================================================
from .neurogenesis import (
    NeurogenesisPool,
    check_apoptosis,
    NeurogenesisScheduler,
)

# =============================================================================
# 公共 API 列表
# =============================================================================
__all__ = [
    # 版本
    "__version__",

    # genesis_core
    "MorphoNeuron",
    "ResonanceSynapse",
    "MODULATOR_REWARD",
    "MODULATOR_SATIETY",
    "MODULATOR_NOVELTY",
    "MODULATOR_STRESS",
    "MODULATOR_FATIGUE",
    "GLOBAL_REWARD_ERROR",
    "GLOBAL_SATIETY",
    "GLOBAL_NOVELTY",
    "GLOBAL_STRESS",
    "GLOBAL_ENERGY_DEFICIT",
    "get_neuron_type_map",

    # neuron
    "IFNeuron",
    "LIFNeuron",
    "PLIFNeuron",
    "QIFNeuron",
    "EIFNeuron",
    "IzhikevichNeuron",

    # synapse
    "PlasticityRule",
    "STDPSynapse",

    # glia
    "Astrocyte",
    "Microglia",
    "GlialNetwork",

    # modulator
    "Modulator",
    "RewardSignal",
    "SatietyFactor",
    "NoveltyResponder",
    "StressActivator",
    "FatigueAccumulator",
    "ModulatorSystem",
    "create_default_modulators",

    # global_state
    "GlobalStateMonitor",
    "GlobalStateScheduler",

    # encoding
    "BaseEncoder",
    "RateEncoder",
    "TemporalEncoder",
    "PhaseEncoder",
    "Decoder",

    # convert
    "SNNtoMNNConverter",
    "ANNtoMNNConverter",
    "convert",
    "SynapticLayer",
    "MNNModel",
    "SNNAdapterRegistry",
    "BaseSNNAdapter",
    "DefaultSNNAdapter",
    "fuse_bn",
    "conv2d_to_indices",

    # monitor
    "BaseMonitor",
    "SpikeMonitor",
    "MembraneMonitor",
    "WeightMonitor",
    "PlasticityMonitor",
    "ModulatorMonitor",
    "replay_membrane",
    "replay_modulator_field",

    # quantize
    "PrecisionConfig",
    "QuantConfig",
    "PRECISION_MAP",
    "compute_quant_params",
    "quantize_tensor",
    "dequantize_tensor",
    "fake_quantize",
    "weight_rescale",
    "ThresholdCenteredQuantize",
    "FakeQuantize",
    "QuantMorphoNeuron",
    "QuantResonanceSynapse",
    "quantize_model_ptq",
    "QATModel",
    "prepare_qat",
    "convert_qat",
    "quantize_model",
    "save_quantized_model",
    "load_quantized_model",
    "apply_precision",

    # utils
    "Layer",
    "Assembly",
    "Column",
    "BrainRegion",
    "Connection",
    "create_assembly",
    "create_layer",
    "create_column",
    "create_brain_region",
    "connect",
    "generate_sparse_connectivity",
    "load_neuromorphic_dataset",
    "create_custom_dataloader",

    # life
    "GenesisNetwork",
    "EvolutionHasher",
    "LifeSaver",
    "LifeLoader",
    "LifeRegistry",
    "SynapticConnection",
    "IntegrityError",
    "LifeFormatError",

    # simulator
    "GenesisSimulator",
    "create_default_simulator",

    # hal
    "GenesisBackend",
    "CPUBackend",
    "CUDABackend",
    "AscendBackend",
    "ThirdPartyBackend",
    "register_third_party_backend",
    "get_available_backends",
    "set_active_backend",
    "get_active_backend",

    # learning
    "SpikeFunction",
    "SpikingLIFCell",
    "STDPLinear",
    "MNNBlock",
    "SpikingMNNModel",
    "IntrinsicLearningScheduler",
    "register_surrogate",
    "get_surrogate",

    # neurogenesis
    "NeurogenesisPool",
    "check_apoptosis",
    "NeurogenesisScheduler",
]