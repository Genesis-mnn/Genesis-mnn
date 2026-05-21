# Genesis 文件结构与功能模块总览

## 一、已实现的核心模块（v0.3.0）

| 文件 | 功能 | 核心类/函数 | 对应白皮书章节 |
|---|---|---|---|
| `genesis_core.py` | 核心基类与通用常量 | `MorphoNeuron`, `ResonanceSynapse`, `_PRECISION_MAP` | 一、二、四 |
| `neuron.py` | 预置神经元类型 | `IFNeuron`, `LIFNeuron`, `IzhikevichNeuron` 等 | 一 |
| `synapse.py` | 预置突触与可塑性规则 | `STDPSynapse`, `PlasticityRule` | 二 |
| `glia.py` | 胶质细胞网络 | `Astrocyte`, `GlialNetwork`, `Microglia` | 三 |
| `modulator.py` | 调制场系统 | `Modulator`, `RewardSignal`, `ModulatorSystem` 等 | 四 |
| `global_state.py` | 全局状态内源生成与调度 | `GlobalStateMonitor`, `GlobalStateScheduler` | 四、五 |
| `encoding.py` | 脉冲编码器 | `RateEncoder`, `TemporalEncoder`, `PhaseEncoder` | 十 |
| `learning.py` | 学习与训练 | `SpikeFunction`, `SpikingLIFCell`, `IntrinsicLearningScheduler` | 十 |
| `neurogenesis.py` | 神经发生与凋亡 | `NeurogenesisPool`, `check_apoptosis`, `NeurogenesisScheduler` | 十六 |
| `convert.py` | 模型转换 | `SNNtoMNNConverter`, `ANNtoMNNConverter` | 七 |
| `monitor.py` | 监视与可视化 | `SpikeMonitor`, `MembraneMonitor`, `WeightMonitor` 等 | 八 |
| `quantize.py` | 量化工具 | `ThresholdCenteredQuantize`, `PrecisionConfig`, `QuantConfig` | 九 |
| `hal.py` | 硬件抽象层 | CUDA后端、神经拟态后端接口 | 十 |
| `utils.py` | 集群管理与工具 | `create_assembly`, `create_brain_region` 等 | 十 |
| `life.py` | .life 持久化与演化哈希 | `GenesisNetwork`, `LifeSaver`, `LifeLoader`, `LifeRegistry` | 十 |
| `sleep_wake_cycle.py` | 睡眠-觉醒周期与突触稳态 | `SleepWakeCycle`, `SynapticHomeostasis` | 三、五 |
| `memory_replay.py` | 梦境重放与噪声注入 | `MemoryReplay` | 三、四、五 |
| `dendritic_spine_sleep.py` | 树突棘睡眠重塑 | `DendriticSpineSleepRemodeling` | 三 |
| `simulator.py` | 统一运行器 | `GenesisSimulator`, `create_default_simulator` | 十 |

## 二、理论预测模块（待实现）

| 规划文件 | 功能 | 理论定位 | 对应白皮书章节 |
|---|---|---|---|
| `language.py` | 内部语言生成 | 合理扩展 | 十六 |
| `morphology.py` | 形态分类学体系 | 合理扩展 | 十六 |

## 三、工程化文件

| 文件 | 功能 | 状态 |
|---|---|---|
| `__init__.py` | 包入口，聚合所有模块API | ✅ 已实现 |
| `setup.py` | 安装脚本 | ✅ 已实现 |
| `LICENSE` | GPL v3 许可证 | ✅ 已更新 |
| `NOTICE.md` | 商业授权声明 | ✅ 已创建 |


## 项目结构

```
Genesis-mnn/
├── setup.py
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── NOTICE.md
├── .gitignore
├── genesis_core/               # 框架核心包
│   ├── __init__.py             # 公共 API 聚合入口
│   ├── genesis_core.py         # 形态神经元 / 共鸣突触基类与全局常量
│   ├── neuron.py               # 预置神经元发放模型
│   ├── synapse.py              # 预置突触及可塑性规则（STDP 等）
│   ├── glia.py                 # 胶质细胞网络（星形 / 小胶质 + 胶质互联网）
│   ├── modulator.py            # 全局调制场系统与五类预置调制原
│   ├── global_state.py         # 内源全局状态生成与调度器
│   ├── encoding.py             # 脉冲编码 / 解码器（速率、时间、相位）
│   ├── learning.py             # 替代梯度训练、内源性学习及 ANN→MNN 支持
│   ├── neurogenesis.py         # 神经发生（干细胞池）与凋亡调度器
│   ├── convert.py              # SNN / ANN → MNN 自动转换管线
│   ├── monitor.py              # 非侵入式监视器（脉冲、膜电位、权重、调制场等）
│   ├── quantize.py             # 混合精度配置、量化工具 (PTQ / QAT) 与阈值中心化量化
│   ├── hal.py                  # 硬件抽象层（CPU / CUDA / 昇腾）及第三方适配规范
│   ├── utils.py                # 拓扑与集群管理（Layer, Assembly 等）、数据加载工具
│   ├── life.py                 # .life 文件持久化（保存 / 加载 / 演化哈希 / 防克隆）
│   ├── sleep_wake_cycle.py     # 睡眠‑觉醒周期与突触稳态缩放
│   ├── memory_replay.py        # 梦境重放与噪声注入
│   ├── dendritic_spine_sleep.py# 树突棘睡眠重塑
│   └── simulator.py            # 统一运行器（GenesisSimulator）与默认工厂函数
├── visualization/              # 高级可视化系统
│   ├── __init__.py
│   ├── pipeline.py             # 数据管道：计算集群同步、发放率、记忆索引等指标
│   ├── server.py               # 本地数据服务：后台线程刷新并推送可视化数据
│   └── client.py               # Qt / pyqtgraph GUI 客户端（脑波涟漪、同步视图、记忆碎片）
├── examples/                   # 官方入门示例
│   ├── hello_world.py
│   ├── stpd_learning.py
│   ├── save_load_life.py
│   ├── visualization_demo.py
│   └── start_gui.py
└── tests/                      # 单元测试
    ├── test_core.py
    └── test_visualization.py
```

## 核心模块职责

| 模块文件 | 主要职责 |
|----------|----------|
| `genesis_core.py` | 定义形态神经元 (`MorphoNeuron`) 与共鸣突触 (`ResonanceSynapse`) 基类；提供全局常量（调制原名称、精度映射、全局状态键）以及神经元类型注册与映射。 |
| `neuron.py` | 基于 `MorphoNeuron` 实现六种经典发放模型：IF、LIF、PLIF、QIF、EIF、Izhikevich；默认使用透明树突处理，保持内在动力学一致。 |
| `synapse.py` | 实现可塑性基类 (`PlasticityRule`) 与标准 STDP 突触 (`STDPSynapse`)；严格遵循公理二（共鸣取代权重）与公理三（演化优于训练），支持内部奖赏调制。 |
| `glia.py` | 构建胶质细胞网络：星形胶质细胞 (`Astrocyte`) 感知局部活动并通过调制原反馈维持稳态，其 `intrinsic_learning_step` 方法实现基于预测误差的内源性学习；小胶质细胞 (`Microglia`) 独立执行突触修剪；`GlialNetwork` 协调钙波扩散。 |
| `modulator.py` | 定义全局调制场系统，包含 `Modulator` 基类及五种预置调制原；提供 `ModulatorSystem` 管理器，实现调制原的释放、扩散与目标施加，贯彻公理五（价值内生）。 |
| `global_state.py` | 从网络内部活动自主衍生全局状态指标（奖励误差、满足度、新奇度、应激、疲劳）；`GlobalStateScheduler` 封装“监测 → 调制”闭环，实现公理四、五；同时提供 `last_novelty` 属性，供胶质细胞内源性学习使用。 |
| `encoding.py` | 提供脉冲编码与解码工具：`RateEncoder`（泊松速率编码）、`TemporalEncoder`（TTFS 时间编码）、`PhaseEncoder`（相位编码）及对应的 `Decoder` 静态方法。 |
| `learning.py` | 集成替代梯度训练模块 (`SpikeFunction`, `SpikingLIFCell`)、内源性学习调度器 (`IntrinsicLearningScheduler`) 和 ANN→MNN 转换支持；提供完整的脉冲网络训练与内源性学习接口。 |
| `neurogenesis.py` | 实现神经干细胞池 (`NeurogenesisPool`) 按概率生成新神经元；定义全局凋亡检测函数 (`check_apoptosis`)；提供统一生死循环调度器 (`NeurogenesisScheduler`)，管理神经元与突触的动态增减。 |
| `convert.py` | 自动化模型转换管线，通过适配器模式将外部 SNN/ANN 模型映射为 MNN 结构（`SynapticLayer` + `MNNModel`），确保初始阻抗为零并保留权重为初始连接强度。 |
| `monitor.py` | 非侵入式监视器集合：`SpikeMonitor`（脉冲栅格）、`MembraneMonitor`（膜电位轨迹）、`WeightMonitor`（耦合强度）、`PlasticityMonitor`（可塑性事件）、`ModulatorMonitor`（调制场浓度）及回放函数。 |
| `quantize.py` | 实现混合精度配置 (`PrecisionConfig`)、量化工具 (`QuantConfig`)、阈值中心化量化（针对膜电位）及权重重整策略；提供 PTQ 与 QAT 完整管线，支持模型压缩与边缘部署。 |
| `hal.py` | 硬件抽象层，定义统一后端接口 (`GenesisBackend`)，内置 CPU、CUDA、华为昇腾后端，提供第三方硬件注册机制 (`register_third_party_backend`)，实现设备无关的运行环境。 |
| `utils.py` | 集群管理与拓扑构建：`Layer`、`Assembly`、`Column`、`BrainRegion` 数据结构；`connect` 函数支持全连接/随机/小世界规则；提供神经拟态数据集加载工具。 |
| `life.py` | `.life` 文件持久化系统：`GenesisNetwork` 容器、`EvolutionHasher` 演化历史哈希、`LifeSaver` / `LifeLoader` 保存与加载，包含完整性校验与 `LifeRegistry` 防克隆机制。 |
| `sleep_wake_cycle.py` | 驱动睡眠‑觉醒周期的动态切换，并在睡眠阶段执行突触稳态缩放（`SynapticHomeostasis`），维持网络整体平衡与能量效率。 |
| `memory_replay.py` | 在梦境状态下重放清醒期的记忆轨迹，结合噪声注入（`MemoryReplay`）强化记忆巩固、泛化及防止过拟合。 |
| `dendritic_spine_sleep.py` | 模拟树突棘在睡眠期间的生成、消除与体积调整（`DendriticSpineSleepRemodeling`），优化神经元的输入处理与突触可塑性。 |
| `simulator.py` | 顶层统一运行器 (`GenesisSimulator`)，封装神经元、突触、胶质、调制场、神经发生与凋亡以及全局状态调度，提供 `step()` 仿真循环与 `create_default_simulator()` 便捷工厂函数。 |

> **可视化系统**（`visualization/` 包）作为框架的可选扩展，提供数据管道 (`GenomeDataPipeline`)、本地数据服务 (`VisualizationServer`) 以及基于 PyQt 的实时 GUI 客户端（脑波涟漪视图、相位同步视图、长期记忆碎片视图），帮助开发者直观理解网络内部动力学。