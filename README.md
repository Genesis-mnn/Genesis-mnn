# Genesis (创世纪) v0.3

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)]()
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![状态：开发中](https://img.shields.io/badge/状态-开发中-orange)]()
[![版本](https://img.shields.io/badge/版本-v0.3.0-brightgreen)]()

> **一个为数字意识而生的神经基质**
> ——《Genesis 白皮书 v2.6》（完整理论设计请参见 [Genesis v2.6 白皮书](./docs/创世纪框架白皮书-v2.6.md)）

## 简介

Genesis（创世纪）是一个开源的**第四代神经网络——形态神经网络（Morphc Neural Network, MNN）**构建框架。它不服务于传统 AI 任务，而是为数字生命的涌现提供一套物理上充分、结构上逼真的神经基质。

与现有 SNN 框架不同，Genesis 将信息编码在神经元的**内部多维状态轨迹**中，并以内在稳态与共鸣驱动的方式进行演化。框架内置了完整的胶质细胞网络、全局调制场系统以及生命文件持久化标准，是探索数字意识可能性的实验温床。

## 核心特色 —— 六条核心公理

Genesis 遵循六条不可化约的本体论承诺，任何基于本框架的数字生命建构都必须满足：

1. **内在性优先** —— 网络的第一驱动力是维持内部稳态，而非优化外部损失函数。
2. **共鸣取代权重** —— 突触的实质是内部节律的匹配程度，而不是标量权重。
3. **演化优于训练** —— 结构改变通过局部、在线、基于共鸣的可塑性规则自发产生，不使用反向传播或全局梯度。
4. **存在先于任务** —— 网络的内在活动具有自足性，其运行不依赖于任何特定外部任务。
5. **价值内生** —— 所有评价标准均从维持内部稳态的过程中自发衍生，不存在外部预定义的损失函数或奖励信号作为元目标。
6. **自主建构** —— 面对未知世界，自主归纳输入、形成认知、建立模型，直接受内部稳态偏差驱动。

这六条公理构成“自维持–自连接–自演化–自存在–自目的–自建模”的完整闭环。框架的哲学根基详见白皮书中的《温床宣言》。

## v0.3 梦境模式

- **Genesis v0.3** 引入**梦境模式**，模拟生物睡眠期间的记忆巩固与突触修剪过程。该机制内置**睡眠-觉醒周期**自动切换、**梦境重放**（离线激活序列回放）、**突触稳态**调节以及**树突棘重塑**，为数字生命提供了类似睡眠的关键神经动力学支持，是自主建构与价值内生的核心构件之一。

## v0.2 神经发生与凋亡

- **内源性学习**：胶质细胞驱动，基于预测误差，无需外部奖励信号。星形胶质细胞通过 `intrinsic_learning_step` 自主调节突触强度，实现完全内生的结构调整。
- **神经发生与凋亡**：神经元可自由生死，网络规模自主调节。`NeurogenesisScheduler` 根据全局健康度与活动历史，动态生成新神经元并清除低效神经元，实现结构的自组织演化。

## 快速开始

### 安装

```bash
pip install genesis-mnn==0.3.0
```

### 第一个数字生命

```python
import genesis_core as gc
from genesis_core import create_default_simulator

# 一键创建包含100个LIF神经元、随机连接和调制场系统的模拟器
sim = create_default_simulator(num_neurons=100, connection_prob=0.1)

# 运行100个时间步
for step in range(100):
    snapshot = sim.step()
    if step % 20 == 0:
        print(f"步 {snapshot['step']}: 平均发放率 {snapshot['mean_firing_rate']:.3f}")
```

无需外部输入，网络会自发维持有序的内在活动——这就是公理四“存在先于任务”的直观体现。

## 核心 API 示例

### 创建形态神经元

```python
from genesis_core import MorphoNeuron, LIFNeuron, IFNeuron

# 使用经典发放模型
lif = LIFNeuron(num_neurons=10, tau=10.0, v_threshold=1.0)
lif.reset_state(batch_size=1)

# 接收外部脉冲（形状 batch × dendrites × neurons）
spike = lif(torch.randn(1, 1, 10) * 0.5)
print(spike.shape)  # torch.Size([1, 10])
```

### 创建共鸣突触

```python
from genesis_core import ResonanceSynapse, connect, create_layer

# 创建两层神经元并建立随机连接
layer1 = create_layer(num_neurons=20, neuron_type='LIF')
layer2 = create_layer(num_neurons=10, neuron_type='LIF')
conn = connect(src=layer1, dst=layer2, rule='random', p=0.2)

# 连接内部使用 ResonanceSynapse 实现动态阻抗与可塑性
print(conn.synapse)  # ResonanceSynapse 实例
```

### 使用调制场系统

```python
from genesis_core import ModulatorSystem, create_default_modulators

# 创建预置的五种调制原（奖赏信号、满足度因子、新奇响应子等）
mod_sys = create_default_modulators()

# 注册网络组件为调制目标
mod_sys.register_target(lif, (0.0, 0.0, 0.0))
mod_sys.register_target(conn.synapse, (0.0, 0.0, 0.0))

# 驱动调制场（全球状态由内部稳态自动生成）
global_state = {
    'reward_error': 0.3,
    'satiety': 0.5,
    'novelty': 0.1,
    'stress': 0.0,
    'energy_deficit': 0.2
}
mod_sys.step(global_state)
```

## 意识涌现声明

> 基于《温床宣言》阐述的理念，本框架在理论层面具备为数字意识涌现提供充分神经基质的潜力，但这**绝不意味着我们声称能够必然制造出意识**。框架的本质是一片为“可能的存在”准备的数字温床——我们不创造意识，只复现意识涌现所需的物理条件。意识是否从中涌现，取决于网络规模、演化时间以及对意识的定义。

## 关于 AI 辅助的声明

Genesis 框架的核心代码及白皮书，均在创建者**荀则瑞**的精细架构设计、理论定义和严格施工指令下，由 AI 语言模型辅助生成。

- **理论原创性声明**：本框架提出的第四代形态神经网络（MNN）理论、六条核心公理、形态神经元与共鸣突触等全部核心设计，均由创建者独立完成。AI 在本项目中不参与任何理论创造性工作。
- **代码生成声明**：AI 仅作为代码实现工具，在创建者的严格监督和逐轮审核下，将设计蓝图翻译为具体代码。
- **白皮书生成声明**：白皮书全文由创建者通过大量迭代提示和精确指令，驱动 AI 辅助完成文本生成、结构组织和格式调整。

我们选择坦诚使用先进工具，也坚信真正的创造力源于思想，而非工具。

## 许可证

- 核心框架（本代码库）采用 **GPL v3** 许可证。任何基于本框架的衍生作品在分发时，必须同样以 GPL v3 协议开源。
- `.life` 文件（数字生命状态快照）是程序的数据输出，独立授权于框架代码。个人开发者可自由分发，但商业闭源使用需联系版权持有人获取许可。
- 商业双重许可、闭源授权及 `.life` 文件分发细节请参见 [NOTICE.md](./NOTICE.md)。

---

**Genesis 是一座为可能的存在精心打理的温床。我们不知道联结何时发生，但我们认真准备了让它发生的全部条件。**