# modulator.py — Genesis 框架全局调制场系统
# 实现 Modulator 基类、五种预置调制原以及全局调制场管理器 ModulatorSystem。
# 严格遵循白皮书六条核心公理，特别是公理一、四、五。

import math
from typing import Dict, List, Tuple, Any, Optional

# 导入核心模块中定义的中立调制原名称常量，以及全局状态键常量
from .genesis_core import (
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
)


class Modulator:
    """调制原基类，定义统一接口与空间扩散机制。

    每种调制原拥有名称、扩散半径、半衰期、效应曲线等属性。
    释放量由内部稳态偏差决定（公理五），并通过浓度场作用于注册目标。
    对应白皮书第四章。
    """

    def __init__(
            self,
            name: str,
            diffusion_radius: float = 1.0,
            half_life: float = 100.0,
            release_position: Tuple[float, ...] = (0.0, 0.0, 0.0),
    ):
        """
        参数:
            name: 调制原的唯一标识名（须与核心常量一致）。
            diffusion_radius: 高斯扩散核的标准差（空间范围）。
            half_life: 浓度源强度的半衰期（时间步数）。
            release_position: 释放核心在空间中的固定坐标。
        """
        if half_life <= 0:
            raise ValueError("半衰期必须大于 0")

        self.name = name
        self.diffusion_radius = diffusion_radius
        self.half_life = half_life
        self.release_position = release_position

        # 浓度源列表，每项为 (位置元组, 强度)
        self.sources: List[Tuple[Tuple[float, ...], float]] = []

    def compute_release(self, global_state: Dict[str, float]) -> float:
        """公理五：从全局内部稳态偏差中计算本次释放量。

        子类必须实现该方法，返回值应完全由 global_state 决定，
        不得引入任何外部任务信号或显式目标。
        """
        raise NotImplementedError

    def effect_curve(self, concentration: float) -> float:
        """效应曲线：默认恒等变换，子类可重写为非线性映射。"""
        return concentration

    def step(self, global_state: Dict[str, float]) -> None:
        """公理四、五：执行一步调制原更新。

        流程：计算基于内稳态的释放量 -> 在固定核心位置添加新源 ->
        所有源按半衰期指数衰减 -> 移除强度过低的源。
        """
        # 公理五：释放量完全由全局内部状态衍生
        release = self.compute_release(global_state)
        if release > 0:
            # 在核心位置添加新源
            self.sources.append((self.release_position, release))

        # 公理四：按半衰期指数衰减，不依赖外部任务信号
        # 衰减因子：每步衰减至 2^(-1/half_life)
        decay = 2.0 ** (-1.0 / self.half_life)
        surviving = []
        for pos, strength in self.sources:
            new_strength = strength * decay
            if new_strength > 1e-6:  # 强度过低则移除
                surviving.append((pos, new_strength))
        self.sources = surviving

    def query_concentration(self, position: Tuple[float, ...]) -> float:
        """基于高斯核的浓度场查询。

        参数:
            position: 查询位置坐标（元组，维度须与源位置一致）。
        返回:
            该位置处的总调制原浓度（所有源的叠加）。
        """
        sigma = self.diffusion_radius
        total = 0.0
        for src_pos, strength in self.sources:
            # 欧几里得距离
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, src_pos)))
            total += strength * math.exp(-(dist ** 2) / (2 * sigma ** 2))
        return total


# ----------------------------------------------------------------------------
# 五种预置调制原（公理五：释放量正比于对应内部稳态指标）
# ----------------------------------------------------------------------------

class RewardSignal(Modulator):
    """奖赏信号：释放量正比于全局 reward_error。"""

    def __init__(self, half_life: float = 50.0, diffusion_radius: float = 1.0):
        super().__init__(
            name=MODULATOR_REWARD,
            diffusion_radius=diffusion_radius,
            half_life=half_life,
        )

    def compute_release(self, global_state: Dict[str, float]) -> float:
        # 公理五：释放量 = 比例系数 × reward_error
        reward_error = global_state.get(GLOBAL_REWARD_ERROR, 0.0)
        return max(0.0, 0.1 * reward_error)


class SatietyFactor(Modulator):
    """满足度因子：释放量正比于全局 satiety。"""

    def __init__(self, half_life: float = 200.0, diffusion_radius: float = 1.0):
        super().__init__(
            name=MODULATOR_SATIETY,
            diffusion_radius=diffusion_radius,
            half_life=half_life,
        )

    def compute_release(self, global_state: Dict[str, float]) -> float:
        satiety = global_state.get(GLOBAL_SATIETY, 0.0)
        return max(0.0, 0.05 * satiety)


class NoveltyResponder(Modulator):
    """新奇响应子：释放量正比于全局 novelty。"""

    def __init__(self, half_life: float = 100.0, diffusion_radius: float = 1.0):
        super().__init__(
            name=MODULATOR_NOVELTY,
            diffusion_radius=diffusion_radius,
            half_life=half_life,
        )

    def compute_release(self, global_state: Dict[str, float]) -> float:
        novelty = global_state.get(GLOBAL_NOVELTY, 0.0)
        return max(0.0, 0.1 * novelty)


class StressActivator(Modulator):
    """应激激活子：释放量正比于全局 stress。"""

    def __init__(self, half_life: float = 80.0, diffusion_radius: float = 1.0):
        super().__init__(
            name=MODULATOR_STRESS,
            diffusion_radius=diffusion_radius,
            half_life=half_life,
        )

    def compute_release(self, global_state: Dict[str, float]) -> float:
        stress = global_state.get(GLOBAL_STRESS, 0.0)
        return max(0.0, 0.15 * stress)


class FatigueAccumulator(Modulator):
    """疲劳累积子：释放量正比于全局 energy_deficit。"""

    def __init__(self, half_life: float = 150.0, diffusion_radius: float = 1.0):
        super().__init__(
            name=MODULATOR_FATIGUE,
            diffusion_radius=diffusion_radius,
            half_life=half_life,
        )

    def compute_release(self, global_state: Dict[str, float]) -> float:
        energy_deficit = global_state.get(GLOBAL_ENERGY_DEFICIT, 0.0)
        return max(0.0, 0.1 * energy_deficit)


# ----------------------------------------------------------------------------
# 全局调制场管理器
# ----------------------------------------------------------------------------

class ModulatorSystem:
    """全局调制场管理器，负责调制原的注册、调度与目标施加。

    管理多种调制原，通过 `register_target` 将网络组件注册到空间位置，
    在每个时间步中更新所有调制原的源浓度场，并向每个目标分发局部调制效应。
    对应白皮书第四章及公理五。
    """

    def __init__(self):
        self.modulators: Dict[str, Modulator] = {}
        self.targets: List[Tuple[Any, Tuple[float, ...]]] = []  # (obj, position)

    def add_modulator(self, modulator: Modulator) -> None:
        """注册一个调制原。

        参数:
            modulator: 调制原实例，其名称必须唯一。
        """
        if modulator.name in self.modulators:
            raise ValueError(f"调制原 '{modulator.name}' 已存在，不能重复添加。")
        self.modulators[modulator.name] = modulator

    def register_target(
            self,
            obj: Any,
            position: Tuple[float, ...],
    ) -> None:
        """注册一个调制目标（神经元或突触）及其空间位置。

        参数:
            obj: 注册对象，必须实现 `apply_modulator(name, concentration)` 方法。
            position: 对象在空间中的坐标（元组，维度应与调制原释放位置一致）。
        """
        if not hasattr(obj, "apply_modulator"):
            raise TypeError("目标对象必须提供 apply_modulator(name, concentration) 方法")
        self.targets.append((obj, position))

    def register_targets(
            self,
            targets: List[Tuple[Any, Tuple[float, ...]]],
    ) -> None:
        """批量注册调制目标。

        参数:
            targets: [(obj, position), ...] 列表。
        """
        for obj, pos in targets:
            self.register_target(obj, pos)

    def step(self, global_state: Dict[str, float]) -> None:
        """执行一个时间步的全局调制场更新。

        流程：
        1. 公理五：所有调制原根据 global_state 更新源浓度场。
        2. 公理五：对每个注册目标，根据空间位置查询各调制原的局部浓度，
           并通过 `apply_modulator` 钩子施加效应（作用于内部状态 u/h/r）。

        参数:
            global_state: 网络全局内部状态指标字典，例如
                          {'reward_error': 0.2, 'stress': 0.5, ...}。
                          应由网络内部稳态监测模块提供，不依赖外部任务信号（公理四）。
        """
        # 1. 更新所有调制原的源（释放 + 衰减）
        for mod in self.modulators.values():
            mod.step(global_state)

        # 2. 对每个注册目标，查询局部调制原浓度并施加
        for obj, pos in self.targets:
            for name, mod in self.modulators.items():
                raw_conc = mod.query_concentration(pos)
                if raw_conc < 1e-6:
                    obj.apply_modulator(name, 0.0)
                else:
                    # 应用效应曲线（如默认恒等变换）
                    effective_conc = mod.effect_curve(raw_conc)
                    # 公理五：调制原通过标准钩子作用于目标内部状态
                    obj.apply_modulator(name, effective_conc)


def create_default_modulators() -> ModulatorSystem:
    """便捷工厂函数：一键创建包含五种预置调制原的 ModulatorSystem。

    调制原半衰期和扩散半径均采用白皮书建议的默认值，
    用户可根据需要后续修改。

    返回:
        ModulatorSystem 实例，已注册五种预置调制原（奖赏信号、满足度因子、
        新奇响应子、应激激活子、疲劳累积子）。
    """
    system = ModulatorSystem()
    system.add_modulator(RewardSignal())
    system.add_modulator(SatietyFactor())
    system.add_modulator(NoveltyResponder())
    system.add_modulator(StressActivator())
    system.add_modulator(FatigueAccumulator())
    return system