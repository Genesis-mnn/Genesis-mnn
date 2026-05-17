# test_modulator.py
import unittest
from genesis_core.modulator import (
    Modulator, RewardSignal, SatietyFactor, NoveltyResponder,
    StressActivator, FatigueAccumulator, ModulatorSystem, create_default_modulators
)
from genesis_core import (
    GLOBAL_REWARD_ERROR, GLOBAL_SATIETY, GLOBAL_NOVELTY,
    GLOBAL_STRESS, GLOBAL_ENERGY_DEFICIT, MODULATOR_REWARD
)

class MockTarget:
    """模拟调制目标，记录接收到的调制原浓度。"""
    def __init__(self):
        self.received = {}
    def apply_modulator(self, name, concentration):
        self.received[name] = concentration

class TestModulator(unittest.TestCase):
    def setUp(self):
        self.system = create_default_modulators()
        self.global_state = {
            GLOBAL_REWARD_ERROR: 10.0,
            GLOBAL_SATIETY: 0.8,
            GLOBAL_NOVELTY: 0.5,
            GLOBAL_STRESS: 0.3,
            GLOBAL_ENERGY_DEFICIT: 0.2,
        }

    def test_release_computation(self):
        """测试五种预置调制原的释放量计算。"""
        mod = self.system.modulators[MODULATOR_REWARD]
        release = mod.compute_release(self.global_state)
        self.assertAlmostEqual(release, 0.1 * 10.0)

        # 其他调制原可类似验证，此处略
        for mod_name, expected in [
            ("满足度因子", 0.05 * 0.8),
            ("新奇响应子", 0.1 * 0.5),
            ("应激激活子", 0.15 * 0.3),
            ("疲劳累积子", 0.1 * 0.2),
        ]:
            mod = self.system.modulators[mod_name]
            self.assertAlmostEqual(mod.compute_release(self.global_state), expected)

    def test_modulator_step_and_diffusion(self):
        """测试调制原的步进与浓度查询。"""
        mod = self.system.modulators[MODULATOR_REWARD]
        mod.step(self.global_state)  # 产生源并衰减
        self.assertTrue(len(mod.sources) > 0, "Step should add a source")

        # 源位置浓度大于零
        conc = mod.query_concentration(mod.release_position)
        self.assertGreater(conc, 0)

        # 远处位置浓度应较小
        far_pos = (10.0, 10.0, 10.0)
        far_conc = mod.query_concentration(far_pos)
        self.assertLess(far_conc, conc)

    def test_modulator_system_registration_and_step(self):
        """测试 ModulatorSystem 的注册目标和 step() 方法。"""
        target = MockTarget()
        pos = (1.0, 0.0, 0.0)
        self.system.register_target(target, pos)
        self.system.step(self.global_state)

        # 目标应收到调制原
        self.assertTrue(len(target.received) > 0)
        self.assertGreater(target.received.get(MODULATOR_REWARD, 0), 0)

    def test_empty_global_state(self):
        """测试无全局状态时调制原不应释放。"""
        empty_state = {k: 0.0 for k in self.global_state}
        mod = self.system.modulators[MODULATOR_REWARD]
        release = mod.compute_release(empty_state)
        self.assertEqual(release, 0.0)


if __name__ == '__main__':
    unittest.main()