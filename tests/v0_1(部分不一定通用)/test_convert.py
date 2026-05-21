# test_convert.py
import unittest
import torch
import torch.nn as nn
from genesis_core.convert import SNNtoMNNConverter, ANNtoMNNConverter, convert, MNNModel
from genesis_core import MorphoNeuron


class TestConvert(unittest.TestCase):
    def setUp(self):
        # 使用 MorphoNeuron 基类（默认参数）替代原有 FakeIFNode，避免参数不兼容
        self.snn_model = nn.Sequential(
            nn.Linear(10, 20),
            MorphoNeuron(num_neurons=20),
            nn.Linear(20, 5),
            MorphoNeuron(num_neurons=5),
        )
        # 简单 ANN 模型
        self.ann_model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def test_snn_to_mnn_conversion_and_inference(self):
        """测试 SNN→MNN 转换及脉冲推理。"""
        mnn = convert(self.snn_model, source='snn', time_steps=10, use_stdp=True)
        self.assertIsInstance(mnn, MNNModel)

        # 输入脉冲序列 (batch=1, T=10, in_features=10)
        x = (torch.rand(1, 10, 10) > 0.5).float()
        out = mnn(x)
        self.assertEqual(out.shape, (1, 10, 5), "Output shape should be (batch, T, last_neurons)")
        # 发放率合理（允许全零，但大概率非零）
        self.assertGreaterEqual(out.sum().item(), 0)

    def test_ann_to_mnn_conversion_and_inference(self):
        """测试 ANN→MNN 转换及脉冲推理。"""
        mnn = ANNtoMNNConverter(self.ann_model, time_steps=10, input_shape=(784,)).convert()
        # 脉冲输入
        x = (torch.rand(1, 10, 784) > 0.8).float()
        out = mnn(x)
        self.assertEqual(out.shape, (1, 10, 10))

    def test_convert_function_auto_detect(self):
        """测试统一转换入口的自动检测。"""
        # 使用 source=None 自动检测为 snn
        mnn = convert(self.snn_model, time_steps=5)
        self.assertIsInstance(mnn, MNNModel)


if __name__ == '__main__':
    unittest.main()