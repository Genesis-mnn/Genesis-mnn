# test_quantize.py
import unittest
import torch
import torch.nn as nn
from genesis_core.quantize import (
    QuantConfig,
    ThresholdCenteredQuantize,
    FakeQuantize,
    quantize_model_ptq,
    prepare_qat,
    convert_qat,
)
from genesis_core.neuron import LIFNeuron


# 一个用于测试 QAT 的简单模块（包含 LIFNeuron）
class NeuronBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.neuron = LIFNeuron(num_neurons=10)

    def forward(self, x):
        return self.neuron(x)


class TestQuantize(unittest.TestCase):
    def setUp(self):
        self.config = QuantConfig(
            default_weight_dtype='int8',
            default_activation_dtype='int8',
            symmetric=True,
            threshold_centered=False,
            weight_rescale=False,
        )

    def test_threshold_centered_quantize(self):
        """测试阈值中心化伪量化模块。"""
        tcq = ThresholdCenteredQuantize(Vth=1.0, k=5.0)
        u = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
        u_q = tcq(u)
        self.assertEqual(u_q.shape, u.shape)
        self.assertFalse(torch.isnan(u_q).any(), "Output contains NaN")

    def test_fake_quantize_module(self):
        """测试标准伪量化模块。"""
        fq = FakeQuantize(qmin=-127, qmax=127, symmetric=True)
        x = torch.randn(3, 4)
        y = fq(x)
        self.assertEqual(y.shape, x.shape)

    def test_qat_prepare_and_convert(self):
        """测试 QAT 管线的 prepare 和 convert。"""
        model = NeuronBlock()

        # 构建 nrn_params 字典，确保所有参数值都与 QuantMorphoNeuron 构造器兼容
        # 从 LIFNeuron 中提取阈值和时间常数等参数，仅当标量时才调用 .item()
        neuron = model.neuron
        nrn_params = {}
        # 尝试获取常见的神经元参数
        for attr_name in ['v_threshold', 'threshold', 'tau', 'v_th']:
            if hasattr(neuron, attr_name):
                val = getattr(neuron, attr_name)
                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        nrn_params[attr_name] = val.item()
                    else:
                        # 保留多元素张量原值
                        nrn_params[attr_name] = val
                else:
                    nrn_params[attr_name] = val
        # 若没有找到，则从 state_dict 中提取可能的标量
        if not nrn_params:
            state = neuron.state_dict()
            for key in state:
                if state[key].numel() == 1:
                    nrn_params[key] = state[key].item()
        # 将提取的标量参数写回神经元属性（使后续管线可以读取到正确类型）
        for k, v in nrn_params.items():
            if hasattr(neuron, k):
                setattr(neuron, k, v)

        config = QuantConfig(threshold_centered=False)
        qat_model = prepare_qat(model, config)
        self.assertIsNotNone(qat_model)

        # 前向传播（训练模式，伪量化生效）
        inp = torch.randn(1, 1, 10) * 0.1
        out = qat_model(inp)
        self.assertEqual(out.shape, (1, 10))

        # 转换为推理模型
        qmodel = convert_qat(qat_model)
        eval_out = qmodel(inp)
        self.assertEqual(eval_out.shape, (1, 10))

    def test_ptq_pipeline_basic(self):
        """测试 PTQ 管线基本流程（无校准数据）。"""
        # 手动创建两个结构相同的简单 LIFNeuron，使用相同的初始化参数
        neuron1 = LIFNeuron(num_neurons=10)
        neuron2 = LIFNeuron(num_neurons=10)
        # 复制参数，使两者初始完全一致（不使用 deepcopy）
        neuron2.load_state_dict(neuron1.state_dict())

        config = QuantConfig(threshold_centered=False, weight_rescale=False)
        # 对其中一个执行 PTQ 量化
        q_neuron = quantize_model_ptq(neuron1, config, calib_data=None)
        self.assertIsNotNone(q_neuron)

        # 验证量化后的模型可以进行推理
        inp = torch.randn(1, 1, 10)
        out = q_neuron(inp)
        self.assertEqual(out.shape, (1, 10))

        # 比较两者的权重差异来验证量化效果
        orig_state = neuron2.state_dict()
        q_state = q_neuron.state_dict()

        weights_differ = False
        for key in orig_state:
            if key in q_state:
                if not torch.equal(orig_state[key], q_state[key]):
                    weights_differ = True
                    break
        # 如果键集合不同，也说明量化改变了模型结构
        if not weights_differ and set(orig_state.keys()) != set(q_state.keys()):
            weights_differ = True
        self.assertTrue(weights_differ, "PTQ should modify weights or model structure")


if __name__ == '__main__':
    unittest.main()