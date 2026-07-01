# common

所有方法共享的基础模块放在这里。这里不写任何具体 baseline 或 CleanCore 的算法逻辑，只提供统一的数据、模型、训练、评价、计时和结果保存能力。

## 文件说明

- `paths.py`：统一管理代码、数据、配置和结果路径。
- `config.py`：读取和合并 YAML 配置。
- `seed.py`：统一设置 `random`、`numpy` 和可选 `torch` 随机种子。
- `data.py`：读取干净或带噪 `.npz` 数据。正式训练默认只读 `train.npz` / `test.npz` 中的 `X/y`。
- `preprocessing.py`：统一 `StandardScaler`，只用训练集计算均值和标准差，再作用到训练集和测试集。
- `model.py`：统一下游 MLP。
- `training.py`：统一 PyTorch 训练循环、可选 train-loss early stopping、`last_model.pt` 和可选 `best_model.pt` 保存。
- `metrics.py`：统一分类指标，包括 accuracy、macro precision、macro recall、macro F1。
- `timer.py`：统一计时，后续 `run.py` 会用它记录 `total_time_sec`。
- `results.py`：统一保存 JSON、YAML 配置和训练历史 CSV。
- `interfaces.py`：统一方法输出格式，支持 selected indices、sample weights、corrected labels 和 predicted noisy mask。

## 当前默认设置

模型：

```text
MLP: input -> 256 -> 128 -> num_classes
activation: ReLU
dropout: 0.2
batch norm: enabled
```

训练：

```text
device: cpu
optimizer: AdamW
learning rate: 1e-3
weight decay: 1e-4
batch size in default.yaml: 256
max epochs in default.yaml: 10000
early stopping: enabled by default
```

Early stopping 采用：

```text
monitor: train_loss
patience: 50
min_delta: 1e-4
```

模型保存：

```text
last_model.pt: 默认保存，表示训练结束最后一轮模型
best_model.pt: 可选保存，按最低 train_loss 判断
```

注意：正式实验需要在已安装 PyTorch、scikit-learn、NumPy 和 PyYAML 的环境中运行。
