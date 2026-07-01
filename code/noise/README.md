# noise

噪声注入模块放在这里。当前用于从 `data/processed/npz_clean/` 生成固定的带噪训练集。

- `injector.py`：通用工具，包括错误类型分配、特征值域计算、特征选择和结果结构。
- `random_noise.py`：随机噪声。
- `shift_noise.py`：偏移噪声。

## 当前噪声设置

默认噪声比例为 `0.20`，即训练集中 20% 的样本被污染。被污染样本中的错误类型比例为：

- label-only：2
- feature-only：2
- mixed：1

也就是在所有带噪样本中，40% 只改标签，40% 只改特征，20% 同时改标签和特征。

特征错误最多污染单个样本 30% 的特征，且至少污染 1 个可变特征。实际被改过的特征位置记录在 `corrupted_feature_mask` 中。

## 合法性约束

标签错误必须仍在原数据集的标签集合内，并且不能等于原标签。

特征错误必须仍在该特征的干净训练集值域内，即 `[min_j, max_j]`。二值特征使用 0/1 翻转。所有被 `corrupted_feature_mask` 标记的特征都会实际发生变化。

## 噪声类型

`random`：

- 标签：随机替换为另一个合法类别。
- 特征：在该特征干净训练值域内随机替换。

`shift`：

- 标签：替换为 `(label + 1) % n_classes`。
- 特征：按特征标准差做方向性平移，并限制在干净训练值域内；如果边界裁剪会导致数值不变，则改为最近的合法不同值。

## 输出位置

带噪数据由 `scripts/make_noisy_npz.py` 生成到：

```text
data/processed/npz_noisy/<noise_type>/rate_0.20/<dataset>/
```

其中 `train.npz` 和 `test.npz` 只包含 `X/y`。噪声真值单独保存在 `noise_info.npz`，仅用于评价和检查，不作为方法训练输入。
