# methods

本目录实现 CleanCore 论文第五章使用的方法。所有方法共享 `common/model.py` 中的 MLP、`common/training.py` 中的训练循环和 `configs/default.yaml` 中的训练参数。方法文件只实现论文特有的数据清洗、样本打分、子集选择、标签处理、特征处理或权重处理逻辑。

## 方法类型

| 方法 | 类型 | 主要论文依据 | 实现产物 |
| --- | --- | --- | --- |
| `cleanlab` | 训练辅助的数据处理型 | Northcutt et al., JAIR 2021 | OOF 概率驱动的疑似错标样本删除 |
| `misdetect` | 训练辅助的数据处理型 | Deng et al., PVLDB 2024 | early-loss 迭代删除 |
| `ctrl` | 训练辅助的数据处理型 | Yue & Jha, IEEE TAI 2024 | 训练损失曲线聚类删除 |
| `herding` | 数据处理型 | Welling, ICML 2009 | 类内 mean-matching 子集和权重 |
| `deepfool` | 训练辅助的数据处理型 | Moosavi-Dezfooli et al., CVPR 2016 | 边界扰动半径驱动的子集 |
| `gradmatch` | 训练耦合型 | Killamsetty et al., ICML 2021 | 连续训练中周期性更新梯度匹配子集和权重 |
| `coretab` | 训练辅助的数据处理型 | Hadar et al., PVLDB 2024 | GBDT datamap 驱动的表格核心集 |
| `goodcore` | 数据处理型 | Chai et al., PACMMOD 2023 | 缺失空间期望距离核心集和权重 |
| `deem` | 训练耦合型 | Deng et al., PACMMOD 2025 | 连续训练中动态更新软标签和梯度匹配子集 |
| `cleancore` | 训练耦合型 | 论文第三章、第四章 | 错误识别、标签/特征处理、可靠性加权核心集 |

## 统一接口

每个方法暴露 `run(ctx: MethodContext) -> MethodOutput`。`MethodOutput` 可以包含：

- `selected_indices`：最终训练使用的样本索引。
- `sample_weights`：与 `selected_indices` 对齐的样本权重。
- `corrected_labels`：全量训练集上的修正标签。
- `corrected_features`：全量训练集上的修复特征。
- `soft_targets`：全量训练集上的软标签。
- `predicted_noisy_mask`：方法识别出的疑似噪声样本。
- `final_predictions`：训练耦合型方法内部最终模型在测试集上的预测。

数据处理型和训练辅助数据处理型方法的最终 MLP 训练、测试评估、时间统计和结果保存由 `run.py` 统一完成。训练耦合型方法在方法内部持续训练同一个最终 MLP，并通过 `final_predictions` 交给 `run.py` 统一计算指标和保存结果。
