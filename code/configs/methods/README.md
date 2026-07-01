# 方法配置

本目录存放方法专属参数。共享模型结构、优化器、batch size、epoch 上限、early stopping、预处理和指标配置放在 `../default.yaml`，所有方法共用。

当前方法配置：

- `defaults/<method>.yaml`：方法全局默认参数。
- `lab1/defaults/<method>.yaml`：实验一默认方法参数。
- `lab1/<dataset>/<method>.yaml`：实验一中某个数据集的覆盖参数。
- `lab1_official/defaults/<method>.yaml`：基线方法的公开实现默认值或论文参数，尽量不做数据集专属方法降参。

正式运行时，`run.py` 会按从通用到具体的顺序合并配置：

```text
configs/default.yaml
configs/methods/defaults/<method>.yaml
configs/methods/<method_config_group>/defaults/<method>.yaml
configs/methods/<method_config_group>/<dataset>/<method>.yaml
--method_config 指定的额外 YAML
命令行覆盖参数
```

实验一使用：

```text
configs/methods/lab1/defaults/<method>.yaml
configs/methods/lab1/<dataset>/<method>.yaml
```

其中数据集专属配置只在需要调整自然方法参数或统一子集比例时设置。方法流程仍保持论文对应流程，不在实验配置中使用额外候选池上限、打分样本上限、reference/eval 上限或固定 `subset_size`。

实验一的 subset selection / hybrid 方法统一使用以下目标比例：

```text
wdbc, banknote, pendigits, magic: 0.10
sensorless, miniboone, skin, covertype: 0.05
susy: 0.01
```

数据清洗类方法不强行统一删除数量，由各自论文流程判断疑似噪声样本。

`gradmatch`、`deem` 和 `cleancore` 属于训练耦合型方法，`warmup_epochs` / `pretrain_epochs` 控制第一次方法更新前的预训练轮数，`selection_interval` / `stage_update_interval` 控制后续方法更新间隔。它们仍然使用 `configs/default.yaml` 中统一的 MLP 架构和公共训练参数。

`cleancore` 的方法参数覆盖论文第 3、4 章中的稳定软标签处理、受控迭代特征修复、可靠性加权梯度覆盖和动态核心集更新。数据集专属配置可以调节触发阈值、修复步数或每阶段修复预算，但不应通过额外样本上限跳过这些机制。

`lab1_official` 用于重新运行基线方法：共享模型、优化器、batch size、epoch 上限、early stopping、预处理和指标仍由公共配置和实验 schedule 控制；只把方法自身参数切换为更接近公开库默认值、公开代码参数名或论文中明确给出的设置。该组不包含数据集专属覆盖文件，因此不会因为 `covertype` / `susy` 等大数据集自动降低方法自身参数。

命令行中的 `--max_epochs`、`--batch_size` 和 `--subset_fraction` 会覆盖配置文件中的对应值。
