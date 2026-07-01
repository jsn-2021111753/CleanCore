# experiments

最终实验说明文件放在这里，用于描述 Lab1-Lab6 的最终实验组合。

当前已有：

- `lab1.yaml`：最终实验一，`random` 20% 错误，所有最终方法配置。
- `lab2.yaml`：最终实验二，`random/shift` 多错误比例。
- `lab3.yaml`：最终实验三，CleanCore 消融，`random` 40% 错误。
- `lab4.yaml`：最终实验四，预训练轮数敏感性。
- `lab5.yaml`：最终实验五，子集预算敏感性。
- `lab6.yaml`：最终实验六，滑动窗口 `L` 敏感性。

实验一默认按数据集逐个运行；同一个数据集内所有方法使用相同的 `threads_per_run`、`max_epochs` 和 `batch_size`。这样可以保证同一数据集上的方法比较使用一致的下游训练预算，同时通过多个方法并行尽量利用 32 核 CPU。

实验一中，所有 subset selection / hybrid 方法在同一个数据集上使用相同的目标 `subset_fraction`：

```text
wdbc, banknote, pendigits, magic: 0.10
sensorless, miniboone, skin, covertype: 0.05
susy: 0.01
```

数据集专属方法配置只调整自然的方法参数，例如折数、warmup epoch、loss epoch、round 数或树模型规模；不使用额外候选池上限、打分样本上限、reference/eval 上限或固定 `subset_size`。

常用命令：

```bash
cd code
bash scripts/run_lab1.sh --dry-run
bash scripts/run_lab1.sh
bash scripts/run_all_labs.sh
```
