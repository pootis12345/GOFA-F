# 下一阶段：最短历史任务配对长度、末端图交织与训练加速

2026-10-14。本文件记录用户在方案讨论后确定的选择及本轮修订。
用户随后授权启动新训练。原生 Bolt/HSDGNN 接入、冻结前缀缓存、单种子限时调度与恢复现已实现，
由 `python -m gofa_f.native_stage` 执行 `configs/next_stage.yaml`；不使用旧 `gofa-benchmark` 运行新阶段。
既有 `configs/benchmark.yaml` 提供同一组数据路径、长度和图来源，其中的旧模型菜单不参与新训练。

后续图上下文修订使用独立 `configs/next_stage_context.yaml`，尚未启动训练；
实现、验证边界及输出端适配设计见 [METR-LA 架构分析](METR_LA_ARCHITECTURE_REVIEW.md)。
当前 `stage2-native-v1` 继续使用原固定快照，不含这项开发修订。

## 历史与预测长度

|数据集|历史 L|预测 H|预测跨度|此前选定的历史任务及配对 H 出处|
|---|---:|---:|---|---|
|METR-LA|12|12|60 分钟|DCRNN 官方 `dcrnn_la.yaml`，seq_len=12、horizon=12|
|PeMS-BAY|12|3|15 分钟|StemGNN 官方 README Table 1，window_size=12、horizon=3|
|AQI / PM2.5-437|24|6|6 小时|ICML 2024 Spatiotemporal Downsampling，附录 Table 3，Window=24、Horizon=6|
|AQI-36|12|12|12 小时|PriSTI IV-C，历史 12 步预测未来连续 12 步|
|Electricity|8|8|8 小时|TSDE KDD 2024，Appendix D.3.2 / Table 13 的确定性 8→8 任务|
|Solar-Energy|24|3|30 分钟|StemGNN 附录 D，10 分钟采样、未来 0.5 小时|

“最短”指此前已审计且核验的历史长度，不声称穷尽所有文献。
按用户最新澄清，固定此前选定的历史任务出处，直接采用同一任务配对的预测长度；
即使有历史长度并列的其他工作，也不为了缩短 H 更换此前选定出处。
此前将历史和预测分别取最短的方案被本修订替代。METR-LA 保留 DCRNN 的 12→12，
PeMS-BAY 保留 StemGNN 的 12→3，不能用另一个工作的一步 H 组合成新任务。
本项目的时间划分、标准化和自然缺失处理仍单独记录，不声称逐项复现来源论文。
不采用原文的测试时在线更新或全序列插补；仍不使用测试标签更新模型。
各模型在同一数据集采用相同 L/H、窗口起点、自然缺失目标和时间划分。

来源：[DCRNN 配置](https://github.com/liyaguang/DCRNN/blob/master/data/model/dcrnn_la.yaml)、
[StemGNN 官方配置表](https://github.com/microsoft/StemGNN#training-and-evaluation)、
[ICML 2024 论文](https://proceedings.mlr.press/v235/marisca24a.html)、
[PriSTI](https://arxiv.org/pdf/2302.09746)、
[TSDE](https://arxiv.org/pdf/2405.05959)、
[StemGNN 附录](https://papers.nips.cc/paper_files/paper/2020/file/cdf6581cb7aca4b7e19ef136c6e601a5-Supplemental.pdf)。
完整来源在 `configs/protocol_sources.yaml`。

## 连续末端插图与间隔插图

原 GOFA 论文 Section 3.3 说明只在最后若干 Transformer 层间插入 GNN，首尾均保留 Transformer。
官方默认使用六个 GNN 层；代码用 `g_layer_idx = i - (num_hidden_layers - num_graph_layers)`
定位末端连续层间隙，并在无图前缀使用 `no_grad`（partial_grad 模式）。
因此“连续”仍是 Transformer-GNN-Transformer-GNN 交织，不是把两个 GNN 紧挨着堆叠。
来源：[论文](https://arxiv.org/html/2407.09709v2)、
[官方前向](https://github.com/JiaruiFeng/GOFA/blob/main/modules/gofa/gofa_modeling.py)、
[默认配置](https://github.com/JiaruiFeng/GOFA/blob/main/configs/default_config.yaml)。

对 4 层 Bolt-mini，两种等参数配置为：

```
间隔：E1 → G1 → E2 → E3 → G2 → E4     索引 [0,2]
末端：E1 → E2 → G1 → E3 → G2 → E4     索引 [1,2]
```

新阶段优先末端配置，因为冻结的 patch 输入与前两层均可在无梯度模式运行并缓存，
只对 G1 后的 E3/E4 和原生 decoder 保留输入梯度；间隔配置只能缓存 E1。
这一判断针对计算成本，不是已证明的精度优势；Mistral 的图语言任务不能替代 Bolt 的时序消融。
每数据集用同一学习率 1e-3、相同初始图权重、数据顺序与 200 次优化更新比较，
每候选最多 5 分钟，依据验证 MAE 选择；平分取末端配置。
若任一候选未完成相同更新数，不将其作为严格的位置优劣结论，正式候选仍只按可用验证结果选择。
该比较替代先前 GOFA 分支的两学习率试探，仍包含在每数据集 25 分钟预算中，不增加总预算。
旧首轮 6 层编码器的 `[1,3]` 指第 2、4 块之后，保持不变；不能混同为 Bolt 的第 1、3 块。

## 40 GiB 显存下的加速顺序

用户已选择本项目显存上限 40 GiB，取消方案中的固定 8 GiB 上限，继续共享 GPU。
14:08 核对 C：H100 总显存 81,559 MiB，空闲 77,655 MiB，compute mode 为 Default。
这是一次观察，不是可长期预占的配额。每次开始检查实时余量，使用上限为
`min(40 GiB, free_memory - 16 GiB)`；不足时降低微批量或等待，不更改独占模式。

1. **关闭 Bolt 梯度检查点**：2–3 个历史 token 不应继续沿用旧 97 token 的重算配置。
2. **较大微批量**：有效批量先固定为 64，依次尝试 64/32/16/8/4/2/1，累积相应为 1/2/4/8/16/32/64。
   有效批量从先前方案 8 改为 64 会影响优化，需通过验证曲线核对，不能视作数学等价加速。
   同一位置对照必须固定有效批量和样本顺序。容量检查纳入首个新训练作业，不独立跑吞吐评测。
3. **缓存冻结前缀**：原生输入模块和前缀保持 eval；将第一图模块之前的隐藏状态、mask、
   历史窗尺度、起点按数据/权重/标准化/布局/精度哈希缓存，只保留当前数据集和布局。
   两种位置可先共用 E1 缓存，末端配置胜出后分块升级成 E2 缓存。
   不缓存图模块后的可训练状态；当前旧 T5 输入适配器可训练，不能套用无梯度截断。
4. **缓存容量估算**：BF16 的 PeMS-BAY 训练前缀约 17 GiB（忽略无效窗口时的上界，
   约 36,475 窗 × 325 节点 × 2 token × 384 维 × 2 字节）。
   40 GiB 可为该缓存与训练工作区提供空间，但真实峰值仍需检查；一次不缓存六个数据集。
5. **计算与输入流水线**：Bolt BF16，HSDGNN FP32+TF32，使用 fused AdamW / Adam。
   实现采用整段 prepared 数组驻留 GPU、按起点直接索引窗口，省去 CPU worker、逐窗复制和预取；
   workers=0。数组本体远小于前缀缓存，计入显存上限。OMP/MKL 各 2 线程，限制共享服务器 CPU 占用。

冻结前缀缓存已通过真实权重的缓存/非缓存预测与梯度一致性检查。TF32、BF16 的精度配置记录在 provenance。
HSDGNN 的主体需要训练，不能缓存其隐藏状态当作加速。
Bolt 原生头一次输出 64 步；本项目只取前 H 步并限制损失目标，**H 变短不使原生头按比例变快**。
不为了缩短 H 重建原生输出头或破坏预训练权重。

## 预算、项目目标与执行约束

新增 GPU 总预算仍为 8 小时，包含缓存构建、机制检查、候选、训练、验证和最终推理。
每数据集 GOFA-F 25 分钟、HSDGNN 40 分钟，共 390 分钟；机制检查 15 分钟，完整评价 75 分钟。
不以“速度超过 GNN”为研究目标；目标是六数据集 MAE 分别超过原生 Bolt 与同规模 HSDGNN。
原生 Bolt 不训练；GOFA-F 冻结完整原生模型只训练两层图模块。
HSDGNN 用主体宽度 408、embed_dim=32 合理匹配约 21M 参数，保留原作者初始化。
FLAN-T5-small 分支、语言交互、跨数据集 GOFA-F 零样本、其他种子与额外消融继续后置。
单种子即使全部胜出，也不自动证明跨域普适性或统计显著性。

2026-10-14 14:46（Asia/Shanghai），按用户最新要求立即停止旧首轮，不再等待 GOFA-F(Chronos) 完成。
旧 runner、父脚本与守护均已退出，保留 4 组完整结果及未完成作业的检查点和日志。
保持旧快照、旧数据和环境。新方案在独立 `.venv-native` 中验证，预算耗尽不自动扩展。

## 新阶段运行与恢复

六数据集各运行 Persistence、原生 Bolt、GOFA-F(Bolt)、匹配 HSDGNN，共 24 份完整评价目标。
图位置候选与学习率候选只看完整验证集 MAE，测试集不参与选择；候选属于训练预算，
不将固定步数候选作为正式精度结论。每候选预留 60 秒用于验证，验证不完整时该次分数无效。
HSDGNN 保持固定官方源码、动态图算子与初始化；适配为单数值通道、训练段标准化、缺失处零填充。

这应称为官方代码的参数匹配限时适配，不能称为原论文精度的忠实复现。
官方没有报告本项目六数据集的成绩；宽度、属性数、调度器、正则、训练长度及划分的
逐项差异见 [HSDGNN 复现审计](HSDGNN_REPRODUCTION_AUDIT.md)。
原规模、原公开任务核验及充分训练对照尚未完成。
使用样本索引生成日内/周内循环相位，不声称是实际星期标签；这些已知时间特征不读取未来观测。
本项目的原始单位 MAE 损失、宽度、学习率选择、早停和限时设置另行记录，不冒充原论文逐项复现。

`last.pt` 保存可训练权重、优化器、Python/NumPy/Torch/CUDA RNG、epoch 与窗口排列游标，约每 30 秒更新。
已完成模型阶段有持久标记，恢复时跳过；完整结果不覆盖。只允许原固定快照与完全相同 manifest 恢复。
预算账本累计运行秒数，恢复时保留已花预算，并对异常退出未记录尾部保守补记 5 秒。
该预算以活动进程墙钟时间保守计费，包含缓存、验证、等待、检查点 I/O 和推理，不含旧首轮。
外部 timeout 在剩余预算结束前 60 秒请求停止，最多再给 60 秒保存，避免意外超时运行。

运行 `bash scripts/run_native_stage.sh outputs/stage2-native-v1`；
核对进程并获准恢复时，使用相同固定快照加 `--resume`，不能从开发 HEAD 任意重跑。
通过 `python -m gofa_f.native_progress outputs/stage2-native-v1` 读取不含测试分数的进度。
短历史和冻结前缀带来的实际速度需要以该阶段记录为依据，本次没有另跑吞吐评测。
