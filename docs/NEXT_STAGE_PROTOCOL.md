# 下一阶段：最短预测长度、末端图交织与训练加速

2026-10-14。本文件记录用户在方案讨论后确定的选择及本轮修订。
当前只更新协议和既有长度配置；**原生 Bolt/HSDGNN 接入、缓存和限时调度仍待实现，未启动新训练**。
`configs/next_stage.yaml` 是设计清单，不能当作 `gofa-benchmark` 的可执行配置。
既有 `configs/benchmark.yaml` 仍运行旧编码器模型，但已经使用以下新长度。

## 历史与预测长度

|数据集|历史 L|预测 H|预测跨度|H 的直接出处|
|---|---:|---:|---|---|
|METR-LA|12|1|5 分钟|Hybrid Deep Learning Models for Traffic Prediction in Transportation Systems，Sections 2.2 / 7|
|PeMS-BAY|12|1|5 分钟|同上；325 站点、5 分钟采样，T'=1|
|AQI / PM2.5-437|24|3|3 小时|Cini 2024 博士论文 Section 4.1.1，纸面页 40，W=24,H=3|
|AQI-36|12|12|12 小时|PriSTI IV-C，历史 12 步预测未来连续 12 步|
|Electricity|8|1|1 小时|ODEStream，TMLR 2025，Algorithm 1 的 y[j+1] 与 ECL 单步预测实验|
|Solar-Energy|24|3|30 分钟|StemGNN 附录 D，10 分钟采样、未来 0.5 小时|

“最短”指已审计且核验的工作，不声称穷尽所有文献。AQI-36 尚未核验更短的同版本预测实验；
不把插补窗口、去噪迭代次数、或在 t+3 仅输出一个值冒充连续一步预测。
历史和预测长度的最短出处可以来自不同工作；组合后的划分、自然缺失处理并非这些论文的逐项复现。
Electricity 的一步长度不引入 ODEStream 的在线测试时更新：本项目仍不使用测试标签更新模型。
各模型在同一数据集采用相同 L/H、窗口起点、自然缺失目标和时间划分。

来源：[交通比较原文](https://eprints.bournemouth.ac.uk/37825/1/Zheng23-HybridModelsCompare%20%28Accepted%20version%29.pdf)、
[Cini 论文](https://andreacini.github.io/assets/pdf/phd_thesis.pdf)、
[PriSTI](https://arxiv.org/pdf/2302.09746)、
[ODEStream](https://arxiv.org/html/2411.07413v2)、
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
   约 36,477 窗 × 325 节点 × 2 token × 384 维 × 2 字节）。
   40 GiB 可为该缓存与训练工作区提供空间，但真实峰值仍需检查；一次不缓存六个数据集。
5. **计算与输入流水线**：Bolt BF16，HSDGNN 先 FP32+TF32；支持时使用 fused optimizer。
   workers=2、pin_memory、持久 worker 和预取；OMP/MKL 各 2 线程，限制共享服务器 CPU 占用。

冻结前缀缓存需要先通过缓存/非缓存预测与梯度一致性检查。TF32、BF16 的精度配置记录在 provenance。
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

当前旧 GOFA-F(Chronos) 仍在运行；守护仍为 armed。其完整结束后先暂停旧 runner，
保持旧快照、旧数据和环境。新方案接入完成并验证前不启动新阶段，预算耗尽也不自动扩展。
