# HSDGNN 来源、数据覆盖与复现边界

审计日期：2026-10-14。对象为当前运行快照 `stage2-native-v1`（代码 `3c31da6`），
不把开发版的新上下文协议误当作当前实验。此次审计没有修改模型、训练、数据或运行环境。

## 结论及公开结果覆盖

当前应称为：**官方模型代码在本项目数据/训练协议下的单属性、总参数匹配、限时适配版本**。
它保留官方模型算子和初始化，但不是原论文实验或公开精度的忠实复现。

在原论文、官方仓库及本轮检索中，未找到 METR-LA、PeMS-BAY、AQI(437)、AQI-36、
Electricity 或 Solar-Energy 上的官方 HSDGNN 成绩。
[原论文](https://doi.org/10.1016/j.engappai.2025.110304)发表于 EAAI 2025；
[官方仓库](https://github.com/zhourongleiden/HSDGNN)登记的原实验数据是
PEMSD4、PEMSD5、PEMSD8、PEMSD11、PSML 和 NSRDB。
PeMS-BAY 不能等同 PEMSD4/5/8/11，Electricity 不等同 PSML，Solar-Energy 不等同 NSRDB。

## 来源和架构核对

固定上游提交：`3b83c6a8abb2229bc176c97ac71f7676bbf584d1`。
通过 GitHub 连接器取得该提交的 Git blob 身份，与 C 文件逐字节比对，以下六项一致：
`model/HSDGNN.py`、`main.py`、`lib/dataloader.py`、`lib/metrics.py`、
`model/trainer.py`、`config_file/HSDGNN_PEMSD4.conf`。
模型 SHA256：`ac1d8c3bf0c22679f769adc78bb2f63f1e76ad2c9593074aaeba1e7527359cfd`。

当前适配器直接导入官方 `HSDGNN`，保留三块残差结构、每块两个 GRU、动态图/属性图算子、
节点/时间嵌入、dropout 和输出求和；没有重写成近似 GNN，也没有增加物理图。
逐参数初始化沿用官方 main.py：矩阵 Xavier uniform，向量 uniform。
原生数据批量和当前有效批量均为 64；当前训练损失也在恢复物理单位后计算 MAE，
并非标准化空间 MAE，但额外应用了自然缺失标签 mask。

## 协议差异

下表中的官方训练设定以固定仓库代码为准，并单列论文与代码不一致处。
可核对 [main.py](https://github.com/zhourongleiden/HSDGNN/blob/3b83c6a8abb2229bc176c97ac71f7676bbf584d1/main.py)、
[配置](https://github.com/zhourongleiden/HSDGNN/blob/3b83c6a8abb2229bc176c97ac71f7676bbf584d1/config_file/HSDGNN_PEMSD4.conf)、
[数据处理](https://github.com/zhourongleiden/HSDGNN/blob/3b83c6a8abb2229bc176c97ac71f7676bbf584d1/lib/dataloader.py)、
[训练器](https://github.com/zhourongleiden/HSDGNN/blob/3b83c6a8abb2229bc176c97ac71f7676bbf584d1/model/trainer.py)。

|项目|官方仓库/论文|当前固定运行|
|---|---|---|
|数据集|上述原六数据集|本项目另一组六数据集|
|数值属性|通常 3，NSRDB 配置 17|每节点 1 个数值属性；两个时间通道仅用于时间嵌入|
|GRU 宽度/嵌入|64 / 32|408 / 32，为匹配 Bolt 的总参数量|
|模型规模|固定配置静态计数约 0.574–0.612M|METR-LA 21.708M，全部可训练|
|L→H|原任务均 12→12|本项目配对长度；仅 METR-LA、AQI-36 为 12→12|
|时间划分|60/20/20|70/10/20|
|标准化|官方 loader 在切分前使用全时段、按属性聚合统计|只拟合已观测训练段，逐节点标准化|
|窗口边界|在各 split 内独立构造窗口|目标全在 split 内，历史允许使用此前已观测 split|
|缺失|普通 L1；评价按值阈值筛选|输入缺失填标准化零；损失/评价排除自然缺失标签|
|学习率|论文 0.001；仓库配置 0.003；每 epoch 乘 0.99|0.001/0.003 各 200 更新候选后验证选择；后续恒定|
|Adam 正则|weight_decay=0|weight_decay=0.0001；fused Adam|
|梯度裁剪|配置关闭|范数阈值 1|
|训练与早停|300 epoch；配置 patience 15，能源配置 30|最多 30 epoch、patience 5；每数据集总限 40 分钟，含候选|
|随机性|配置 seed=0；论文报告多次运行|用户限定 seed=42 一次|
|精度路径|官方 FP32 环境|FP32 + TF32|

评价还有 MAPE 定义差异：官方分母为 `true+0.001` 且按真值阈值筛选；
本项目用绝对真值、自然 mask 和 1e-5 阈值。验证末批与聚合方式也不同。
这些差异均需披露，不能直接将两套成绩混入同一精度复现表。

官方 loader 使用全时段统计，包含验证/测试分布信息；本项目坚持训练段拟合，
不应为了逐字复制该流程而引入未来统计量。
另外官方 loader 的 `reshape(-1,3)` 对 NSRDB 的 17 属性配置存在静态兼容疑点，
原任务核验前还需检查实际数据形状/发布版本。本轮没有运行原始数据实验，不能宣称已验证该路径。

## 对科研结论的影响

1. 单数值属性时，属性关系矩阵变成 1×1；算子仍存在，但不能展示原论文多属性依赖建模的完整优势。
   当前六个注册任务均为单属性预测，应给所有方法同等输入，不凭空给某一方法添加额外观测。
2. 约 37 倍的参数扩张是总参数匹配需要，不保证性能单调提高；全部从头训练还可能增加优化难度。
   它与冻结 Bolt 加约 0.1M 适配参数的可训练规模和训练预算并不相同。
3. 官方调度器、正则和训练长度没有被忠实沿用，当前又可能因预算结束而未收敛。
   即使 GOFA-F 超过当前版本，也只能支持本配置/预算下的比较，不能等同超过原工作充分训练水平。
4. 机制测试、源码哈希和有限梯度证明实现来源/流程；不能证明复现了原论文精度。

后续正式验证宜保留原规模 HSDGNN 与总参数匹配 HSDGNN 两行，按验证集分别调参；
补一个原公开任务的端到端复现核验，并给最终对照充分训练或明确标注预算受限。
原论文与官方代码的学习率差异需同时登记，不把任一方悄悄当作唯一标准。
上述为待安排工作，不授权启动新任务、增加 GPU 预算或修改当前快照。
