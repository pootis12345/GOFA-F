# TITAN 启发的 GOFA-F 图交换架构

2026-10-15（Asia/Shanghai）。已实现并在 C 禁 CUDA 验证；尚未训练或取得新精度结果。
协议 `bolt-prior-routed-v1`，配置 `configs/titan_inspired.yaml`。

## 借鉴范围

来源：[A Time Series is Worth Five Experts](https://arxiv.org/html/2409.17440)，arXiv:2409.17440v1。
借鉴变量关系建模、输入相关路由和先验引导；保留 Bolt 的时间建模与 GOFA 的层间交织。
截至核对日，[官方 model.py](https://github.com/sqlcow/TITAN/blob/main/model.py) 仍为 `coming soon`。
本实现是新的 GOFA-F 模块，不是 TITAN 复现，也没有新增完整五专家、DTW、教师蒸馏或专家预训练。

## 实际结构

冻结 Bolt-mini、late_contiguous `[1,2]`、五维历史统计、rank=64 图瓶颈、rank=4 私有输出增量均保留。
两层图模块分别新增 rank=16 的 query/key 映射和一个输出两维的节点路由器。
路由输入为当前历史 token 的有效均值，已经包含图模块原有的历史水平/尺度/最近值/趋势/覆盖率投影。
没有固定节点 ID 或按节点数分配的参数；没有把未来标签、验证目标或测试目标作为路由输入。

令 `z_i` 为低维历史 token，`s_i` 为其有效均值，`A[i,j]` 为先验图权重：

```text
q_i = tanh(Wq s_i), k_j = tanh(Wk s_j)
D[i,j] = softmax_j(log A[i,j] + q_i @ k_j / sqrt(16))
alpha_i = (1 - prior_floor) * sigmoid(router_0(s_i))
g_i = tanh(global_gate + router_1(s_i))
message_i = (1-alpha_i) * aggregate(A,z)_i + alpha_i * aggregate(D,z)_i
hidden'_i = hidden_i + g_i * up(GELU(message_i))
```

`A=0` 的边和完全无观测的源节点不进入 D；没有有效邻居的行返回零消息。
两种聚合分别按每个历史 token 的可用源质量归一化，再混合，不能先混合权重后忽略缺失覆盖差异。
`prior_floor=0.5` 保证非线性之前至少 50% 的消息聚合分量来自先验图；不保证任何单条边有固定最小权重。
交通保留道路距离图，空气质量保留地理图，能源保留训练段相关图；本版不增加先验图之外的新边。
动态关系在一个窗口内由历史决定，并在历史 patches 间共享；不是预测步级路由。

Q 随机初始化，K、节点路由器和上下文投影零初始化，全局门沿用零初始化。
初始 D 等于归一化先验，节点门等于原共享门，最终初始预测与原生零增量预测一致。
新增参数使用隔离 RNG 初始化，确保两个版本共有图权重逐元素相同，不改变下一层的配对初始化。
零门会使部分参数在首次反向时梯度为零，不能据此判定模块未参与前向。

## 接口、缓存与恢复

`NativeBolt` 新增可选参数 `graph_exchange="prior_routed"`、`graph_attention_rank=16`、`graph_prior_floor=.5`。
显式选择新模块时要求上下文和 late_contiguous；旧代码默认 `static`，历史布局和检查点含义不改变。
可使用 `NativeBolt(native, adjacency, horizon, **config["architecture"])` 构造新模型。
`configs/titan_inspired.yaml` 是架构说明，不是已注册训练阶段；旧 stage3/stage4 入口不能用它启动新实验。

动态关系全部在第一处图交换之后计算；仍可共享冻结前两块的 E2 前缀缓存。
路由和消息归一化强制 FP32，输出增量转换回输入 hidden dtype；保留 BF16 骨干通路。
训练恢复协议额外绑定 exchange、布局、图宽度、特征维数、注意力宽度、先验比例和输出 rank。
改变这些配置不能直接续训；旧静态训练器的恢复协议保持原样。

当前实现使用 `[B,N,N]` 稠密矩阵并施加图边掩码，计算不是稀疏复杂度。
以最大 N=437、B=64 计，一份 FP32 关系矩阵约 46.6 MiB；训练峰值还包括梯度/骨干等，不能据此推断全程峰值。
不增加完整 Transformer 专家；实际 GPU 时间和 BF16 CUDA 通路尚未验证，不声称新模型更快。

## 参数与验证

每层新增 `2*64*16 + 64*2 + 2 = 2,178` 个参数，两层共 **4,356**。
总注册参数 **21,349,382**；可训练张量参数 **113,286**，比 G2-V 的 108,930 增加约 4.0%。
点预测只监督中位数和前 H 步，有效可训练参数上界为 110,994—111,030；这是潜在参与口径，不是自由度满秩证明。
历史 UFG-A 的 108,781—108,975 相对上述两个口径仍在 ±10% 内，尚未补齐约 21M 总参数匹配。

C 原生环境 147 通过、11 按依赖跳过；UFG 环境补充 12 通过，其中一项重复，合计 **158 个独立用例全部覆盖**。
覆盖配对初始化、输入相关性、图边支持、缺失/空邻域、节点置换、批次隔离、自环隔离、有限梯度、CPU BF16，
以及短历史、冻结骨干、缓存预测/梯度、完整状态恢复和错误恢复配置拒绝。
Ruff 静态检查通过。测试全程 `CUDA_VISIBLE_DEVICES=''`，OMP/MKL 各两线程，不修改依赖。

六集真实权重 CPU 机制检查全部通过：训练历史单窗口、原始图、五维特征，初始预测最大差 0；
启用非零路由后，缓存/未缓存预测与梯度一致，原生骨干无梯度且权重不变。没有进行精度或吞吐评测。
完整证据与代码哈希：`docs/results/titan-architecture-cpu-v1.json`；原报告位于 C/B 的
`outputs/titan-architecture-cpu-v1/`。脚本 `scripts/check_routed_graph_cpu.py` 仅允许禁 CUDA 运行。

## 下一阶段边界

本次交付只修改架构；没有 GPU 等待器、训练窗口或新训练作业。
后续实验仍为 seed42、既定六组 L→H、验证选择、测试锁定后评价；需单独登记候选和固定快照。
用户选择的下一执行窗口为等待器启动后 12 小时，包含等待，不设累计 GPU 活动上限。
GPU 仍须连续三次完全空闲才入场，每 10 秒监控其他任务并保存让出，40 GiB 上限/16 GiB 余量不变。
保留单种子、AQI 重叠、开发期测试已查看及不同优化目标等边界。
无固定节点参数只提供后续跨数据集实验的结构便利，不能据此声称已具备单检查点泛化或语言交互能力。
