# GOFA-F

面向多变量时序预测的图语言模型研究实现，借鉴 [GOFA](https://github.com/JiaruiFeng/GOFA)
在预训练 Transformer 内部交织图消息传递的思路。**目前没有正式数据集上的实测成绩，不能宣称达到较好结果。**

## 模型与对照

每个节点对应一条传感器序列。模型先读取预测起点之前的历史，在编码器第 1、3 层之后插入
带零初始化门控的图交换模块，再由后续预训练层处理交换后的隐藏状态，最终输出连续多步预测。
图由训练段的成对有效观测相关性构建，验证和测试数据不参与构图或拟合标准化参数。

|实验|输入与骨干|目的|
|---|---|---|
|T5 / GOFA-F(T5)|通用 `google-t5/t5-small` 编码器，数值 patch + 固定文本提示|同一语言模型加入图交换前后|
|Chronos / GOFA-F(Chronos)|`amazon/chronos-t5-small` 编码器，原生数值分桶词表|同一时序预训练模型加入图交换前后|
|DST-SGNN|官方 IJCAI 2025 实现，固定上游提交，匹配规模|近期 GNN 对照|
|Persistence|最后一个有效观测|基本合理性检查|

首版使用**编码器特征 + 监督预测头**，不是完整 T5 的文本生成，也不是 Chronos 原生采样解码。
因此结果中的 `t5` / `chronos` 表示该预测适配协议，不能冒充原模型的零样本结果。
默认冻结预训练权重，训练适配器、预测头和图层，仍保留穿过冻结 Transformer 的梯度。
总参数和可训练参数分别报告；参数接近的主比较采用总参数 ±10%，不能声称训练预算也匹配。

## 数据与实验

基于 [awesome-gnn4ts](https://github.com/kimmeen/awesome-gnn4ts) 选择，并按用户确认扩展能源领域的六个数据集：

|领域|数据集|注意事项|
|---|---|---|
|交通|METR-LA、PeMS-BAY|零值按数据集约定作为缺失；不要使用全数据拟合的标准化版本|
|环境|AQI、AQI-36|可能共享站点/时间，分别报告，不能当作独立跨域迁移证据|
|能源|Electricity、Solar-Energy|公开的电力负荷/太阳能发电数据，按用户确认扩展领域；保留真实零值|

候选协议：每个数据集 3 个预测长度 × 3 个随机种子，完整矩阵 378 个作业（含两个分别匹配规模的 GNN）。
能源数据来自 [LSTNet 原作者数据仓库](https://github.com/laiguokun/multivariate-time-series-data)，
也是后续 GNN 预测研究的常用数据。按用户选择替代需要申请权限的工业数据。
详细定义、原始数据来源和通过标准见 [实验协议](docs/EXPERIMENTS.md)。
注册矩阵不代表已经取得这些数据或跑完实验。

## 服务器运行

本地修改代码后上传 GitHub；在服务器 `/home/dcg/dcg_data/GOFA-F` 拉取并运行。
已有环境可直接激活；新环境使用 Python 3.11，CUDA 对应版本的 PyTorch。

```bash
git clone https://github.com/pootis12345/GOFA-F.git /home/dcg/dcg_data/GOFA-F
cd /home/dcg/dcg_data/GOFA-F
python -m pip install -e '.[dev,data]'
ruff check .
pytest -q
python scripts/fetch_baseline.py
python scripts/smoke_pretrained.py
python scripts/download_public_data.py
python scripts/prepare_public_data.py
```

`pytest` 使用小型随机初始化的编码器验证机制；`smoke_pretrained.py` 才下载真正的预训练骨干并验证 GPU 前后向。
所有命令在服务器执行。本地不运行项目或测试。

将原始 HDF 转为统一格式（在服务器执行）：

```bash
gofa-prepare --input data/raw/metr-la.h5 --key df --frequency 5min --zero-missing --output data/processed/metr_la.npz
gofa-benchmark --plan
python scripts/pilot.py --datasets metr_la pems_bay aqi aqi_36 electricity solar
gofa-benchmark --seeds 42 --horizons 12 --output outputs/stage1-v1
gofa-benchmark --datasets metr_la --models persistence t5 gofa_t5 chronos gofa_chronos --output outputs/pilot-v1
gofa-benchmark --output outputs/benchmark-v1
gofa-report outputs/benchmark-v1 --output outputs/report.md
```

长时间的首轮完整训练可用 `bash scripts/run_stage1.sh`：六数据集、预测长度 12、种子 42，
共 42 组，使用完整时间划分和验证早停。它是完整矩阵的第一阶段，尚不支持多种子结论。
脚本使用已缓存权重，保存进程锁、提交号和退出状态，结束后生成结果表。

HDF 内部键名需以实际文件为准。CSV/Excel 必须显式列出传感器列，防止时间戳和异常标签成为输入。
NPZ 包含 `values[T,N]`、`mask[T,N]`，可选 `segments[T]` 和 `train_end/val_end`。
多段轨迹禁止跨段窗口；默认按时间 70/10/20 划分，验证/测试目标不会跨界。

每次训练输出配置、代码提交、权重来源版本、数据 SHA256、分割边界、标准化参数、邻接矩阵、
最佳验证检查点、逐预测步指标、耗时和峰值显存。输出目录不覆盖，失败记录保留。
冒烟运行带 `status: smoke`，不会进入正式结果表。

GitHub 自动运行远程 CPU 检查；可选的 GPU Actions 工作流需要已注册的
`self-hosted, linux, x64, gpu` runner。直接 SSH 运行不依赖 runner。
SSH 连接信息仅放在被忽略的 `.remote/` 目录，密码与私钥不进入仓库。

## 当前阶段

工程实现和远程验证已推进；参数匹配后的完整训练、统计比较和消融仍需完成。
六数据集已在 H100 服务器准备完毕，具体版本见 [数据清单](docs/DATASETS.md)，
已通过的检查见 [远程验证记录](docs/VERIFICATION.md)。
任何性能结论以服务器生成的 `result.json` 为准。

来源：[GOFA / ICLR 2025](https://arxiv.org/abs/2407.09709)、
[Chronos](https://github.com/amazon-science/chronos-forecasting)、
[DST-SGNN / IJCAI 2025](https://www.ijcai.org/proceedings/2025/0796.pdf)。
