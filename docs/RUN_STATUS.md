# 运行交接记录

2026-10-13，H100 服务器首轮完整数据实验已启动。

- 开发仓库：`/home/dcg/dcg_data/GOFA-F`。
- 运行快照：`/home/dcg/dcg_data/GOFA-F-runs/stage1-v2`（detached worktree）。
- 固定代码：`8e955b66c26f4d64b555e526813c7a03242f190a`。
- 训练进程：PID `703442`（仅供本次会话定位，重启后可能失效）。
- 输出：开发仓库下 `outputs/stage1-v2/`。
- 日志：`outputs/stage1-v2.log`。
- 状态：`outputs/stage1-v2.status`。
- 完成后结果表：`outputs/stage1-v2.report.md`。

该阶段有 42 组完整数据训练/评估：六数据集，预测长度 12，种子 42，7 种模型设置。
最长 30 epoch，验证早停 patience 5，没有冒烟步数或评估窗口限制。
开始时已完成 Persistence 的第一组完整评估，T5 正在训练；尚不能声称 GOFA-F 性能达到目标。

后续在开发仓库编辑并上传 GitHub，不在运行快照编辑源码，不修改其使用中的 `.venv`。
当前快照独立保留官方 GNN 来源文件，模型权重按提交号固定并离线加载。
数据和输出通过符号链接共享，源代码通过 `PYTHONPATH` 显式绑定到快照。
数据文件不可在训练期间重写；任何新预处理版本应放入新目录。

最初两个运行的 `git_dirty=true` 来自未忽略的 `.venv`、`data`、`outputs` 符号链接。
已核对快照的 tracked diff 为空，并在 Git 本地 exclude 中忽略这三个链接；没有更改源码或结果文件。
后续运行会记录干净状态。这一早期元数据差异需在审计时保留。

`outputs/stage1-v1/` 是迁移到固定快照之前的中止尝试，保留以供审计。
报告时仅指定当前阶段目录，不能把不同尝试的同一种子重复汇总。

下一阶段需要评估完整训练的验证结果，固定设置，再扩展种子 43/44、预测长度 24/48 与消融。
测试集结果不得用于选择超参数。

## B 接续核对（2026-10-13 20:33，Asia/Shanghai）

- B 的空目录 `D:\GOFA-F` 已从 GitHub 建立本地检出，起点 `35ab2d7`；C 开发仓库同一提交且干净。
- B 使用自身 SSH 配置和现有认证连接 C，没有复制 A 的私钥或连接配置。
- PID `703442` 仍有效，父进程 `703436`；状态文件为 `running`。
- 运行 cwd 为 `stage1-v2` 快照，提交仍为 `8e955b6`，tracked/untracked 状态干净。
- 仅 METR-LA/Persistence 有完整结果，METR-LA/T5 已完成 23 个 epoch（日志 epoch 从 0 开始）。
- GPU H100 PCIe，20:17 核对时约 1.25 GiB 显存、90% 利用率；不能据低显存推断可重复启动。
- 未显示或比较测试分数，未修改快照、共享数据、权重缓存或 `.venv`。

本记录是一次观察，不是实时状态。后续可用 `python -m gofa_f.progress outputs/stage1-v2 --expected 42`
读取不含测试分数的状态快照，并另行核对进程。验证专用学习率搜索工具见 `docs/TUNING.md`；
108 组搜索和完整 378 组扩展均未启动，当前完整训练继续运行。

B 与 C 的命令行 HTTPS 推送未能完成认证：B 凭据管理器等待，C 的 helper 指向已失效的 `gh` 路径。
改用当前已连接的 GitHub 工具同步本轮提交，不复制凭据，不修改训练环境。
