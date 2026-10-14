# 运行交接记录

## 开发更新：节点上下文（2026-10-14）

图模块新增共同单位下的水平、尺度、最近值、趋势、覆盖率，以及可用源时钟的日/周周期特征。
独立配置 `configs/next_stage_context.yaml`、侧文件 `data/features/context-v1/`；
当前固定快照 `stage2-native-v1` 仍为 `3c31da6`，未重启、未应用新结构。
新结构仅完成 C 上 81 项 CPU 测试及真实权重 CPU 通路检查，没有精度提升结论。
输出低秩适配尚为下一项消融设计，未训练；不能自动追加 GPU 预算。
细节及 METR-LA 文献结果见 `docs/METR_LA_ARCHITECTURE_REVIEW.md`。

16:47（Asia/Shanghai）只读观察：runner `1534896` 存活，快照工作区干净，
METR-LA 四份结果完整；PeMS-BAY / GOFA-F(Bolt) 仍在训练，已完成 27 个 epoch。
预算账本约已用 4,323 秒。这是一次状态观察，不是实时值或新结构的结果。

## 新阶段运行中（2026-10-14 15:35 启动，Asia/Shanghai）

用户要求启动新训练。原生 Bolt-mini、GOFA-F(Bolt)、官方匹配 HSDGNN 接入及机制检查已完成。
新阶段只用种子 42，六个数据集各 4 种设置（含 Persistence），共 24 份完整评价目标。
采用此前最短历史任务的配对 L/H；40 GiB 共享显存上限、至少 16 GiB 余量，新增总预算 8 小时。
训练使用新的固定快照和独立 `.venv-native`，不恢复旧首轮。
每数据集 GOFA-F 上限 25 分钟、HSDGNN 上限 40 分钟；候选只看验证集，预算耗尽保存并停止。
当前实现、恢复与查看进度的命令见 `docs/NEXT_STAGE_PROTOCOL.md`。

- 固定代码：`3c31da62b39876c18a3e62ae3561d83f5be74da0`。
- 固定快照：`/home/dcg/dcg_data/GOFA-F-runs/stage2-native-v1`。
- 训练 PID：`1534896`；超时包装 PID：`1534895`，后续使用前须重新核对身份与存活。
- 输出：开发仓库下 `outputs/stage2-native-v1/`；日志：`outputs/stage2-native-v1.log`。
- 状态：`outputs/stage2-native-v1.status` 与阶段目录内 `status.json`。
- 预算账本：阶段目录内 `budget.json`；启动记录：`outputs/stage2-native-v1.launch.json`。
- 启动后核对已完成 METR-LA 两种图位置各 200 次更新，按完整验证 MAE 选择间隔位置 `[0,2]`，
  继续同一检查点的正式训练；已观察到超过 800 次更新。这不是完整测试结果或普遍位置优劣结论。
- 微批量/有效批量为 64/64，无梯度累积；关闭梯度检查点，进程显存启动时约 2.8 GiB。
  后续 GNN 或更大数据集若需降低微批量，保持有效批量 64；不提高 40 GiB 上限或改独占模式。
- 原生真实权重机制检查累计约 7.97 秒，计入新阶段总预算；外部 timeout 预留 60 秒保存。
- 新快照启动后核对干净；旧首轮保持停止，未修改旧快照、数据或 `.venv`。

在 C 的终端持续查看日志：`tail -f /home/dcg/dcg_data/GOFA-F/outputs/stage2-native-v1.log`。
结构化进度：在开发仓库执行
`watch -n 10 'PYTHONPATH=src .venv-native/bin/python -m gofa_f.native_progress outputs/stage2-native-v1'`。
当前只授权这一新阶段；不要再次并行启动相同阶段或恢复旧首轮。

## 最新状态：旧首轮已停止（2026-10-14 14:46，Asia/Shanghai）

用户最新要求“不再运行旧首轮”，取代此前“GOFA-F(Chronos) 完成后暂停”的安排。
核对 PID、启动时刻、用户、命令和快照目录后，已停止 runner `703442` 和守护 `1379816`；
父脚本 `703436` 也已退出。14:46 复核三个 PID 均不存在，GPU compute-apps 列表为空。
没有启动新训练或测速，也没有操作其他用户进程。

- `outputs/stage1-v2.status`：`stopped by user; old stage1-v2 cancelled`。
- `outputs/stage1-v2.pause.json`：`state=stopped_by_user`，记录停止时间与被中断作业。
- 保留 4 组完整结果：METR-LA 的 Persistence、T5、GOFA-F(T5)、Chronos。
- METR-LA / GOFA-F(Chronos) 保留 `best.pt` 与 `history.jsonl`；完成 19 个 epoch（0–18），
  尚无 `result.json`，不能当作已完成正式结果。该检查点不含完整恢复状态。
- 固定运行快照和开发仓库在停止时均干净；未修改数据、环境或已有检查点。

后续默认 L→H 已按此前选定的最短历史任务配对修正为：12→12、12→3、24→6、12→12、8→8、24→3。
原生 Bolt/HSDGNN 接入及加速仍待实现。以下内容为历次观察和旧安排，以本节最新状态为准。

## 历史记录

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

用户在 B 接续时进一步要求先只进行一个种子。当前配置只注册种子 42，
三个预测长度的默认计划缩为 126 组；43/44 暂缓，原 378 组仅为后续扩展方案。
正在运行的 stage1-v2 本来就只有种子 42，无需重启；未启动预测长度 24/48 或额外搜索。

B 与 C 的命令行 HTTPS 推送未能完成认证：B 凭据管理器等待，C 的 helper 指向已失效的 `gh` 路径。
改用当前已连接的 GitHub 工具同步本轮提交，不复制凭据，不修改训练环境。

## 用户暂停要求（2026-10-14 11:41，Asia/Shanghai）

用户取消吞吐评测，要求 METR-LA / GOFA-F(Chronos) 完成后先暂停首轮后续作业。
不要启动测速、DST-SGNN 或其他数据集的后续训练，等待用户进一步指示。
GPU 为共享资源，不改独占模式，不操作其他用户的任务。

- 当前首轮 runner 仍为 PID `703442`，父进程 `703436`。
- C 上已启用外部守护 PID `1379816`；11:41 核对状态为 `armed`，当前训练继续。
- 守护使用固定副本 `.remote/pause-stage1-v2.py`，不修改运行快照或 `.venv`。
- 条件：`outputs/stage1-v2/metr_la/gofa_chronos/h12/seed42/result.json` 完整可读、
  `status=complete`，且数据集、模型、种子、预测长度和记录的 runner PID 全部匹配。
- 条件满足后先冻结 runner 的后续调度，再用 SIGTERM 结束这个 runner，释放其 GPU 资源。
  仅操作本用户该阶段进程，使用 `/proc` 启动时刻防止 PID 复用误伤。
- 记录：`outputs/stage1-v2.pause.json`；日志：`outputs/stage1-v2.pause.log`。
  成功后记录 `state=paused`，阶段 `.status` 写为 `paused after metr_la/gofa_chronos/h12/seed42`。
  父脚本 EXIT trap 可能短暂写出 failed(143)，最终以守护的明确暂停记录为准。

首轮快照仅保存 `best.pt` 的模型权重、epoch 和配置，不含优化器、随机状态或数据迭代位置，
不支持精确的中途断点续训。这里等待当前作业完整结束后才停止调度，保留所有完整结果。
未来获准继续时，可以跳过已完成作业；不能把最佳权重加载描述为完整状态恢复。
恢复前必须重新核对守护、进程和是否出现未完成的新作业目录，不能直接重复启动。

用户随后要求按数据集最短已有历史、优先物理图修订后续协议。开发默认配置已更新为
12/12/24/12/8/24 步（依次 METR-LA、PeMS-BAY、AQI、AQI-36、Electricity、Solar），
旧 96 步配置另存 `configs/benchmark_stage1.yaml`，当前快照继续保持原协议。
出处与模型流程见 `docs/MODEL_FLOW_AND_PROTOCOL_V2.md`、`configs/protocol_sources.yaml`。
此修订不授权启动新训练；已设置的完成后暂停要求仍有效。服务器为共享 GPU，继续禁止独占。

2026-10-14 14:08（Asia/Shanghai）重新核对：B/C 为 `e7b5184`，旧 runner 与守护仍存活、
pause_state=armed，GPU 为 H100 PCIe/Default，总显存 81,559 MiB、空闲 77,655 MiB。
用户已将下一阶段本项目显存上限选为 40 GiB，允许在共享余量内扩大微批量。
开发默认预测长度现为 1/1/3/12/1/3；当前快照不变。
下一阶段已选择原生 Bolt-mini、其图版本及新 GNN HSDGNN，新增 GPU 预算 8 小时，
六数据集分别训练，先精度、语言与迁移后置；完整设计在 `docs/NEXT_STAGE_PROTOCOL.md`。
本轮只更新长度配置、相应验证搜索兼容与设计清单，未接入/启动新的模型训练或 GPU 测速。

2026-10-14 用户进一步澄清：预测长度须取此前最短历史任务对应的值，不独立寻找最短 H。
开发默认 L→H 修正为 METR-LA 12→12、PeMS-BAY 12→3、AQI 24→6、AQI-36 12→12、
Electricity 8→8、Solar-Energy 24→3，取代上一条的 1/1/3/12/1/3。
此前选定的历史来源保持不变，来源与配对证据见 `configs/protocol_sources.yaml`。
仅修订后续协议；原首轮快照不变。用户随后取消旧首轮，执行结果见本文件顶部最新状态。
没有启动新训练或测速。
