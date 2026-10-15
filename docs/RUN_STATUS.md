# 运行交接记录

## Stage12 未合格，Stage13 私有头容量准备部署（2026-10-16，Asia/Shanghai）

Stage12跑满100epoch（35,900更新）后 `no_eligible_variant`，等待器3067411/worker3068792退出，无失败。验证全步3.0955940661、第12步3.7417769070，未测试；目前合格模型仍Stage8T2。结果 `docs/STAGE12_RESULTS.md` 与机器可读副本，旧阶段不重启。

Stage13仅将T2私有输出适配rank4改为rank6，图和其他协议保持；119,302可训练参数是仍处于历史METR-LA UFG-A108,927自由度±10%内的最大整数秩预算，不做秩精度搜索。专项5项CPU检查通过，含真实参数、共同权重配对、私有头隔离和精确恢复，准备全仓库回归/独立部署。协议 `docs/STAGE13_HEAD_PROTOCOL.md`，验证门槛、严格共享GPU准入及10:58绝对截止保留。以下Stage12部署为历史。

## Stage12 独立快照和空闲等待器已部署（2026-10-16 05:01，Asia/Shanghai）

运行源码 `9ef6b091652566c30760aa26ff57f569e55a6aec`，快照 `/home/dcg/dcg_data/GOFA-F-runs/stage12-metr-contrast-v1`，输出C开发仓库 `outputs/stage12-metr-contrast-v1/`、日志同级 `.log`、身份同级 `.launch.json`。C禁CUDA、OMP/MKL各2线程208项通过/11可选依赖跳过，快照12项输入核验通过；Stage11旧等待器和worker已退出后才部署。

CPU等待器首次北京时间2026-10-16 05:01:17启动，PID3067411/start_ticks601913623；使用前核对身份，禁止重复启动。绝对训练截止10:43、硬截止10:58，等待恢复不顺延。唯一d1_contrast在T2原图上使用邻居混合消息减自身latent，不增参数，保留物理先验≥0.5，判定仍看预注册验证门槛。新GPU子任务严格三次间隔30秒空闲准入，运行中他人加入不退出，40GiB/Torch38GiB/16GiB余量不变。部署时尚无GPU机制或精度结果；10:58前继续依据验证机制顺序迭代。下方准备及旧阶段记录为历史。

## Stage11 未合格，Stage12 对比消息准备部署（2026-10-16，Asia/Shanghai）

Stage11 `no_eligible_variant`，等待器3028018/worker3032859退出，无失败。验证全步3.0788605691、第12步3.6878264836，未测试；当前合格模型仍Stage8T2。结果见 `docs/STAGE11_RESULTS.md` 和机器可读副本，旧阶段勿重启。

对锁定T2的训练/验证CPU消息诊断显示与自身latent平均余弦约0.91–0.95，仅作机制线索。Stage12回到原T2相关支持，仅将图增量改为邻居混合消息减自身latent，不增参数，保留原生残差和物理先验≥0.5。协议 `docs/STAGE12_CONTRAST_PROTOCOL.md`，CPU专项6项通过，准备全仓库回归及固定快照；硬截止仍北京时间10:58、验证门槛保持。下方Stage11部署为历史。

## Stage11 固定快照和严格空闲等待器已部署（2026-10-16 04:05，Asia/Shanghai）

运行源码 `171b188f8af93174b2d64916f2e0f369e8bc1848`，快照 `/home/dcg/dcg_data/GOFA-F-runs/stage11-metr-lagged-v1`，输出 C 开发仓库 `outputs/stage11-metr-lagged-v1/`、日志同级 `.log`、身份同级 `.launch.json`。C禁CUDA、OMP/MKL各2线程：202项CPU通过/11可选依赖跳过；快照13个输入哈希核验通过，Stage10旧等待器及worker退出后才部署。

CPU等待器首次北京时间2026-10-16 04:05:18启动，PID3028018/start_ticks601577812，使用前重新核对身份，禁止重复启动。绝对训练截止10:43、硬截止10:58不变；所有新GPU子任务须三次间隔30秒完全空闲准入，运行中他人加入不退出，40GiB/Torch38GiB/16GiB余量有效。唯一l1_lagged用原T2加lag1/2/3有向关联图（新增97边），门槛相对锁定Stage8T2，详见 `docs/STAGE11_LAGGED_PROTOCOL.md`。部署时尚无Stage11GPU机制或精度结果，10:58前继续按验证信息顺序迭代。下方准备与旧阶段记录为历史。

## Stage10 未合格，Stage11 滞后相关支持已预注册（2026-10-16，Asia/Shanghai）

Stage10 `no_eligible_variant`，等待器2984281/worker2986445已退出，无失败。验证全步3.0878077061、第12步3.7064949538，两项均未通过，未测试；现有合格模型仍是Stage8T2。结果见 `docs/STAGE10_RESULTS.md` 与机器可读副本，勿重启旧阶段。

Stage11回到T2的19维小结构，只加入训练前缀lag1/2/3的有向top8关联支持，不增参数，动态权重取并集逐边最大。新增97条有向边，图SHA `74a351a4ea001e5ab72e74921e9fdc7c2ade8e74807c3abf829cb1173414e5a5`，无跨连续时间段配对。专项7项CPU机制通过，准备全仓库回归及独立部署。协议 `docs/STAGE11_LAGGED_PROTOCOL.md`，验证门槛及绝对10:58截止保留；新GPU子任务必须严格空闲准入。以下Stage10部署为历史。

## Stage10 固定快照和CPU等待器已部署（2026-10-16 03:04，Asia/Shanghai）

运行源码 `3d3f95be58db7a269951556e289f8e534d3a6fcf`，快照 `/home/dcg/dcg_data/GOFA-F-runs/stage10-metr-encoded-v1`，输出 C 开发仓库 `outputs/stage10-metr-encoded-v1/`，日志同级 `.log`、身份记录同级 `.launch.json`。B/C/GitHub代码同步，C禁CUDA全仓库195通过/11可选依赖跳过，快照11项输入核验通过；旧Stage9等待器及worker已退出后才部署。

CPU等待器首次北京时间2026-10-16 03:04:31启动，PID2984281/start_ticks601213052；使用前核对身份，禁止重复启动。绝对训练截止10:43、硬截止10:58，恢复和等待不顺延。严格三次间隔30秒空闲准入后才允许GPU机制/训练，运行中他人加入不退出；40GiB/Torch38GiB/16GiB余量保留。唯一候选e1_encoded以锁定Stage8T2为对照，验证门槛和结构见 `docs/STAGE10_ENCODED_PROTOCOL.md`；部署时尚无Stage10精度结果。10:58前继续按验证信息进行后续顺序迭代，旧结果不覆盖。以下准备和Stage9记录为历史。

## Stage9 未达预定收益，Stage10 非线性历史编码准备部署（2026-10-16，Asia/Shanghai）

Stage9 `no_eligible_variant`，等待器2940624与worker2943503已核实退出，无异常。验证全步3.0727575920、第12步3.6824765599；改善仅约0.18%，未过0.5%门槛，测试留空、当前合格对照仍是Stage8T2。结果见 `docs/STAGE9_RESULTS.md` 及机器可读副本，勿重启本阶段。

Stage10仅把24维有序历史分支改为24→32→64 GELU编码，19维摘要/时钟保持线性，在图聚合前融合，初始增量为零。协议 `docs/STAGE10_ENCODED_PROTOCOL.md`；门槛继续相对Stage8T2验证，不按测试新增候选。C禁CUDA专项6项通过，准备全仓库回归及独立固定快照。用户授权连续迭代的绝对截止仍是北京时间10:58，新GPU子任务必须严格空闲准入。下方Stage9部署记录为历史。

## Stage9 独立固定快照等待GPU（2026-10-16 02:09，Asia/Shanghai）

运行源码 `e7acbb556d6b95d361d31c3c9224607266791394`，固定快照 `/home/dcg/dcg_data/GOFA-F-runs/stage9-metr-ordered-v1`，输出 C 开发仓库 `outputs/stage9-metr-ordered-v1/`、日志同级 `.log`、身份记录同级 `.launch.json`。C 禁CUDA、OMP/MKL各2线程：全仓库189通过/11可选依赖跳过，固定快照清单10输入核验通过；Stage8等待器及worker已核实退出后才启动本阶段。

CPU等待器首次于北京时间2026-10-16 02:08:54启动，PID2940624/start_ticks600879381；使用前重新核对身份，禁止重复启动。绝对训练截止10:43、硬截止10:58，等待和恢复不顺延。首观测为waiting_for_idle、活动0；GPU机制与精度尚未发生。唯一候选o1_ordered保留Stage8T2相关图，以T2锁定验证为对照，阈值和细则见 `docs/STAGE9_ORDERED_PROTOCOL.md`。本轮结束后在用户原定10:58窗口内继续顺序迭代，不按测试返选；各版旧结果与快照保留。下方CPU准备及Stage8记录为历史。

## Stage8 已完成，Stage9 已预注册并完成专项CPU验证（2026-10-16，Asia/Shanghai）

Stage8 `launcher/status=complete`，等待器2837909与worker2848602已退出，无失败。T1验证全步/第12步3.136663/3.787753未过全步门槛，未测试；T2验证3.078344/3.683162通过，锁定后测试全步3.406267、第12步4.136418。相较Stage6改善，但距TITAN外部参照3.08仍有1.056418差距。完整结果见 `docs/STAGE8_RESULTS.md` 与机器可读副本；勿重复启动Stage8。

用户授权持续迭代至北京时间2026-10-16 10:58。Stage9仅增有序段内偏差/掩码，保留T2的物理和相关图，以锁定T2为对照，从预训练权重重训。门槛为验证全步≤3.062952183796903且第12步≤3.6831616007344037，仅合格者锁定后测试。独立协议 `docs/STAGE9_ORDERED_PROTOCOL.md`。C禁CUDA、OMP/MKL各2线程的8项新增专项测试（含真实权重与恢复）通过，准备全仓库回归和固定快照部署；尚无Stage9GPU结果。严格空闲准入、40GiB/16GiB余量、运行中别人加入不退出继续有效。下方Stage8部署记录为历史。

## Stage8 METR-LA 细粒度历史图上下文等待 GPU（2026-10-15 23:50，Asia/Shanghai）

固定源码 `62f56fabe7c5e68ff8b0bbc3a6ad976b4ab3aecb`，C 快照 `/home/dcg/dcg_data/GOFA-F-runs/stage8-metr-temporal-v1`，输出在 C 开发仓库 `outputs/stage8-metr-temporal-v1/`、日志同级 `.log`、启动身份记录同级 `.launch.json`。B/GitHub/C 已同步代码，固定快照工作区干净。C 禁用 CUDA 的快照回归181通过、11跳过；清单核对10个输入哈希。相关图 SHA `d25fb015fb887d5fae60411af29cd20315fec31f8d635b4a965f07708312e8d0`，新增1,678条有向非自环动态候选边。

CPU 空闲等待器于 2026-10-15 23:48:28 首次启动，实际 PID `2837909`、start_ticks `600036782`；使用前须重新核对身份与存活，不得重复启动。`window.json` 规定训练截止 2026-10-16 11:33:28、硬截止11:48:28（北京时间），等待不顺延。23:50只读确认 `launcher.json=waiting_for_idle`、活动0，GPU 正被其他任务占用；尚无 Stage8 GPU 机制或精度结果。每个子任务须三次间隔30秒完全空闲才进入；运行中他人加入不退出、低显存余量暂停更新，本项目最多40 GiB、留16 GiB，不设置独占模式。只在终止或异常时通知；旧阶段不重启。详细协议见 `docs/STAGE8_METR_TEMPORAL_PROTOCOL.md`。以下部署前记录及 Stage7 结果为历史记录。

## Stage8 代码和 CPU 审计完成，等待固定快照部署（2026-10-15）

METR-LA 12→12、seed42 的 T1 四段历史图摘要与 T2 训练前缀 top-8 相关边已写入独立 Stage8 入口。C 上禁用 CUDA、OMP/MKL 各两线程的全仓库 CPU 回归为181通过、11项可选依赖跳过；真实 Bolt 权重与 METR-LA 前向、缓存一致性和精确恢复检查通过。真实协议审计确认训练前缀23,990行、207传感器及原物理图；相关图 SHA `d25fb015fb887d5fae60411af29cd20315fec31f8d635b4a965f07708312e8d0`，相对于物理图增加1,678条有向非自环动态候选边。本节部署前状态：尚无 Stage8 GPU 机制或精度结果，不得据此推断改进。细则见 `docs/STAGE8_METR_TEMPORAL_PROTOCOL.md`。下方 Stage7 完成状态为历史记录。

## 最新状态：Stage7 单检查点迁移评价已完成（2026-10-15 21:32，Asia/Shanghai）

固定代码 `2c9ba946882cf22787bce22602289bc9c77361fa` 的快照 `stage7-locked-transfer-v1` 已完成六集验证与测试；`launcher.json` 和 `status.json` 均为 `complete`，`failures` 为空，CPU 等待器 PID 2715356 及其 GPU 子进程已退出。METR-LA 源任务准确复现 Stage5 MAE；其余五集锁定检查点的测试 MAE 均高于各集单独训练的同架构模型。PeMS-BAY 差 13%，AQI-437/AQI-36 分别为 2.34/1.86 倍，Electricity/Solar-Energy 约为 6.82/6.77 倍。六集完整指标、逐步误差、显存、账本与局限见 `docs/STAGE7_RESULTS.md` 及 `docs/results/stage7-locked-transfer-v1.json`；B 已下载小型原始输出。监督器活动累计 55.905 秒，所有目标权重更新数均为0。本阶段不得重复启动 GPU 作业；每小时跟进在交付终止结果后暂停。下文等待器部署是历史记录。

## 最新状态：Stage7 单检查点六集迁移评价等待 GPU（2026-10-15 21:02，Asia/Shanghai）

固定快照 `stage7-locked-transfer-v1`、代码 `2c9ba946882cf22787bce22602289bc9c77361fa` 已与 GitHub、B、C 同步。C 禁用 CUDA、OMP/MKL 各两线程的六数据集真实 Bolt 权重前向检查通过；源 Stage5 METR-LA 检查点 SHA `7c3e3957d9a9fded1cff5302db775ef1bbc156c8e9dbcf97f26b07eb671beee9` 与锁定记录一致，快照清单的 22 个输入哈希核验通过。协议见 `docs/STAGE7_TRANSFER_PROTOCOL.md`。

CPU 空闲等待器于 21:02:11 首次启动，初始 PID `2715356`、start_ticks `599039152`；使用前须重新核对身份及存活，禁止重复启动。首次核对 `launcher.json` 为 `waiting_for_idle`、活动秒数0，GPU 正被其他进程占用，尚无 Stage7 GPU 评价结果。输出在 C 开发仓库 `outputs/stage7-locked-transfer-v1/`，日志同级 `.log`；固定八小时墙钟窗口于 2026-10-16 05:02:11 截止，不因等待顺延。每个 GPU 子任务须三次间隔30秒确认完全空闲才启动；运行中其他用户加入不退出，显存余量不足暂停更新，项目上限40 GiB、至少留16 GiB。每小时跟进 `gofa-f` 仅在终止或异常时通知。

本轮固定一份 Stage5 METR-LA 图模型检查点，对六集不更新权重直接评价；各目标仍使用自身训练前缀标准化与原图，因此应称校准后的权重迁移，不能称严格零样本。Stage5/Stage6 及旧快照不重启。下文 Stage6 已完成是历史记录。

## 最新状态：Stage6 METR-LA 长步预测已完成（2026-10-15 20:17，Asia/Shanghai）

固定快照 `stage6-metr-long-v1`、代码 `bcdc74a2213ef0179706a11386d68468fcf50461`；等待器和 GPU 子进程均已退出，`launcher.json`、`status.json` 均为 `complete`，无失败记录。C1/C2 和按验证规则触发的 C1 自环控制全部早停并完成锁定测试。验证集全 12 步原始 MAE 分别为 3.14136/3.13675/3.21524。C2 比 C1 仅低 0.147%，符合预定 0.5% 内选较小模型规则，因此选 C1；没有根据测试集返选。C1 测试全 12 步 MAE 3.46977、第 12 步 4.24974；相比 Stage5 的 4.60272 有改善，但仍比 TITAN 外部参照 3.08 高 1.16974。C2 第 12 步为 4.24377，自环为 4.37098。活动账本 3,874.27 秒，未触及六小时截止。完整逐步误差、参数、更新数、显存和局限见 `docs/STAGE6_RESULTS.md`，机器可读副本见 `docs/results/stage6-metr-long-v1.json`。本阶段勿重复启动 GPU 作业。

## Stage6 等待器已部署（历史记录，2026-10-15 18:32，Asia/Shanghai）

新候选 C1（六维源时钟）和 C2（再加共享节点直接预测残差头）已实现；两者均从预训练 Bolt-mini 重新初始化，固定 seed42、12→12、late_contiguous `[1,2]`。选择仅看完整验证集 12 步原始 MAE，合格时才训练相应自环控制，所有检查点 SHA 锁定后才评测测试集。独立配置与协议见 `configs/stage6_metr_long.yaml`、`docs/STAGE6_METR_LONG_PROTOCOL.md`。

C 禁 CUDA、OMP/MKL 双线程：仓库 CPU 测试 171 通过/11 可选依赖跳过；真实 Bolt 权重 C1/C2 初始预测与原生完全一致、缓存预测和梯度一致、骨干冻结，无测试目标加载。源 HDF 的 207 个传感器列与物理图节点顺序及准备数组逐值一致，零值按既有协议掩码；12→12 时间划分约 70/10/20。TITAN 官方模型代码仍为占位，因此 3.08 仅作跨实现外部参照。核验记录见 `docs/results/stage6-protocol-audit.json`、`docs/results/stage6-cpu-verification.json`。

固定快照 `/home/dcg/dcg_data/GOFA-F-runs/stage6-metr-long-v1` 使用代码 `bcdc74a2213ef0179706a11386d68468fcf50461`，源码保持干净。C 开发输出 `outputs/stage6-metr-long-v1/`，日志同级 `.log`。首次启动 CPU 等待器 PID 2611046/start_ticks 598142343；使用前重新核对身份与存活，禁止重复启动。六小时窗口从 2026-10-15 18:32:44 起算；2026-10-16 00:17:44 停止训练并锁定评价，00:32:44 硬截止，不因等待顺延。首次只读状态 `waiting_for_idle`、连续空闲1、活动秒数0；尚无 Stage6 GPU 机制或精度结果。每个新GPU子任务须三次间隔30秒确认完全空闲；运行中他人加入不退出，余量不足暂停更新，40 GiB 自身上限、16 GiB 共享余量。旧快照及结果不重启。

## 最新状态：Stage5 六集完整训练已完成（2026-10-15 16:43，Asia/Shanghai）

固定快照 `stage5-titan-six-v1`、代码 `5d68ef3caaba0f8f7b6460b8301782ee8f87cb68`。
只读核对 `launcher.json` 与 `status.json` 均为 `complete`，监督器 2403369 和子进程 2433421 已退出；
六集均有锁定 SHA、完整验证和测试结果，`failures` 为空。没有重启 GPU 作业。
测试原始 MAE（METR-LA/PeMS-BAY/AQI-437/AQI-36/Electricity/Solar-Energy）为
3.63528/1.14560/16.4618/32.2666/315.194/0.419836，六集均低于旧 G2-V、UFG-S/A 和历史限时 HSD 的对应 MAE。
监督活动账本 9,306.24 秒，六集更新数分别为 24,771/33,630/3,552/2,784/28,800/46,575；
Electricity 达到 100 epoch，其余均按早停结束。没有墙钟截断或未完成组。
完整指标、参数、显存、Electricity 诊断与局限见 `docs/STAGE5_RESULTS.md`，
机器可读副本见 `docs/results/stage5-titan-six-v1.json`；B 的小型原始输出见 `outputs/stage5-titan-six-v1/`。
本轮所有任务已结束，后续勿重复启动。下文 Stage5 “等待器已部署”是历史记录。

## Stage5 等待器已部署（2026-10-15 12:46，Asia/Shanghai）

固定代码 `5d68ef3caaba0f8f7b6460b8301782ee8f87cb68`，快照
`/home/dcg/dcg_data/GOFA-F-runs/stage5-titan-six-v1`，输出开发仓库 `outputs/stage5-titan-six-v1/`。
日志同级 `.log`，启动记录同级 `.launch.json`。首次PID2403369/start_ticks596063576；使用前重新核对。
12:46:15启动，12:46:56只读确认等待器存活、waiting_for_idle、连续空闲0、活动0；尚无GPUworker/机制或精度。
首次窗口固定：2026-10-16 00:31:15停止训练进入锁定评价，00:46:15硬截止，不顺延。
每GPU子任务启动/恢复前确认空闲；运行中他人加入不退出，低显存余量保留进程暂停更新，40GiB上限保留。
每小时跟进 `gofa-f` 已恢复，仅通知终止/异常，终止交付后暂停。旧Stage4不重启，不编辑运行快照。
下文“准备部署”是代码核验阶段的历史记录。

## Stage5 已完成代码/CPU核验，准备部署（2026-10-15）

用户批准完整TITAN启发架构直接六集，seed42/late_contiguous/原图与既定L→H，固定原始MAE/AdamW，不搜索。
独立入口 `scripts/run_routed_stage.sh`，固定快照拟名 `stage5-titan-six-v1`，新输出同名。
C禁CUDA覆盖173个独立CPU用例，六集真实入口前向有限/测试目标排除，18旧检查点SHA与六个旧缓存格式核对通过。
配置、恢复、12小时首次启动窗口、里程碑10/30/100、全局锁定后评价和分阶段峰值记录均接入。
最新资源要求：每个GPU子任务启动/恢复前严格空闲准入，运行中他人加入不退出；低显存余量保留进程暂停更新。
40GiB上限、12小时硬截止保留。旧阶段不重启、不重训GNN。GPU机制尚未执行。
细则 `docs/STAGE5_PROTOCOL.md`，CPU证据 `docs/results/stage5-cpu-verification.json`。

## TITAN 启发架构已实现并通过 CPU 验证（2026-10-15）

新增 `bolt-prior-routed-v1`：late_contiguous `[1,2]` 上的先验约束动态图注意力和节点级门控，
保留 Bolt、五维历史统计及私有 rank4 输出增量。至少 50% 先验聚合分量常驻，不新增图外边。
参数增加4,356，总注册21,349,382、可训练113,286；历史静态模型默认行为保持。
C禁CUDA、双线程核验158个独立CPU用例全覆盖，六集真实权重初始差0、冻结/缓存/梯度检查通过。
文档 `docs/TITAN_ARCHITECTURE.md`，配置 `configs/titan_inspired.yaml`，
证据 `docs/results/titan-architecture-cpu-v1.json`。只是借鉴 TITAN，不是官方复现。
没有启动训练或GPU等待器，没有新精度结论。下一执行窗口已选12小时，尚未登记候选/固定快照/实际起止时间。
旧stage4已完成，不重启；自动跟进保持暂停。下文均保留原阶段记录。

## 最新状态：Stage 4 v2 已完成（2026-10-15 09:14，Asia/Shanghai）

08:51:59正常结束，stage/launcher均complete；09:06确认监督器2004343、worker2004876退出，
09:13复核GPU无进程、0MiB、Default模式，固定快照和六集输入未改变。
18组主对照＋3组消融全部完成最终评价，21个锁定检查点SHA均已重新核验，无失败或未完成组。
运行代码仍为 `79a9c86cbb10a4a7118b56235fbaa2aad18303a1`，不要从开发HEAD重复启动。
原始单位MAE：G2-V六集胜原生Bolt，四集同时胜UFG-S/A；Electricity输UFG-A7.22%，AQI-36输两种UFG。
活动监督账本24,011.408秒（6:40:11），包括v1首次39.642秒；478,970个正式更新，落选pilot另800更新。
完整六集指标、消融、Electricity验证差距分解、参数与每组停止原因见 `docs/STAGE4_RESULTS.md`，
机器可读证据见 `docs/results/stage4-pure-graph-v2.json`。B已下载小型结果、逐节点诊断和历史。
该阶段执行已完成，全部数据集超过GNN、总参数匹配、语言交互、单检查点跨数据集能力仍未完成。
每小时跟进 `gofa-f` 将在交付终止结果时暂停；不将提前完成后的时间转为新候选。
下文部署/运行中状态均为历史观察。

## Stage 4 v2 部署记录（2026-10-15 02:11，Asia/Shanghai）

固定代码 `79a9c86cbb10a4a7118b56235fbaa2aad18303a1`，快照
`/home/dcg/dcg_data/GOFA-F-runs/stage4-pure-graph-v2`；输出开发仓库
`outputs/stage4-pure-graph-v2/`，日志同级 `.log`，启动记录 `.launch.json`。
CPU监督器初始PID `2004343`、start_ticks `592254839`，首次waiting_for_idle，活动累计39.642秒。
v1已停止；v2的 `migration.json` 保留METR-LA/G2-V 95更新、游标6080、完整训练状态与缓存，
没有完整验证或测试分数。首次机制失败不删除，后续必须重新核验进程身份/状态，不能重复启动。
v2源码在C禁CUDA双环境覆盖138个独立CPU用例；GPU机制仍须连续空闲入场后重新核验。
GPU活动总量和单组分钟无限制；11:01进入锁定评价，11:16硬截止；严格共享/让出规则不变。
每小时自动跟进 `gofa-f` 已切换v2，仅通知终止或异常。旧快照及结果均保留。

## Stage 4 已部署（2026-10-15 02:03，Asia/Shanghai）

02:04首次UFG CUDA机制检查中，2+1续训与连续3步的最大失败元素差为1.735e-6，
超过原1e-6绝对容差。原生机制通过后已启动G2-V，暂停时METR-LA完成95更新，
尚未有完整验证；检查点/优化器/RNG/采样游标及缓存将迁移保留。已核实身份后停止本项目监督器/worker，
两者均退出、CUDA已释放，旧账本39.642秒保留。准备v2将“状态精确恢复”与
“GPU后续数值轨迹近似”分开核验，不据此声称CUDA逐位复现；旧快照保持不变。

固定运行代码 `95e0a0dca295a3f5f09fb329cc90f8378a4c0cb8`，快照
`/home/dcg/dcg_data/GOFA-F-runs/stage4-pure-graph-v1`；
输出开发仓库 `outputs/stage4-pure-graph-v1/`，日志同级 `.log`，启动记录 `.launch.json`。
02:03 CPU等待器PID `1997975`、start_ticks `592204080`，首次状态为waiting_for_idle、
连续空闲1次、活动秒数0，尚无GPU worker；此为历史观察，使用前必须重新核对。
快照与137项CPU核验源码的规范化哈希一致，六集输入及24份历史评价引用绑定成功。
不设GPU累计或单组分钟上限，11:01锁定评价、11:16硬截止；其他任务出现即保存让出。
每小时跟进 `gofa-f` 已更新并启用，只通知终止或异常，完成汇总后暂停。

用户授权夜间自主推进至北京时间11:16，随后取消累计GPU活动及分组分钟上限；
仍须GPU完全空闲才运行，其他任务出现即保存让出，绝不独占。
仅新增纯图UFGTime，HSD仅留历史，暂缓21M总参数匹配。
G2-V关闭源时钟，匹配108,930实际可训练自由度；协议见 `docs/STAGE4_PROTOCOL.md`，
来源/训练差异见 `docs/UFGTIME_REPRODUCTION_AUDIT.md`。
当前开发代码在C隔离目录完成双环境核验：原生环境126通过/11按依赖跳过，
UFG环境22通过/2按Chronos依赖跳过，合并覆盖全部137个独立CPU用例。
六数据集G2-V/self共12次真实Bolt初始预测差均为0；UFG-S/A共12次真实形状前向/梯度有限。
01:57复核最终代码：六集输入、24份历史原始标签评价与完整双环境依赖均已绑定；
可移交CPU证据见 `docs/results/stage4-cpu-verification.json`。
独立 `.venv-ufg` 安装和验证全程禁用CUDA，旧环境/快照/数据未改；尚无新GPU精度结果。
后续须先核对输出与已有进程，避免重复启动；下文Stage3状态为已完成历史。

## 最新状态：Stage 3 已完成（2026-10-14 23:31，Asia/Shanghai）

`stage3-late-context-v1` 于 23:20:28 正常结束，状态与启动器均为 `complete`；
23:31 只读确认 supervisor `1787228`、worker `1795465` 已退出。
固定快照仍为 `0916e8fecf2c6729aa7f23ebfcd7bcab7bd1345e`，不要重复启动。
完成八组关键消融、四组扩展及六数据集测试，验证集选择 G2（11 维上下文＋rank=4）。
G2 六集胜原生 Bolt、五集胜历史限时 HSD，Electricity 仍落后 HSD 8.59%。
活动总账本 6397.47 秒（1:46:37），没有追加剩余预算；每小时跟进 `gofa-f` 已暂停。
完整结果、实际更新数、停止原因和科研边界见 `docs/STAGE3_RESULTS.md`，
可移交摘要见 `docs/results/stage3-late-context-v1.json`。B 已下载所有小型 JSON/Markdown 输出。
本轮执行完成不代表整个项目目标完成。下文等待/运行中状态均为历史观察。

## Stage 3 开发与空闲等待协议（2026-10-14）

21:23 已部署并启动 CPU 等待器，21:25（Asia/Shanghai）只读确认：
`launcher.json.state=waiting_for_idle`，连续空闲计数 0，活动秒数 0；
等待器 PID `1787228` 存活、无 worker 子进程、不在 GPU 进程表中。
GPU 正由其他任务使用。此观察不是实时状态，后续必须重新核对。

- 固定代码：`0916e8fecf2c6729aa7f23ebfcd7bcab7bd1345e`。
- 固定快照：`/home/dcg/dcg_data/GOFA-F-runs/stage3-late-context-v1`。
- 输出目录：开发仓库 `outputs/stage3-late-context-v1/`。
- 启动记录/日志：`outputs/stage3-late-context-v1.launch.json` / `.log`。
- 初始只生成 `launcher.json` 与启动器清单；worker 入场后才生成 `manifest.json`、
  `status.json`、活动账本及 GPU 机制报告。没有 GPU 机制或新精度通过的结论。
- CPU 真实权重记录已保存在开发输出 `outputs/stage3-cpu-realweights.json`，
  校验部署提交与 CPU 验证源码在行尾规范化后完全一致。
- 不从开发 HEAD 重启作业，不重复启动等待器；恢复应使用相同快照和清单。
- 用户已授权每小时跟进，自动化 `gofa-f` 已启用；等待与普通进度保持安静，
  仅终止结果或异常通知，报告终止结果后停用。它不启动、重启或修改 GPU 作业。

用户已批准：新 GOFA-F 固定 late_contiguous，先代码/CPU 验证，GPU 完全空闲才运行。
新活动预算 4 小时，空闲等待不计费，其他任务出现即保存并退出。
实现与规则见 `docs/STAGE3_PROTOCOL.md`；C 上 108 项 CPU 测试及六数据集真实权重
CPU 初始一致性检查已通过。新入口 `scripts/run_adaptation_stage.sh`，GPU 检查尚待空闲。

20:10（Asia/Shanghai）确认 stage2-native-v1 六数据集 24 份评价全部完成，
原 runner/timeout 已退出。账本 14,716 秒，GOFA 原始 MAE 六集胜 native Bolt、四集胜
限时匹配 HSD；METR-LA、Electricity 仍落后。下文“运行中”记录均为历史观察。

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
