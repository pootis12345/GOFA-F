# 运行交接记录

## Stage16 用户再次授权部署至今晚23:30（2026-10-17，Asia/Shanghai）

用户在v11墙钟截止交付后明确要求：“重新部署，硬截止时间设为今晚23：30”。本次新授权覆盖此前11:00截止和“交付后不自动重启”，新训练截止 **2026-10-17 21:30**、硬截止 **23:30**，仍保留两小时评价。原始2026-10-16 16:03:02起算、9956.406928363256秒活动账本、167更新/offset10737及pending49窗口117586标签均不重置。普通恢复不得再次延期。

使用独立快照/输出 `stage16-chattime-metr-v12`，科学模型、数据、图、权重、优化器、原生解码、KV实现和数值容差不变。新接续只在显式 `--authorized-deadline-extension` 下接受已到wall_deadline且全部进程退出的CPU等待源；必须无机制或KV诊断产物、无失败、无已选GOFA或预测、checkpoint SHA和累计活动均与来源migration一致。迁移保留模型/优化器/未完成梯度/RNG/游标/历史/峰值，不复制源final_selection或零完成汇总产物。旧v11终止报告和快照完整保留。

B静态、C Ruff通过；C禁CUDA/OMP/MKL各2线程，65项恢复/阶段/启动器专项通过。新增长期等待截止源的显式授权、检查点/账本未变、无预测、活进程/活动scope拒绝、真实到期与精确状态迁移测试。提交后创建v12固定快照，CPU完整回归和真实7B复核通过才自动部署等待器；实际身份/源码/清单/恢复结果另行补记，当前不能把CPU专项当CUDA成功。

GPU仍须完全空闲连续三次、每10秒检查，无计算或图形应用、util0、已用≤256MiB、Default；未知重置等待。运行中他人加入不退出，低余量暂停，项目40GiB/Torch38GiB/至少留16GiB，不独占、不操作他人进程。沿用每小时跟进、正常等待/训练静默；此前已授权失败时在既定截止内自主安全修复，不放宽容差或增加候选。下方v11已终止记录是历史，当前部署以v12实际文件为准。

## Stage16 v11 已终止：未获GPU完全空闲准入（2026-10-17 11:16，Asia/Shanghai）

本条覆盖以下所有“运行中/等待中”描述。v11在10:59:00按硬截止前60秒规则标记 `wall_deadline`，bootstrap10:59:12完成CPU汇总并退出。PID316422/start_ticks608709160、318760/start_ticks608724854均已退出，扫描无本快照进程，worker与active_scope均为空。09:00后未补训，11:00后没有GPU作业，不再自动部署修复或新阶段。

v11一直未满足完全空闲三次/10秒准入，没有新GPU机制/训练/评价，新增更新及活动均0；终止检查其他任务仍占15,687MiB，即使util0也不满足准入。CPU321通过/11旧可选跳过，真实7B CPU差0仍不代表CUDA修复成功。v10失败证据保留；v11没有新增失败，不将资源等待误记为可重启的实现故障。

167更新、epoch0 offset10737/22969（46.7456%）、pending49窗口117586标签及模型/优化器/梯度/RNG/游标保留。last.pt SHA `4a262c97c0a813032f15e885ca74d2b98cffd27654d74582c30ba23cff3cc1d1` 与迁移时一致；原始16:03:02起点、09:00/11:00截止和9956.406928363256秒累计账本不变。没有完整验证，gofa=null；原生验证0/3281、两组测试0/6313，所有正式精度及生成/回退比例留空。历史训练/显存值不能当作v11新增活动。

固定源码仍 `9bb224f70f3fc06ef5062a94b9a520d4f6ba5bb7`，快照 `/home/dcg/dcg_data/GOFA-F-runs/stage16-chattime-metr-v11` 不修改。结果 `docs/STAGE16_RESULTS.md`，终止证据 `docs/results/stage16-chattime-metr-v11-terminal.json`，B已有小型report副本；代码及文档同步GitHub/B/C，A可拉取继续。本次仅归档，跟进 `gofa-f-stage16` 在交付后停用。未完成ChatTime对照与初始研究目标，不声称TITAN差距缩小。下方均是历史。

## Stage16 夜间失败自主修复授权（2026-10-17，Asia/Shanghai）

用户最新明确要求：“我要睡觉了，若本轮失败则自行进行修复并重新部署”。本条覆盖此前“失败后仅报告并停止”的跟进规则：失败GPU子任务仍须安全停止并保留证据，但在既定窗口内允许自主修复、重新CPU验证、提交同步、新建固定快照后经完全空闲准入部署，不等待用户再次确认；不能原样无限重启、关闭验收或放宽容差。

范围仍是Stage16同模型/数据/图/优化协议的实现与恢复故障修复，不增加架构、骨干、候选、损失或超参数搜索。原起算与全部检查点/未完成梯度/RNG/游标/验证日志/活动账本/峰值保留。新阶段使用递增v12/v13等独立快照、清单及输出，旧版本不编辑；部署前核实旧本项目进程全部退出，无GPU任务重叠。训练/验证/评价故障分别精确恢复；已有测试预测不能返回训练或据测试返选。

资源策略不变：每个新GPU子任务完全空闲连续三次、每10秒检查，无计算/图形应用、利用率0%、已用≤256MiB、Default；查询未知继续等。运行中他人加入不退出，低余量暂停，项目40GiB/Torch38GiB/留16GiB，不独占、不操作他人进程。**今天10月17日09:00训练截止、11:00硬截止**不顺延；09:00后不补训，11:00后只读/CPU汇总。安全修复不可行或时间不足则如实记录，不以延期绕过。

最新只读检查v11控制器316422/start_ticks608709160、等待器318760/start_ticks608724854身份和存活正确，仍waiting_for_idle；GPU被其他任务使用，无新GPU机制/更新。167次更新和pending49窗口117586标签保持，账本9956.406928363256秒不变，B/C/GitHub一致。`gofa-f-stage16` 每小时跟进已更新：正常等待/训练及可控修复静默，最终终止、无法安全自主修复或需用户处理时通知；可修复的中间失败不删除跟进，重部署后须更新为实际新版本。下方早期失败停止规则均按本条解释。

## Stage16 用户请求修复，v11完整CPU检查中（2026-10-16，Asia/Shanghai）

用户在v10失败通知后明确请求“能否修复”，已接续修复工作。新方案保持原生方形因果SDPA的完整77位置查询形状和张量布局，固定文本Q置零并丢弃其输出，仅动态30token做投影/MLP；模型、权重、图、头、优化器和BF16容差不变。旧短查询/显式掩码与原路径不同是已核实的实现差异，但是否为GPU误差根因仍待实测。协议 `docs/STAGE16_KV_REPAIR.md`，旧v10失败报告和快照完整保留。

源码 `9bb224f70f3fc06ef5062a94b9a520d4f6ba5bb7`，固定快照 `/home/dcg/dcg_data/GOFA-F-runs/stage16-chattime-metr-v11`，输出 `outputs/stage16-chattime-metr-v11/`。CPU控制器2026-10-16 23:53:50启动，PID316422/start_ticks608709160，使用前核对。94项专项CPU、B静态/C Ruff通过；C禁CUDA/OMP/MKL各2线程完整回归321通过、11旧可选跳过，真实7B复核正在进行，通过后自动部署。跟进 `gofa-f-stage16` 已重新启用，仅终止/异常通知。

只接续已退出的preflight单阶段失败且源checkpoint与原migration SHA相同、无预测的v10；167更新/offset10737/未完成49窗口117586标签和所有恢复状态/账本不重置，不复制旧final_selection。继承已授权deadline_amendment，原始2026-10-16 16:03:02起算、**10月17日09:00训练截止/11:00硬截止**保持，不再次延长。GPU仍完全空闲三次每10秒，40GiB/Torch38GiB/16GiB余量不变；GPU检查须fresh和actual continued两项通过才训练，失败仍停止不放宽或自动重试。新增 `fixed_prompt_fresh.json`、`fixed_prompt_continued.json` 保存命名梯度及边界差异诊断，CPU通过不代表修复已在GPU成立。以下v10终止及更早记录均为历史。

23:58部署核验：真实7B BF16 CPU预测/梯度最大差0，完整边界隐藏状态max_abs=0/changed_elements=0，实际走fixed_prompt_kv_v2复用；332条XML记录（321通过/11跳过）、Stage16专项84项。监督器PID318760/start_ticks608724854正常waiting_for_idle。禁CUDA独立比对实际v10/v11检查点，除实现binding外全部值完全一致，账本9956.406928363256秒保留；原始起点/09:00/11:00及10秒/三次/256MiB准入均已核实。尚无新GPU机制结果，GPU有其他任务，不宣称CUDA修复或提速已成功。证据 `docs/results/stage16-cpu-verification-v11.json` 和 `docs/results/stage16-kv-repair-v11.json`。

## Stage16 终止：固定文本KV未通过GPU梯度一致性（2026-10-16 23:33，Asia/Shanghai）

v10已在21:56:39因preflight/fixed_prompt_fresh失败退出，不能再把下方waiting_for_idle当实时状态。GPU梯度比较失败张量11/262144元素不满足预注册BF16 rtol/atol2e-3，异常最大绝对差0.010773658752441406；未放宽、未重启。CPU314通过/11旧可选跳过、真实7B 3节点CPU预测梯度差0，不构成207节点CUDA通过。实际continued缓存检查未执行，缓存提速未成立。

控制器204084/监督器206214/CPU调度worker211329均已核实退出，/proc审计没有本项目v10进程，账本worker=null、active_scope=null。167更新/offset10737（46.7456%）、未完成49窗口117586标签及模型优化器梯度RNG完整保留，v10新增更新0；last.pt SHA `791ca1e2f06fd3a727687e010324e7d96bb21724d461fe3f53dccfdad2e7161d` 与迁移时一致。没有完成epoch或验证，gofa=null，原生验证/测试0/3281和0/6313，GOFA测试0/6313，全部正式精度留空。

监督活动账本9956.406928363256秒；训练9865.833998076618秒沿用v6，v10仅新增preflight9.271920159459114秒。机制/训练跨接续allocated峰值30.1001/18.1828GiB、reserved35.0117/20.2656GiB。训练原状态stop_reason仍为历史training_deadline_or_signal，实际本阶段停止原因为GPU机制失败，不是明早墙钟截止。明早09:00/11:00授权保留在不可变窗口中，但不会自动重启已失败阶段。

结果 `docs/STAGE16_RESULTS.md`，证据 `docs/results/stage16-chattime-metr-v10-terminal.json`。本轮没有ChatTime对照精度，旧Bolt/UFG/HSD仅参考，不补造结果或按测试返选。报告终止后停用gofa-f-stage16跟进；后续GPU恢复需新的明确安排。以下均为历史。
## Stage16 用户延期至2026-10-17 11:00（Asia/Shanghai）

用户明确要求“将硬截止时间改为明天早上11：00”：硬截止改为**2026-10-17 11:00**，沿用两小时锁定评价，训练截止**09:00**。原始2026-10-16 16:03:02起算和累计活动不重置；本次明确授权覆盖旧11小时上限及10月16日21:30/23:30截止，后续普通恢复不能再次延长。完全空闲、三次连续、10秒检测及40GiB/Torch38GiB/16GiB余量保持。

旧v9于21:35因mechanisms missed training deadline标记incomplete退出；bootstrap187208/launcher189376/worker195814均已核实退出，没有新版GPU机制或预测。167更新、offset10737/22969、未完成49窗口117586标签的模型/优化器/梯度/RNG完整保留。新接续只接受已退出、完整保存、无预测的过期检查点，不复制旧final_selection。

运行源码 `0c2cce771447127dc14b6e11f201e46f711cb1f6`，固定快照 `/home/dcg/dcg_data/GOFA-F-runs/stage16-chattime-metr-v10`，输出 `outputs/stage16-chattime-metr-v10/`。CPU控制器北京时间21:46:24启动，PID204084/start_ticks607944509；使用前核对。B静态/C Ruff通过，C禁CUDA/OMP/MKL各2线程314回归通过、11旧可选跳过，真实权重复核正在进行，通过后自动部署。跟进 `gofa-f-stage16` 已指向v10及10月17日09:00/11:00截止。

延期通过显式fresh continuation授权，来源window SHA、原/新截止写入window.deadline_amendment；恢复拒绝改变已登记截止。migration.original_deadline_retained=false表示本次已授权延期，并非丢失状态；original_start_retained和state_restore=exact必须通过。以下均为历史，旧阶段不重启。

21:50实际部署核验：v10真实7B BF16 CPU固定缓存预测/梯度差均0，CUDA未初始化；325条XML记录（314通过/11跳过），Stage16专项77项。监督器PID206214/start_ticks607959440已waiting_for_idle，window训练截止1792198800（10月17日09:00）、硬截止1792206000（11:00）、原start1792137782.879529均正确；外层检测10秒/三次/256MiB并保持yield_on_foreign=false。模型/优化器/梯度/RNG/游标exact，账本9925.884896796197秒保留，v9旧final_selection未复制。来源三个进程均退出；当前另有GPU任务，尚无新GPU机制或精度结果。证据 `docs/results/stage16-cpu-verification-v10.json` 和 `docs/results/stage16-morning-deployment-v10.json`。

## Stage16 用户恢复完全空闲准入、10秒检测（2026-10-16，Asia/Shanghai）

本条覆盖以下共享显存准入：新GPU子任务必须连续三次、每10秒确认无计算/图形应用、利用率0%、已用显存≤256MiB、Default模式。查询失败或状态未知重置计数继续等待。外层等待器与每个GPU子任务均采用10秒，间隔写入window/manifest和launcher_manifest并在恢复时校验。运行中他人加入不退出、低显存暂停、40GiB/Torch38GiB/16GiB余量及不操作他人进程保持。

旧v8监督器176180在确认preflight仅CPU等待、active_scope为空后合作停止；bootstrap134316/worker177359也已核实退出，没有新GPU机制、更新或预测。167更新、offset10737、pending49窗口117586标签的检查点及活动账本保留。新固定源码 `e9407b3f39ad92dd02c067f2b1419f8203fd9f4a`，快照 `/home/dcg/dcg_data/GOFA-F-runs/stage16-chattime-metr-v9`，输出 `outputs/stage16-chattime-metr-v9/`。不修改旧快照或清单。

CPU准备控制器北京时间21:24:42启动，PID187208/start_ticks607814364，使用前核对。B静态/C Ruff通过；C禁CUDA/OMP/MKL各2线程308回归通过、11旧可选跳过，正在完成真实权重CPU复核后自动部署；GPU仍需实际机制验证。跟进 `gofa-f-stage16` 已指向v9与完全空闲10秒策略。原始16:03:02起算及**今天21:30训练截止/23:30硬截止**均不延长。以下均为历史。

21:28部署核验：真实7B BF16 CPU固定缓存预测/梯度差均0，CUDA未初始化；CPU共319条XML记录（308通过/11跳过）、Stage16专项71项。新监督器PID189376/start_ticks607830295已进入waiting_for_idle，window和launcher_manifest检测间隔均10秒，idle_samples=3、idle_max_used_mib=256、yield_on_foreign=false。v8检查点/账本精确迁移，167更新与pending49窗口保留；当前GPU有他人任务，未启动GPU机制/训练。证据 `docs/results/stage16-cpu-verification-v9.json` 和 `docs/results/stage16-idle10-deployment-v9.json`。

## Stage16 固定文本 KV 加速已实现并精确接续，等待共享余量（2026-10-16 21:15，Asia/Shanghai）

用户授权加速第1项：仅缓存冻结前30层中47个固定文本token的KV及边界隐藏状态，继续计算30个动态token并重建完整77位置；图位置、后缀、原生生成、微批量1、有效batch64和科学协议不变。源码 `30943b9acccd5d73c3f0a6a9000b7730e537ecdf`，固定快照 `/home/dcg/dcg_data/GOFA-F-runs/stage16-chattime-metr-v8`，输出 `outputs/stage16-chattime-metr-v8/`。协议 `docs/STAGE16_FIXED_PROMPT_KV.md`。

C禁CUDA/OMP/MKL各2线程306通过、11旧可选跳过（XML317条含跳过），Stage16专项70项，真实7B BF16缓存/原路径预测及梯度最大差均0。v7短形状预填充的CPU数值失败保留，改为原始完整形状预填充一次后通过，未放宽容差。GPU必须另行通过fresh及实际continued checkpoint两项缓存预测/梯度检查，才允许训练；尚不能声称加速已在GPU生效。

21:12核实身份后合作停止v6监督器，bootstrap4135772/launcher4140814/worker4143990/GPU4150553均已退出。最后保存167次更新、epoch0 offset10737/22969（约46.7%）、未完成累积49窗口/117586有效标签。模型、优化器、梯度、Python/NumPy/Torch RNG、游标、验证日志及账本精确接续，监督活动9908.55239078775秒保留；旧结果不覆盖。源checkpoint SHA `2315a930296ca3ddb7fe561c3bb7cb2e5e3db53efcbb1ac3b9244c4a310ebddb`；migration明确state_restore=exact，来源/目标绑定及SHA均保存。

v8 CPU控制器PID134316/start_ticks607430843；监督器PID176180/start_ticks607739521；CPU调度workerPID177359，使用前重新核对。外层共享准入已完成，内层GPU机制检查仍等待；切换期间另一任务占用约30GiB，本项目尚未创建新CUDA上下文。实际free≥56GiB/Default连续三次30秒的共享门槛、40GiB/Torch38GiB/16GiB余量不变，不操作对方任务。

原起算16:03:02、**今天21:30训练截止/23:30硬截止**保留，不因接续/等待延长。当前没有完成epoch或完整验证，不能锁定最佳精度或填GOFA正式测试表。每小时跟进 `gofa-f-stage16` 已更新指向v8。证据 `docs/results/stage16-cpu-verification-v8.json`、`stage16-fixed-kv-migration-v8.json`、`stage16-fixed-kv-window-v8.json`。以下v6与更早部署均为历史，禁止重复启动。

21:17追加禁CUDA实际源/目标检查点逐项审计：除实现binding外完全一致，所有旧进程确实退出。当前另一任务用显存约46.6GiB，实际free33363MiB（约32.6GiB）不足56GiB门槛，v8仍仅CPU等待，未启动GPU机制/训练。证据 `docs/results/stage16-fixed-kv-deployment-v8.json`；等待期间不累计GPU活动，不能给出新加速倍率。

## Stage16 改为显存余量共享准入（2026-10-16，Asia/Shanghai）

用户提供当前pi05_libero推理服务器代码并授权：若不要求功能性独占，则不再等待GPU完全空闲。脚本无独占模式要求，实测约20.3GiB占用/58.8GiB实际空闲，允许共享；共享可能影响对方推理延迟，不保证原80ms耗时或实时性能。判断依据及资源协议 `docs/STAGE16_SHARED_GPU.md`。

新GPU子任务改为连续三次、间隔30秒确认Default模式及**实际free≥56GiB**（项目40GiB+16GiB余量）；允许其他计算/图形进程及非零利用率。查询失败、状态未知或余量不足仍等待，使用实际free而不是total-used。运行中他人加入不退出、低余量暂停更新，Torch38GiB/项目40GiB和16GiB余量保持，不设置独占模式或操作对方进程。下方旧的完全空闲准入已被本条覆盖。

原v5等待器3935816和准备控制器3932279已核实退出，GPU活动0；接续源码 `145789cc218cc5b9f9071ef92ad1ad1a3993909d`，快照 `/home/dcg/dcg_data/GOFA-F-runs/stage16-chattime-metr-v6`，C输出 `outputs/stage16-chattime-metr-v6/`，日志同级 `.log`、身份同级 `.launch.json`。CPU准备控制器PID4135772/start_ticks606715582，共享准入等待器PID4140814/start_ticks606727598，使用前核对身份。完整CPU回归290通过、11旧可选跳过，新增54项全覆盖，真实7B权重CPU复核通过后已部署共享准入；证据 `docs/results/stage16-cpu-verification-v6.json`。自动跟进 `gofa-f-stage16` 已指向v6。

原起算仍16:03:02，**今天2026-10-16训练截止21:30、硬截止23:30**，恢复不得顺延。模型、数据、图、优化器、候选和数值容差未变，GPU机制仍须通过才训练。下方v5及更早记录均为历史，禁止重启旧等待器。

18:27只读复核：共享GPU机制已通过（完整207节点真实形状），缓存/非缓存预测与梯度、初始Persistence、冻结、纯文本绕过、批图隔离、缺失节点、累积梯度和优化器精确恢复均通过，未放宽BF16容差。preflight峰值allocated32,319,703,552字节、reserved37,589,352,448字节（约30.1/35.0GiB）。阶段状态running/train，CPU调度workerPID4143990/start_ticks606735393；活动账本约31.4秒，训练检查点/完整验证以实际progress/history为准。对方openpi-server仍存在。部署及GPU证据 `docs/results/stage16-shared-v6.json`；尚无精度结论。

## Stage16 用户将硬截止提前到2026-10-16 23:30（Asia/Shanghai）

本条覆盖下方旧的10月17日截止记录。用户要求硬截止改为**今天10月16日23:30**，保留两小时锁定评价，因此训练截止**21:30**；原始起算仍为**16:03:02**，接续或恢复不得重置时钟。模型、数据、图、优化器及候选范围不变。

旧v3 CPU等待器3824577/start_ticks605883460及准备控制器3819503已核实退出；停止时GPU活动0、worker为空、没有检查点或预测文件。新入口只允许接续已退出、零GPU活动且不可变输入一致的CPU等待记录，拒绝真实训练/预测产物，并记录来源窗口/清单SHA。CPU汇总产生的零完成评价记录不视为训练产物。v4在这项判断过严时安全停止，未创建窗口或GPU作业，失败记录保留。

接续运行源码 `0d73f05c34dbff3452acb6ed4f80d2a0b1d1c928`，快照 `/home/dcg/dcg_data/GOFA-F-runs/stage16-chattime-metr-v5`，C开发输出 `outputs/stage16-chattime-metr-v5/`、日志同级 `.log`、身份同级 `.launch.json`。CPU准备控制器PID3932279/start_ticks606212644；新等待器PID3935816/start_ticks606223992，使用前核对。C禁CUDA/OMP/MKL各2线程280项回归通过、11旧可选跳过，新增51项全覆盖，真实7B CPU机制复核通过；32项截止/接续专项及真实v3输入/退出身份核验通过。每小时跟进 `gofa-f-stage16` 已更新。

已只读核对v5 `window.json`：原起算16:03:02.879529保留，训练截止21:30:00、硬截止23:30:00均为2026-10-16北京时间；接续来源包含原窗口/清单哈希。新等待器状态waiting_for_idle，GPU活动0、worker为空。证据 `docs/results/stage16-deadline-v5.json` 和 `docs/results/stage16-cpu-verification-v5.json`；此时尚无GPU机制或精度结果。

截止前60秒开始保存退出；新GPU子任务三次间隔30秒严格空闲准入、运行中他人加入不退出、40GiB/Torch38GiB/16GiB余量等共享策略保持。下方v3原窗口和更早阶段均为历史，禁止按旧03:03截止重启。

## Stage16 ChatTime 两组已部署，等待 GPU 空闲（2026-10-16 16:05，Asia/Shanghai）

本轮有新的用户授权，仅运行 METR-LA 12→12、seed42 的原生 ChatTime 和 GOFA-F(ChatTime)，不启动 LLaMA-2、其他数据集或新 GNN。运行源码 `483a876c10b94430409c1d1fc0f882980b77e331`，固定快照 `/home/dcg/dcg_data/GOFA-F-runs/stage16-chattime-metr-v3`，C 开发输出 `outputs/stage16-chattime-metr-v3/`，日志同级 `.log`，等待器身份同级 `.launch.json`，CPU 准备控制器记录同级 `.bootstrap.launch.json`。协议见 `docs/STAGE16_CHATTIME_PROTOCOL.md`。

官方代码及三个权重分片均按固定版本下载并通过官方 SHA256 核验。C 新建 `.venv-chat`，复用原环境依赖只读，新增 sklearn1.5.2/protobuf5.28.3；旧环境、数据和快照未修改。禁 CUDA、OMP/MKL各2线程：270项CPU回归通过、11项旧可选依赖跳过，新增41项全部通过；真实7B权重3节点机制通过，包括原生隐藏状态、Persistence初始预测、缓存/非缓存预测及梯度、图梯度、冻结、纯文本绕过、累积梯度和优化器精确恢复。CPU证据 `docs/results/stage16-cpu-verification.json`，不是GPU机制或精度结论。

CPU准备控制器PID3819503/start_ticks605872007，CPU等待器PID3824577/start_ticks605883460，使用前重新核对身份。首次窗口北京时间 **2026-10-16 16:03:02**，训练截止 **2026-10-17 01:03:02**，硬截止 **03:03:02**；包含等待、恢复不顺延，截止前60秒保存退出。部署观测 `waiting_for_idle`、活动0，GPU已用20837MiB，未获准入、尚无GPU机制/训练/评价结果。新GPU子任务仍连续三次间隔30秒空闲准入；运行中他人加入不退出，低余量暂停更新；40GiB/Torch38GiB/16GiB余量，不设置独占模式、不操作他人进程。

本轮新增可训练1,120,914（图/路由1,071,750，连续头49,164），总注册6,821,464,722；历史UFG-A不再参数匹配。完整前缀缓存估算2,999,480,985,963字节，禁用全盘缓存，使用按需前缀和最多64GiB内存LRU。先完整验证选最佳、锁定SHA后才测试；原生贪心生成与失败点Persistence回退分别报告，未完成全集不填正式测试表。Stage15未完成测试继续留空。

v1缺protobuf、v2发现官方提示模板少一个空行，均在CPU阶段安全停止，未创建窗口；已修复并保留两版日志/快照。v3 CPU通过后才启动本次11小时窗口，部署证据 `docs/results/stage16-deployment-v3.json`。每小时跟进 `gofa-f-stage16` 已启用，仅终止/异常通知；旧 `gofa-f` 已不存在，未重复创建同名任务。本轮结束不自动进入LLaMA-2。以下均为历史记录。

## 自主窗口结束：Stage15验证合格、测试未获准入（2026-10-16 11:02，Asia/Shanghai）

绝对自主窗口10:58已结束，不再启动GPU或Stage16。Stage15训练满100epoch/35,900更新，验证全步3.0626643915、第12步3.6691021727，通过预定门槛；最佳31,592更新的SHA `160eeaf6074a287d8d7380cee66b78dadd4531ffc6d1c2f19e9dc30dd06baf61` 已在09:46锁定。最终评价持续等GPU空闲，未获准入，10:57:02监督器wall_deadline退出，等待器3215096/worker3215767均已核实退出，failures为空。测试指标留空，禁止为补测延长窗口或自动重启。

最近已有测试仍Stage8T2；Stage15验证合格不代表测试差距缩小，0.509%全步验证收益也仅略过门槛。结果 `docs/STAGE15_RESULTS.md`、完整账本/CPU导出已有逐步验证JSON已保存；全夜汇总 `docs/OVERNIGHT_20261016_RESULTS.md`。Stage8–15共10候选/280,020更新，监督器活动合计4.445小时（含Stage8在自主窗口前启动部分）。截止后只读/CPU汇总，随后停止本条跟进；保留C锁定检查点，后续须有新运行授权窗口才能评价。下方部署记录均为历史。

## Stage15 REG路线rank6固定快照和等待器已部署（2026-10-16 09:08，Asia/Shanghai）

运行源码 `42b64818a9874ac12832a9e57a6ed0c4b59093f8`，快照 `/home/dcg/dcg_data/GOFA-F-runs/stage15-metr-reg6-v1`，C开发输出 `outputs/stage15-metr-reg6-v1/`、日志同级 `.log`、身份同级 `.launch.json`。B/C/GitHub同步，C禁CUDA/OMP/MKL各2线程229通过/11可选依赖跳过，固定快照12输入核验通过，旧Stage14进程已退出。

CPU等待器首次北京时间2026-10-16 09:08:48启动，PID3215096/start_ticks603398801，使用前核对身份，禁止重复启动。绝对训练截止10:43、硬截止10:58不顺延。唯一r1_reg_rank6组合REG-only消息与rank6私有头，可训练119,302自由度，协议 `docs/STAGE15_REG_CAPACITY_PROTOCOL.md`；部署时尚无GPU精度。每个新GPU子任务严格空闲准入、40GiB/Torch38GiB/16GiB余量、运行中他人加入不退出仍有效。10:58前仅据验证机制持续顺序迭代。下方准备和旧阶段为历史。

## Stage14未合格，Stage15 REG路线与rank6组合准备部署（2026-10-16，Asia/Shanghai）

Stage14已no_eligible_variant，等待器3145789/worker3146463退出，无失败。M1验证全步/第12步3.0803319922/3.6877857832，M2为3.0667078433/3.6651309559；M2改善约0.378%未过0.5%门槛，均未测试，合格对照仍T2。结果 `docs/STAGE14_RESULTS.md` 和机器可读副本保存，旧阶段不重启。

Stage15固定唯一r1_reg_rank6，检验Stage14 M2路线与预算固定rank6的互补性，从预训练重新初始化，不加其他候选。预注册 `docs/STAGE15_REG_CAPACITY_PROTOCOL.md`，相同验证门槛及10:58绝对截止；准备C禁CUDA回归和独立部署。下方Stage14记录为历史。

## Stage14 token角色固定快照与等待器已部署（2026-10-16 07:08，Asia/Shanghai）

运行源码 `38c1643eb698a6dba42be30e132a7f51c6d0e265`，快照 `/home/dcg/dcg_data/GOFA-F-runs/stage14-metr-tokens-v1`，C开发输出 `outputs/stage14-metr-tokens-v1/`，日志同级 `.log`、身份同级 `.launch.json`。B/C/GitHub同步，C禁CUDA/OMP/MKL各2线程223通过/11可选依赖跳过，固定快照12项输入清单核验通过；Stage13旧等待器和worker退出后才部署。

CPU等待器首次北京时间2026-10-16 07:08:37启动，PID3145789/start_ticks602677691；使用前重新核对身份、禁止重复启动。绝对训练截止10:43、硬截止10:58，等待与恢复不顺延。两组m1_data_only/m2_reg_only只改变图信息进入的token角色，参数与原T2一致，协议 `docs/STAGE14_TOKEN_PROTOCOL.md`。首次观测waiting_for_idle、活动0，尚无GPU机制或精度。新GPU子任务严格三次间隔30秒空闲准入；运行中他人加入不退出，40GiB/Torch38GiB/16GiB余量保留。10:58前继续仅据验证机制顺序迭代。下方准备及旧阶段记录为历史。

## Stage13未合格，Stage14 token角色路线准备部署（2026-10-16，Asia/Shanghai）

Stage13已 `no_eligible_variant`，等待器3103432/worker3104082退出，无失败。验证全步3.0771045153、第12步3.6767182618，仅改善约0.04%，未过全步0.5%门槛，未测试；合格对照仍Stage8T2。结果 `docs/STAGE13_RESULTS.md` 及机器可读副本已保存，旧阶段不重启。

Stage14回到T2/rank4，不增参数，固定两组分别只给数据patch或REG汇总token注入图增量，原生注意力掩码不变。C禁CUDA/OMP/MKL各2线程全仓库223通过/11可选依赖跳过；真实权重、图掩码、缓存和恢复核验通过。预注册 `docs/STAGE14_TOKEN_PROTOCOL.md`，准备同步和固定快照。验证门槛、严格GPU空闲准入、40GiB/16GiB余量及10:58绝对截止保留。下方Stage13部署为历史。

## Stage13 rank6私有头固定快照与等待器已部署（2026-10-16 05:59，Asia/Shanghai）

运行源码 `078c9f5a1abdae79127f388c1d1db56dd7ce4a4f`，快照 `/home/dcg/dcg_data/GOFA-F-runs/stage13-metr-head-v1`，输出C开发仓库 `outputs/stage13-metr-head-v1/`、日志同级 `.log`、身份同级 `.launch.json`。C禁CUDA、OMP/MKL各2线程213项CPU通过/11可选依赖跳过；快照12项输入核验通过，Stage12旧等待器和worker退出后才部署。

CPU等待器首次北京时间2026-10-16 05:59:48启动，PID3103432/start_ticks602264777，使用前核对身份、禁止重复启动。绝对训练截止10:43、硬截止10:58，等待恢复不顺延。唯一h1_rank6保持T2图，仅扩私有头到rank6，真实可训练119,302参数、点预测涉及115,918自由度，处于历史METR-LA UFG-A的±10%内；初始预测/共同图权重配对通过。协议 `docs/STAGE13_HEAD_PROTOCOL.md`，验证门槛仍相对Stage8T2，部署时尚无GPU精度结果。新GPU子任务严格空闲准入，40GiB/Torch38GiB/16GiB余量、运行中他人加入不退出继续有效；10:58前按验证机制持续顺序迭代。下方准备与旧记录为历史。

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
