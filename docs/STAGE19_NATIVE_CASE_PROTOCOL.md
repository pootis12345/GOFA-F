# 原生ChatTime固定NN5病例检查

用户要求核对是否原生生成8次，并以原生流程测试此前重复病例。
本轮不恢复Stage19训练，1177更新/pending30/390标签与last SHA
a37c4f4e793d073d20c8bc2af9ba437977a068f953518e134fd4794ac278cd3b保持不变。

## 官方流程核验

固定ChatTime源码8c2d4c209302d2b2bd6cc3c154586842341e3247与已下载ChatTime-1-7B-Chat权重20baa88。
官方model/model.py默认max_pred_len16、num_samples8、top_k100、top_p1、temperature1、FP16。
每块使用do_sample=True返回8条，解析/反变换后逐步nanmedian；滚动时将该块中位数接到历史并重新MinMax。
24步是16+8两块，各采样8条，并非8条完整24步独立轨迹。
官方代码链接：https://github.com/ForestsKing/ChatTime/blob/8c2d4c209302d2b2bd6cc3c154586842341e3247/model/model.py

已通过GitHub固定提交tree API核验3个已下载文件的Git blob，运行前再次强制核验：
model/model.py af932802db789a7293b2a4bf35eec1877b8db379；
utils/prompt.py 89f65f4609266f0dde0e2bce9ba00ece9850f61f；
utils/tools.py ec41d9b88296a197d12fb7d811f4f8e6afe7d79e。
external目录不是独立Git checkout，git -C会回溯到项目仓库，故不以该命令输出作为官方版本证据。

## 固定病例与对照

NN5变量92、起点568、48→24，严格使用原验证历史520:568，未来仅用于诊断误差，不参与输入。
不加载任何已训练图参数，直接调用官方ChatTime构造器和未修改的predict，保留默认FP16设备映射。
单个seed42，不挑选采样种子，不使用重复惩罚或有限数值强制输出，不增加训练或架构候选。

1. official_sample8：完整官方默认采样8/中位数/16+8滚动。
2. official_greedy1：同FP16、官方提示、滚动和解析，仅改do_sample=False、单条，定位采样策略影响。
3. stage19_prompt_greedy1：同第2项，仅将提示格式替换为当前Stage19的字符串，定位提示空行影响。

代码核对发现当前Stage19被命名为OFFICIAL_PROMPT的字符串，与官方模板在system→Instruction及Input→Response之间
少了两处空行。这与此前“官方提示完全一致”的描述不符，作为实际协议差异单列。
本次不原地修改Stage19模板、缓存键、训练或旧对照，所有既有结果保持其原协议解释。
第2/3项是机制对照，不是作者默认设置，不能将其称为官方8样本结果。
第3项仍使用官方数值序列化/FP16/滚动，所以与旧Stage19 BF16受约束单块24还不完全相同。

观测包装器仅捕获pipeline返回，完整保留每块8条原始回答、解析数值、有效性/重复、缩放参数、逐步中位数与最终24步。
包装器返回原samples，且断言捕获中位数与原predict最终输出完全相同；结束后恢复官方函数引用。
报告各块恒值样本数/8、中位数不同值数量、最长重复段、有效预测覆盖及MAE；无效值保留，不套用Stage19回退。

## 执行约束

独立快照/输出stage19-native-case-probe-v1；代码入口gofa_f.short_native_case_probe。
B静态、C禁CUDA OMP/MKL各2线程，官方blob/解析/滚动捕获单测与完整CPU回归，代码和输入绑定后部署。
既有环境已具备accelerate/Transformers/FP16依赖，不改旧环境。
CPU监督器每10秒连续3次完全空闲（无计算/图形进程、利用率0、used≤256MiB、固定UUID/Default）后准入。
40GiB项目组/38GiB Torch、10秒审计及合作停止不变；不与其他本项目GPU任务重叠，不操作他人进程。
活动及等待单独记账；检查后退出GPU，原训练仍暂停，每小时跟进仍关闭。

这是一条已看源验证病例的定性诊断，不证明数据集整体性能、表4复现或图模块总体贡献。
官方主测试与旧Stage19还存在精度、提示、采样、分块和缩放差异，不能把所有变化都归于采样次数。
