# Stage32结果与Stage33有限后续

Stage32已正常止损，原因`failed_confirmation_sdwpf`。新的64条SDWPF确认中，原生宏MAE221.662693，校准系统232.290438（+4.79%）；37条改善、27条退化，但较大退化抵消了改善。CPU单独232.605338。原生/CPU融合的图权重为0，与无图CPU对照一致，不能宣称图LLM增益。

8个预注册候选全部完成，按最差开发折规则选中norm_hgb_local。开发两折约改善12.2%、12.7%，不能外推为确认改善。没有发现执行或LP故障，机制通过，未运行Kelmarsh或冻结测试。封存系数SHA`77a0da9b07cadc93052ec8a319a3aa0b8c71bb04ba2379f132161772f0b2275b`不变。

Stage32新增GPU活动480.769474秒，约8.01分钟；继承Stage31后累计3367.833412秒。CPU候选活动997.067231秒另计，峰14362MiB，等待/资源暂停0，已核实本项目控制器与GPU worker退出。没有实现故障重启或返选第二候选。

## Stage33预注册后续

剩余授权窗口内开展一个固定的历史SCADA方案，详见[协议](STAGE33_CAUSAL_SCADA_PROTOCOL.md)。只为SDWPF新增过去48点风速、温度、桨距、无功功率、方向信息的CPU特征，所有未来字段与ERA5禁用；其他五集参数精确冻结。原始发布字段已核验，隔离PyArrow17读取目录不会改动旧环境或数据。

四个固定HGB候选，共用Stage32的16384条前60%训练查询。Stage31/32失败确认明确转为已暴露开发数据，与原128条合并为256条，用三个前向时间折选择；不将其继续称独立确认。训练前封存同64变量的全新、不重叠确认起点。六确认全部通过才一次CPU冻结测试重放。

B静态和C禁CUDA两线程专项10项通过。固定快照完整CPU698通过、21旧可选依赖跳过（719条XML），CUDA未初始化。原始SCADA的时间、节点和功率已与冻结Stage21数据精确对齐，未改变目标掩码；派生数据哈希和新64条确认在拟合前封存。正在四个CPU候选拟合，首个residual_local已完成并通过预设开发门槛，第二个进行中；尚未选出最终候选、未运行新确认或冻结测试，不能宣称六集改善。固定截止仍为北京时间2026-10-22 11:15:12，memory_cap_only、40GiB/38GiB、10秒审计不变。

这是目标监督、额外历史SCADA输入的混合系统，非零样本GOFA-F，不能作为等输入的图LLM贡献证明。所有失败与成本保留，不削弱比较基线以保证排序。


固定源码为`3695c7634d2a2e94324d6afab3875cb37210dec1`。manifest SHA为`37d33deba228d0a8cef6e098bbf6709835ac9aacfc60bd9f2774272fb3a80398`；新确认清单SHA为`f487e895ee52d02691deba73fb22509b1bb34373c6dc481214f30ebfc7251320`。详细进程身份保留本地，不发布。

查看进度（在项目目录）：

```bash
PYTHONPATH="$PWD/src" .venv-chat/bin/python -m gofa_f.scada_wind_iteration --config configs/stage33_causal_scada.yaml --output outputs/stage33-causal-scada-v1 --phase progress
```

CPU准备和候选拟合成本独立记录；GPU账本继承3367.833412秒，Stage33新增GPU应减去`inherited_prior_seconds`。未生成的确认和测试留空。
