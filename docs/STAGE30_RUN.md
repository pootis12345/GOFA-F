# Stage30自主实验窗口与当前运行

用户已授权北京时间2026-10-22 00:15:12起的11小时自主迭代，并明确允许临时恢复自动跟进。固定截止11:15:12；新GPU工作在截止前60秒停止，之后仅CPU/只读汇总。普通进度安静，完成、截止、异常或需用户处理时通知。

Stage29仍为开发门槛失败的封存历史。Stage30采用[预注册有限方案](STAGE30_AUTONOMOUS_PROTOCOL.md)，不改六集、冻结测试样本、父3500、八采样、数值核或验收容差。每集8种CPU方案、3档既定正则，使用前60%目标标签训练和旧开发记录选择；不是零样本GOFA-F。全部候选及失败记录保留，确认与测试不反馈调参。

当前固定运行源码`a8741e1a0ad1c3d45e87d702ebc95c2a99be2954`，独立快照/输出名`stage30-autonomous-iteration-v2`。v1在父阶段退出检查遇到CPU-only结果没有GPU启动器文件的实现错误，未训练候选或创建CUDA上下文；已核实退出并保留日志，v2增加对应回归后重新部署。

v2完整CPU回归685通过、21旧可选依赖跳过，CUDA未初始化。已开始有限CPU候选训练与前向时间开发验证，尚无完整六集开发门槛或独立确认结果。显存限制继续memory_cap_only，项目40GiB/Torch38GiB、10秒审计。

若六集开发门槛全部通过，封存候选与系数，再按顺序进行未使用的独立确认，全部确认通过才一次CPU重放冻结测试。任何确认失败均如实停止，不用其标签继续优化。只有实际最终测试六集严格改善，才开始剩余时间内可完成的公平比较实验。

本地尚无LLaMA-7B权重。必须核验准确版本与合法访问，不以已有Mistral/Qwen或ChatTime冒充。LLaMA及GNN比较不预设排序、不弱化基线；源训练零样本模型与目标监督混合系统分开报告。若无法完成，结果留空并说明原因。

进度可在服务器项目目录查看：

```bash
PYTHONPATH="$PWD/src" .venv-chat/bin/python -m gofa_f.autonomous_iteration --config configs/stage30_autonomous_iteration.yaml --output outputs/stage30-autonomous-iteration-v2 --phase progress
```

终止状态优先；未完成六集确认和冻结测试前不能宣称目标达成。
