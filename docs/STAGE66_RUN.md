# 独立可变长度图专家进度

阶段目标为源训练的共享图专家在固定六集零样本宏MAE超过共享UFG。尚无本轮完整目标成绩，未宣称目标完成。

源状态：outputs/stage66-variable-graph-source-v1/{expert,ufg}/status.json、activity.json、all/selection.json、all/curve.jsonl。协调状态为同根pipeline_status.json；目标结果位于outputs/stage66-variable-graph-targets-v1。一个模型全部长度共享同一state_dict，不按形状选模型。

```sh
GOFA_WORKSPACE="$PWD" PYTHONPATH="$PWD/src" "$PWD/.venv-ufg/bin/python" -m gofa_f.variable_graph_source --config "$PWD/configs/stage66_variable_graph_source.yaml" --output "$PWD/outputs/stage66-variable-graph-source-v1" --watch
```

先专家、后新多长度共享UFG、锁定后自动完成主768和补充192目标查询，原生主任务8采样只读复用。ETA到固定上限，早停可能缩短；不能将最初若干秒的QA速率当最终精确耗时。上限未满足充分训练早停要披露，不算完成UFG充分性要求。

测试43项通过，真实四源32条混合长度反向：专家0.775秒、UFG0.970秒，均CPU两线程、CUDA未初始化。QA两更新/保存恢复均通过，状态不移交正式训练。源/目标清单、代码、单模型初始化SHA和小型证据保留。详细服务路径和进程身份仅保存在B/C证据，不写入公开结果。

现有Stage60 GOFA-F源训练及Stage65两主任务共享UFG保持原固定快照健康运行；本阶段不会重启或覆盖它们。跨种子稳定性与GOFA-F融合尚待独立专家实测达标后推进。
