# 查看共享UFG训练与六目标重测

本轮只训练一套共享参数、一个优化器、一个源最佳；12→12和48→24使用同一模型。旧两形状权重不作为当前合格对照，原结果仍保留。

在C查看源训练和后续目标进度：

```sh
GOFA_WORKSPACE="$PWD" PYTHONPATH="$PWD/src" "$PWD/.venv-ufg/bin/python" -m gofa_f.ufg_shared_source --config "$PWD/configs/stage65_shared_ufg_source.yaml" --output "$PWD/outputs/stage65-shared-ufg-source-v1" --watch
```

ETA在10次正式更新后依据实测修正，包含剩余源验证；最终CPU目标重测另列，不包含未知资源或I/O变化。触发源早停可提前结束，达到上限不宣称已收敛。源完成后控制器自动对六集768条查询重测共享UFG，原生与TSGFM预测只读复用，不新增GPU。

输出分别为stage65-shared-ufg-source-v1及stage65-shared-ufg-targets-v1的status/activity/manifest、源all/selection与曲线、目标逐查询预测及完整report。终止后ETA不适用。每小时跟进保持关闭。

实际已部署并核实健康：运行源码ebd184a330d6cce7d1fcc0411d985426e84dbb2d，清单SHA5e8617b702657e4676425035e7c098669361d014446df0e76f542b19eb9bdd38。启动独立审计在154更新核验Adam步骤154、优化器组1、全部8个来源/形状池均已参与；同一模型树，初始张量SHA与QA精确相同。源确认和新目标结果尚未产生；旧对照后接已退出，GOFA-F源训练继续。详情和启动证据留B/C。
