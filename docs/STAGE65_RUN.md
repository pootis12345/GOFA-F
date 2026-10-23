# 查看共享UFG训练与六目标重测

本轮只训练一套共享参数、一个优化器、一个源最佳；12→12和48→24使用同一模型。旧两形状权重不作为当前合格对照，原结果仍保留。

在C查看源训练和后续目标进度：

```sh
PYTHONPATH=/home/dcg/dcg_data/GOFA-F/src GOFA_WORKSPACE=/home/dcg/dcg_data/GOFA-F /home/dcg/dcg_data/GOFA-F/.venv-ufg/bin/python -m gofa_f.ufg_shared_source --config /home/dcg/dcg_data/GOFA-F/configs/stage65_shared_ufg_source.yaml --output /home/dcg/dcg_data/GOFA-F/outputs/stage65-shared-ufg-source-v1 --watch
```

ETA在10次正式更新后依据实测修正，包含剩余源验证；最终CPU目标重测另列，不包含未知资源或I/O变化。触发源早停可提前结束，达到上限不宣称已收敛。源完成后控制器自动对六集768条查询重测共享UFG，原生与TSGFM预测只读复用，不新增GPU。

输出分别为stage65-shared-ufg-source-v1及stage65-shared-ufg-targets-v1的status/activity/manifest、源all/selection与曲线、目标逐查询预测及完整report。终止后ETA不适用。每小时跟进保持关闭。
