# Stage61 独立TSGFM适配专家

固定方案见[协议](STAGE61_TSGFM_PROTOCOL.md)。这一步先训练不经过ChatTime的独立图预测专家；无目标训练/评价入口，无新的LLM联合训练。现有Stage60训练在原快照独立运行，本阶段不编辑它。

实现是224003参数的共享时空图专家，不是原来的历史GRU/MLP。包含并行空间/时间注意力、时距嵌入、消息合并及GRUCell，读取完整历史和真实物理子图；多步头、缺失和因果尺度为明确适配。

C禁CUDA、OMP/MKL各2线程的八个实际源形状前反向初测，平均约0.0223秒/查询，各查询保留实际3—12节点；31节点两形状另有公式及全部参数梯度验收。此小样本吞吐不包括每更新保存、验证、共享CPU干扰，正式10次更新后使用实际驻留时间修正ETA，不能据此保证总耗时。正式源最佳与源外推仍未完成，不能称零样本强于UFG。

固定有效batch16/微1、lr1e-3、最多10000更新；每250完整256条源开发验证、至少2000后按八次无显著改善早停。首次全源筛查通过才进入匹配的源分组外推；失败封存，不按目标成绩换配方。

查看当前阶段进度：

```bash
GOFA_WORKSPACE=/home/dcg/dcg_data/GOFA-F \
PYTHONPATH=/home/dcg/dcg_data/GOFA-F/src \
/home/dcg/dcg_data/GOFA-F/.venv-chat/bin/python -m gofa_f.tsgfm_source \
  --config /home/dcg/dcg_data/GOFA-F/configs/stage61_independent_tsgfm.yaml \
  --output /home/dcg/dcg_data/GOFA-F/outputs/stage61-independent-tsgfm-source-v1 --progress
```

ETA仅对应当前独立源拟合，后续源外推、Memory接入和目标评价尚属条件工作；终止后ETA为空。CPU新增成本与并行Stage60分账，共同祖先不重复计费。此阶段的代码/CPU验收/源筛查通过均不构成总目标完成。
