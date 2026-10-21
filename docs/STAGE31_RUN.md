# Stage31：开发门槛通过，已封存并开始独立确认

继续用户授权的11小时自主实验，固定截止北京时间2026-10-22 11:15:12，不因修复或阶段切换延期。临时自动跟进已明确获准，普通进度安静。

Stage30已科学止损并封存：PeMS03、EngRAD、SDWPF找到合格开发候选，PEMS-BAY、MeteoNet、Kelmarsh没有；GPU活动0、CPU候选活动419.498秒，没有确认或测试。原v1 CPU退出审计错误的零GPU失败证据保留。

Stage31复用已固定的48个CPU模型，比较较简单融合族，不新增树训练。完整CPU回归688通过、21旧可选依赖跳过。六集均已通过每折不退化且合并至少改善0.5%的固定开发门槛，候选及全开发重拟合参数已经封存。固定运行源码`a1fbc4860a3023d377a0cfbacda6ed8a4a514bd5`，独立输出`stage31-bounded-simplification-v1`。

已经在memory_cap_only共享策略下开始独立确认。四项真实CUDA源病例精确重放通过，未创建梯度；当前项目组观测峰14306MiB，低于40GiB，等待0、资源暂停0。每个新目标路径仍需原始完整/增量logits与采样token验收，不能把源病例通过视为所有路径均完成。

开发选择：

| 数据集 | 方案族 | CPU方案 | 开发MAE | 原生MAE |
|---|---|---|---:|---:|
| pems03 | history_augmented | normseasonextra64 | 22.942656 | 30.902973 |
| pems_bay | history_only | 无CPU | 2.192373 | 2.358938 |
| meteonet_nw2018 | simple | normextra64 | 2.637452 | 3.549398 |
| engrad | history_augmented | normseasonextra64 | 2.137274 | 2.261780 |
| sdwpf | simple | norm150 | 175.808719 | 195.607833 |
| kelmarsh2019 | simple | normextra64 | 228.346910 | 242.845017 |

这些都是已经查看开发样本上的选择结果，不是独立确认或最终测试。现在不能宣称六集目标达成。独立确认任何一集失败就停止本方案，不根据该确认结果回改此方案。只有全部确认通过，才一次CPU重放Stage21冻结测试预测，测试不反馈调参。图权重严格0才跳过新增图预测，指标留空。

封存参数SHA：`e6e0c01103f2b3456c3bf6a69478cd1e54cc68bcca46a21f762e6bd3ed686428`。

这是目标监督及目标验证校准的混合系统，不是零样本GOFA-F或独立盲测。全部结果、公平对照和失利数据集都需保留；不承诺预期的LLaMA/GNN模型排序，不削弱基线。准确LLaMA-7B本地权重尚缺，未完成的比较不能补造。

进度：

```bash
PYTHONPATH="$PWD/src" .venv-chat/bin/python -m gofa_f.bounded_simplification --config configs/stage31_bounded_simplification.yaml --output outputs/stage31-bounded-simplification-v1 --phase progress
```

[固定简化协议](STAGE31_SIMPLIFICATION_PROTOCOL.md)、[脱敏部署证据](results/stage31-deployment-public-summary.json)。
