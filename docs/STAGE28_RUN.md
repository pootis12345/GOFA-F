# Stage28运行与进度

本轮已完成实现、B静态检查、C禁CUDA两线程完整CPU回归：666通过、21旧可选依赖跳过。
固定运行源码为`49f1a3164132444027d1997c882f7002378eb264`，独立输出`stage28-state-residual-calibration-v1`。

北京时间2026-10-21 21:55核验：四项真实CUDA源病例精确重放通过；新增SDWPF原生完整/增量路径差异0、采样token一致。
正在生成新增64条开发查询的原生预测，已保存8/64条查询中位数。图模型开发预测与其新路径检查尚未执行，不能将原生通过视为全部验收完成。
本项目观测峰14306MiB，未超限、无资源暂停或失败。启动初期按共享GPU实测速率估算总体约3小时，包含门槛通过后的确认；估计仍不稳定，并非截止时间。

在服务器项目目录查看进度：

```bash
PYTHONPATH="$PWD/src" .venv-chat/bin/python -m gofa_f.state_residual_progress outputs/stage28-state-residual-calibration-v1 --watch
```

进度包含当前数据集/模型、已完成查询、采样编号、阶段和条件总体ETA、累计GPU活动及显存峰值。
本轮没有训练epoch或优化器更新。恢复仅补缺失轨迹，活动账本不清零；终止后ETA不适用。

执行顺序固定：新增SDWPF开发预测 → CPU两折门槛 → 全128条重拟合与非恒等检查 → 参数封存 → 六集确认 → 仅CPU测试重放。
任一门槛失败均封存结束，不自动追加方案。五集校准器固定，父3500与原生模型不更新。
资源采用memory_cap_only共享准入，项目40GiB/Torch38GiB、10秒审计；每小时跟进保持关闭。

这不是六集已改善的结果报告。系统属于目标验证校准后的开发重评，既有目标测试已查看；结果不能称严格零样本或独立盲测。

[固定协议](STAGE28_PROTOCOL.md)与[脱敏部署证据](results/stage28-deployment-public-summary.json)。
