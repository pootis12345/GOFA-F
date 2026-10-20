# Stage21 v2 部署状态

AQI-36已按用户授权替换为法国MeteoNet NW2018温度。其余五集及共享3500更新检查点保持不变。

- 完整CPU回归561通过、21旧可选依赖跳过。
- 六集固定768配对查询，共12288条采样轨迹；不训练、不校准。
- 原始单位、时间同步、缺失、节点映射和源目标隔离审计通过；数据细节见固定协议。
- 独立stage21-graph-benchmark-v2快照及输出已建立，监督器健康、waiting_for_idle、尚无GPU预测。
- GPU源病例四项重放与新数据完整/缓存路径检查仍待准入，失败停止，不放宽BF16容差。
- 完全空闲3次×10秒准入，项目40GiB/Torch38GiB及10秒资源审计。
- Stage20、训练和每小时跟进保持停止。旧v1重叠证据保留。

在C中将SNAPSHOT和RUN分别设为本轮快照与输出目录，使用原Python环境查看进度：

    PYTHONPATH="$SNAPSHOT/src" python -m gofa_f.graph_benchmark_progress "$RUN" --watch --interval 10

查看当前阶段和总体ETA时，以launcher.json与progress.json为准。等待期间没有可靠完成时间；
轨迹开始后按已保存的8条路径数值槽估计，额外机制工作与全部GPU活动由activity.json记录，不设硬时限。

本轮只能评估图增强系统的迁移表现。单种子、小样本、目标套件在此前成绩可见后确定、预训练暴露不可核及缺少图消融均需披露。

GitHub外发被自动审批阻止：默认分支及独立分支推送均需用户明确授权完整载荷。B/C实现和部署已完成，未绕过审批。
