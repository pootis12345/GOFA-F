# 已准备的数据版本

全部数据在 H100 服务器 `data/processed/`，原始包与逐文件 SHA256 在 `data/raw/`。
每个 prepared NPZ 旁的 `.metadata.json` 记录来源、形状、采样率、自然缺失比例和哈希。

|领域|名称|时间步数|节点数|采样间隔|
|---|---|---:|---:|---|
|交通|METR-LA|34,272|207|5 分钟|
|交通|PeMS-BAY|52,128|325|5 分钟|
|环境|AQI / PM2.5-437|8,760|437|1 小时|
|环境|AQI-36 / PM2.5-36|8,759|36|1 小时|
|能源|Electricity|26,304|321|1 小时|
|能源|Solar-Energy|52,560|137|10 分钟|

交通和环境源版本为 TorchSpatiotemporal/ProcessedDatasets 提交
`e7b92503354c28146e16bcb89bf75b7435b93808`。
能源源版本为 laiguokun/multivariate-time-series-data 提交
`7f402f185cc2435b5e66aed13a3b560ed142e023`。

环境数据使用原 HDF 的 `pm25` 浓度键，保留 NaN 对应的缺失标记。
交通数据中的零值按该 benchmark 的惯例作为缺失；能源数据的真实零值保留。
AQI / AQI-36 有重叠，不能作为独立跨域迁移证据。
AQI 训练段没有观测的站点仍保留，用训练段的跨站点统计量回退标准化。

已准备数据并不表示正式实验已完成。主表只能从完整训练的 `result.json` 生成。
