# 数据与复现约定

## 数据来源与下载

本项目使用 OpenBMI 的运动想象（MI）部分。原始数据论文覆盖54名受试者、不同日期的两次采集及MI、ERP、SSVEP三类范式；本项目仅选取S01–S30的MI文件，不把原论文的任务分类成绩当作身份识别基线。

- 数据存档：[GigaDB 100542](https://gigadb.org/dataset/100542)，持久标识：[10.5524/100542](https://doi.org/10.5524/100542)。
- 原始论文：Lee, M.-H., et al. (2019). *EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI illiteracy*. GigaScience, 8(5), giz002. [DOI](https://doi.org/10.1093/gigascience/giz002)。
- 作者工具箱：[PatternRecognition/OpenBMI](https://github.com/PatternRecognition/OpenBMI)。本Python项目不依赖安装该MATLAB工具箱。

下载步骤：

1. 打开GigaDB存档，查看文件列表及数据使用条款，选择MI文件；不要下载ERP或SSVEP替代MI。
2. 获取S01–S30各自session1和session2的MAT文件，共60份，保留文件名，例如 `sess01_subj01_EEG_MI.mat`、`sess02_subj01_EEG_MI.mat`。下载服务和目录可能调整，以存档提供的入口为准。
3. 将这60份文件放到仓库 `data/raw/`；若保留下载目录层级，则修改 `examples/*.csv` 的path列。路径相对于对应CSV所在目录解析。
4. 按README转换和生成训练/注册/查询清单。S01–S20两日期训练；S21–S30的S1仅注册、S2仅测试。先核对文件完整性、单位和电极顺序，再训练。

原数据、作者工具箱和本项目代码的授权范围应分别核对；本仓库不重新分发原始数据，也不替原数据提供额外授权。若研究使用这些数据，请引用原始论文与数据存档。

## 输入

原始OpenBMI索引为path,subject_id,session_id，身份及日期为数字。相对路径以索引CSV的目录解析：examples模板中的../data/raw指向仓库data/raw。读取普通MAT，物理单位必须显式给V或uV；不要把设备ADC数值直接当微伏。

处理后清单字段为subject_id,session_id,record_id,path。每份NPZ是一段独立连续记录，含eeg[channels,samples]、sample_rate、channel_names、unit=uV、valid_windows、valid_channel_samples、preprocessing_id。尾部不足完整窗口的采样点舍弃。整条信号无有效片段时报告失败，并保留在评估分母。

自有原始NPZ至少含eeg、sample_rate、channel_names、unit；可附invalid_intervals与原因。使用preprocess子命令生成质量掩码。禁止重复预处理，不能复用别的处理配置的摘要或掩码。

## 身份与日期

先划分身份和完整记录，再切窗。跨日源训练每人需要至少两个session；prepare_manifests.py将目标的session_1只作注册、session_2只作查询，要求目标身份均具备两日，且与训练身份不重叠。模板索引不是独立采集日的数量说明。

OpenBMI的MI_train和MI_test后缀是离线和在线阶段。两者均属于记录中的采集日期，不能直接据后缀把文件当作本任务的训练集和测试集。

## 产物

model.pt用于编码器推理；model.last.pt保存优化器与随机状态；model.history.json保存各轮训练/开发记录；gallery.json绑定权重摘要；final_test.json包含准确率、失败数、逐记录预测、身份拆分与混淆矩阵。更换权重必须重新注册。

精确恢复要求输入清单、文件内容、路径、配置及运行环境兼容；移动历史实验路径后不可直接假定旧恢复状态能继续。仅支持同结构同处理约定的strict初始化；完整训练后用新目标做一次测试，不按目标错误重新选轮次。

自有数据集应使用带数据源前缀的身份ID，避免不同数据集标签碰撞。换电极布局或采样约定应重新训练；新增数据必须有可用的使用权限。仓库不提供数据和预训练权重，示例模板不代表数据许可。
