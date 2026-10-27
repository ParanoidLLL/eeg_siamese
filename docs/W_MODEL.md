# W实现与本地使用

W在每个4秒或5秒窗口的波形隐藏向量h上加入谱残差r：h'=h+r。功率在微伏尺度计算，按1–4/4–8/8–13/13–30/30–40Hz积分，使用log(max(P,1e-12))及训练集固定逐特征标准化。OpenBMI62电极×5带=310维，谱MLP为310→64→64，末层零初始化；EEGMMIDB64电极则320维。它不是全相对功率归一化，原幅度信息仍存在于波形通路和对数功率中。

使用周期Hann单段Welch密度估计，频率积分步长为采样率/窗口样本数。OpenBMI是250Hz完整5秒（0.2Hz分辨率），EEGMMIDB为128Hz完整4秒（0.25Hz）。配置中的采样率、物理缩放和显式spectral_window_seconds必须与数据一致。换数据集需更新通道顺序、窗口、质量patch与预处理ID，重新拟合统计并训练。

软质量分数仍用于注意力/池化；缺失电极-时间单元在谱提取前隔离。一个电极的窗口内有硬失效patch，就不将该电极该窗口计作有效谱观测，不能把补零误作物理低功率。每条训练记录先按电极有效窗平均log功率，再用训练记录向量拟合每特征均值/标准差；极小标准差设为1。注册和查询仅使用已存常数。

`train-w`先训练原Q，再从同一Q权重初始化W以及可选的等预算U。训练期间复用经实验验证的single包装器：冻结Q仅保留来源/初始化一致性，识别输出只来自可训练专家，不混合两个身份向量，也没有自适应门控或额外专家身份损失。污染视图重新计算谱特征。最后导出active W单编码器，注册/推理不再携带冻结Q。

旧Q配置默认关闭谱分支。`configs/w_openbmi.json`与包内同名文件是待拟合配方，不携带虚假的零均值/单位方差训练统计；用train-w拟合后，在各阶段配置和最终checkpoint中保存真实常数。

## 恢复与已有检查点

一键配方拒绝复用非空输出目录，避免不明确地覆盖/混合阶段。被中断阶段可在核对来源后用通用train恢复：

```sh
python -m eeg_siamese.cli train --manifest data/splits/train.csv --config artifacts/w_run/W/config.json --output artifacts/w_run/W/model.pt --resume artifacts/w_run/W/model.last.pt --device cuda
```

恢复需要相同训练清单、未改变信号/配置及相同运行环境；记录完整优化器和随机状态。完成后导出：

```sh
python -m eeg_siamese.cli export-w --checkpoint artifacts/w_run/W/model.pt --output artifacts/w_run/model.pt
```

export-w也支持原研究代码保存的未调制W single检查点，拒绝把F/D双专家融合或C调制模型误标成W。它只提取active编码器并保留来源哈希与人员来源，不变更训练权重或读取测试信号。输出路径必须不存在。

本地整理不是一次新的真实EEG训练，不改变历史研究检查点或评价记录。导出的现有五折模型只训练过相应16人，其性能不能冒充全20人最终模型的性能。模型架构迁移与冻结权重跨设备直接迁移是不同实验。
