# W：对数功率残差轴向 Transformer 脑纹识别

发布版本 **1.2.0**。W保留原质量感知轴向Transformer的波形通路，在每个窗口的隐藏特征上加入对数功率残差，再沿用质量加权池化、128维身份向量和余弦匹配。

训练主线是 **Q预训练 → W继续训练**。新增谱分支末层为零，W起点与Q预测等价；波形主干与谱分支共同训练。可同时训练等新增预算的原模型续训对照U。训练统计只拟合训练清单中的记录，注册和查询不参与拟合。

## 安装与验证

Python 3.10+，在仓库根目录执行：

```sh
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell：.venv/Scripts/Activate.ps1
python -m pip install -e ".[test,monitor]"
python -m pytest -q
python -m eeg_siamese.cli train-w --help
```

GPU训练使用已安装的兼容PyTorch和`--device cuda`，CPU流程验证用`--device cpu`。项目不自动下载EEG或权重。

## 训练W

先按[数据约定](docs/DATA.md)将OpenBMI源身份转换为带单位、通道顺序、采样率与硬掩码的NPZ，训练CSV格式为`subject_id,session_id,record_id,path`。相对数据路径以CSV目录解析。源身份必须与后续未见身份评估人员分离；跨日期训练时每个源身份要有两日期。

```sh
python -m eeg_siamese.cli train-w --manifest data/splits/train.csv --config configs/w_openbmi.json --output-dir artifacts/w_run --device cuda --control
```

默认Q训练35轮，U/W分别从同一Q起点新增35轮，每轮100步、4人×2个跨日期样本、SupCon加0.5污染一致性。`--control`额外训练U；省略则仅训练Q与W。阶段轮数由配置`training.epochs`指定，`--warmup-epochs`可单独设置Q轮数。命令只训练，不自动注册、测试或选择最好种子。输出目录须为空，避免覆盖已有模型。

主要产物：

- `artifacts/w_run/model.pt`：可直接用于原注册/预测命令的W单编码器；不含训练包装器中的冻结Q或优化器。
- `Q/`、`W/`、可选`U/`：固定终点模型、完整恢复状态、各阶段实际配置及日志。
- `spectral_statistics.json`：训练记录来源哈希、逐特征统计与拟合方式。
- `COMPLETE.json`：训练预算、输入清单/配置和最终模型哈希；不填造测试准确率。

[详细W流程与断点恢复](docs/W_MODEL.md)。原`train`、注册、校准、预测和评估入口继续可用。`train-w`是两阶段流程，不能将尚未拟合统计的`w_openbmi.json`直接交给旧`train`入口。

## 注册与评估

```sh
python -m eeg_siamese.cli enroll --checkpoint artifacts/w_run/model.pt --manifest data/splits/enroll.csv --output artifacts/w_run/gallery.json --device cuda
python -m eeg_siamese.cli evaluate --checkpoint artifacts/w_run/model.pt --gallery artifacts/w_run/gallery.json --manifest data/splits/query.csv --output artifacts/w_run/evaluation.json --closed-set --device cuda
```

这是闭集身份识别。开集拒识仍须按原流程使用独立开发人员校准阈值，不能拿闭集准确率代替开集能力。每次评估前冻结检查点与协议。

## 已有证据

OpenBMI源20人五折跨日期复核、三个种子的固定终点均值：Q **90.00%**，等新增预算U **91.67%**，W **97.50%**；主要比较W−U为+5.83个百分点，按人区间[0,14.17]个百分点。源20曾用于开发，协议是每折1:4、完整阶段记录汇总；不是新最终10人的成绩，也不能与不同图库规模或32秒方案直接排名。

EEGMMIDB冻结复评中Q为80.26%、W为87.50%，该19人此前也已分析过。两个数据来源都支持保留W候选，但不证明任意数据集或学校干电极设备都能获得同样收益。[完整结果边界](docs/W_RESULTS.md)。

## 版本与目录

当前GitHub发布线为`release`，1.2.0在1.1基础上加入W训练和导出流程。原质量感知Q配置与命令保留；W使用独立的`train-w`配方，配置中的数据协议须与自己的信号一致。[更新记录](CHANGELOG.md) · [提交和推送](docs/UPLOAD.md)。

`src/eeg_siamese/`为核心实现，`configs/`及包内`configs/`为同步模板，`examples/`为相对路径索引，`scripts/`为通用工具，`tests/`为机制和流程测试。已验证研究核心模块保留以兼容现有检查点，常用W流程通过上述入口执行；研究用Windows训练队列不包含在通用发布入口中。

数据、模型权重、注册库、环境和运行日志留在被忽略的`data/`、`artifacts/`、`reports/`及`.venv/`。Git版本提供算法实现，不包含一个已完成全源训练的新最终权重；现有五折权重也不是全20人训练的部署模型。

按所有者选择，暂不提供开源许可证或额外开源使用授权。数据和第三方依赖遵循各自条款。
