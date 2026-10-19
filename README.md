# Quality Aware Axial EEG Transformer

用于跨日期 EEG 身份识别的轴向 Transformer。模型保留电极与时间片两个维度，结合硬掩码、软质量注意力和污染一致性训练，将记录编码为128维身份向量，再通过注册模板的余弦相似度识别人。

本仓库只有完整轴向方案，不包含原始EEG、预处理数据、训练权重或注册库。已完成实验为源20人两日期训练、新10人S1注册/S2测试，18/20=90%；它是单数据集、单种子、20条相关阶段记录的结果，不代表跨数据集普遍性能。详见[结果与限制](docs/RESULTS.md)。

## 本地项目

当前目录是唯一维护的源码版本，直接用VS Code打开本目录即可。旧上传副本、ZIP和工作备份已移到项目外归档，不需要解压、复制或进入另一套源码目录。

本地Git仍在 `release` 分支，保留原历史；旧云端仓库已删除，失效的远程关联已移除。[重新上传与日常提交](docs/UPLOAD.md)说明如何从VS Code连接新仓库。

## 安装

Python3.10或更高版本。在仓库根目录：

```sh
python -m venv .venv
# macOS或Linux
source .venv/bin/activate
# Windows PowerShell 使用 .venv/Scripts/Activate.ps1
python -m pip install -e ".[test,monitor]"
python -m pytest -q
```

GPU训练需安装适配运行机器的PyTorch；`--device cpu`可验证流程，`--device cuda`使用可用CUDA设备。本项目不自动下载数据或权重。GitHub Actions使用CPU运行机制测试。

## 从数据到测试

1. 按[数据来源与下载步骤](docs/DATA.md#数据来源与下载)获取具有使用权限的OpenBMI运动想象MAT文件，放入data/raw，或修改examples中的CSV路径。模板包含S01–S30两日期的预期文件名，不包含数据。
2. 分别转换源20人和目标10人。显式指定物理单位；默认处理为250Hz、0.5–45Hz、5秒窗和逐通道采样掩码。

```sh
python -m eeg_siamese.cli convert-openbmi --index examples/openbmi_source20.csv --input-unit uV --config configs/preprocessing.quality.json --output-dir data/processed/source20
python -m eeg_siamese.cli convert-openbmi --index examples/openbmi_target10.csv --input-unit uV --config configs/preprocessing.quality.json --output-dir data/processed/target10
```

3. 生成清单。源身份两日期全部用于训练；目标身份只用session_1注册、session_2查询。脚本检查训练/目标身份隔离、注册/查询日期隔离以及每名源身份至少两日期。

```sh
python scripts/prepare_manifests.py --source data/processed/source20/manifest.csv --target data/processed/target10/manifest.csv --output-dir data/splits/source20_target10
```

4. 用转换生成的配置训练、冻结、注册和测试。开始前记录实际运行机器的环境：

```sh
python scripts/capture_environment.py --output artifacts/axial_run/environment.json
```

```sh
python scripts/run_pipeline.py --config data/processed/source20/general.json --train-manifest data/splits/source20_target10/train.csv --enroll-manifest data/splits/source20_target10/enroll.csv --query-manifest data/splits/source20_target10/query.csv --output-dir artifacts/axial_run --device cuda
```

5. 查看当前运行的实际结果和曲线。

```sh
python scripts/plot_results.py --run-dir artifacts/axial_run --tensorboard-dir artifacts/tensorboard
python -m tensorboard.main --logdir artifacts/tensorboard --host 127.0.0.1 --port 6006
```

浏览器打开http://127.0.0.1:6006/。最终测试准确率是一个固定评估点，不是每轮目标测试曲线。脚本不会把示例90%成绩写入新实验。

## 结构

```text
configs/                  用户可编辑的模型与预处理配置
examples/                 相对路径的数据索引模板
src/eeg_siamese/           轴向模型、质量、数据、训练和推理
src/eeg_siamese/configs/   随Python包分发的默认配置模板
scripts/                  清单划分、训练注册测试、可视化
tests/                   信号与训练流程回归测试
docs/                    模型说明、输入约定和实验结果
.github/workflows/        CPU测试与打包检查
```

本地运行产物在 `artifacts/`，完整汇报在 `reports/`，原始或处理数据按需放在 `data/`，Python环境在 `.venv/`；这些目录均被Git忽略。源码和说明文档以当前目录中的版本为准。模型配置使用62电极是当前OpenBMI实验设置；换数据集须核对单位、采样率、电极名称顺序和日期协议，并重新训练，不强行加载不兼容权重。

## 更多说明

- [模型原理与训练目标](docs/MODEL.md)
- [数据约定和复现边界](docs/DATA.md)
- [实验结果与TensorBoard图](docs/RESULTS.md)
- [环境版本与复现边界](docs/ENVIRONMENT.md)
- [更新上传与仓库设置](docs/UPLOAD.md)

默认35轮来自源身份五折开发选择。最终目标S2不用于选权重、早停、调参或阈值；若依据目标错误调整方案，必须另留测试数据。修改默认配置时，同步configs与src/eeg_siamese/configs中的模板，测试会检查一致性。

## 授权状态

按所有者选择，暂不提供开源许可证或额外开源使用授权。原始数据及第三方依赖遵循各自条款。
