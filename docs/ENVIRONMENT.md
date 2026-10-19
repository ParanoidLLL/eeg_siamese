# 环境与复现

## 已验证范围

仓库保留可安装的依赖范围，便于不同机器安装；它不等于精确锁定历史训练环境。

- 历史90%结果：Windows、RTX 5070 Laptop GPU、seed42、35轮；配置、数据划分见[模型](MODEL.md)和[结果](RESULTS.md)。当时未保存完整Python/PyTorch/CUDA/驱动版本清单，不能补写猜测值，也不能承诺逐位重现。
- 当前Mac验证环境：[版本快照](environments/macos-validation-20261019.json)。这是2026-10-19采集的本地环境，用于测试和安装验证，不是Windows历史训练环境。
- GitHub CI：Ubuntu、Python3.11、CPU PyTorch；每次实际版本随该次Actions产物保存。CI验证软件流程，不验证GPU性能或90%成绩。

## 新实验开始前保存环境

在实际运行训练的机器、实际Python环境中执行：

```sh
python scripts/capture_environment.py --output artifacts/axial_run/environment.json
```

脚本记录Python、操作系统、已安装包版本、PyTorch编译CUDA运行时、cuDNN和可见GPU；不导出用户名、主机名、环境变量、包安装路径或凭据。已有快照不会被覆盖；新实验使用新的输出目录。安装包列表是版本记录，不是经验证的跨平台安装锁文件。

Windows GPU机器另外执行以下命令，将驱动信息留在本地实验目录：

```sh
nvidia-smi --query-gpu=name,driver_version --format=csv
```

PyTorch的CUDA运行时、系统CUDA Toolkit和显卡驱动是不同信息，不能互相替代。选择与设备和驱动兼容的安装命令，以[PyTorch官方安装页面](https://pytorch.org/get-started/locally/)为准；不要在Mac上直接照搬Windows CUDA包。

复现实验还需保存：代码commit、实际模型与预处理配置、数据文件摘要和身份/日期清单、随机种子、训练历史与冻结权重。环境版本相同也不保证跨硬件逐位一致。当前已看过的S21–S30成绩不可用于继续调参后再宣称独立最终测试。
