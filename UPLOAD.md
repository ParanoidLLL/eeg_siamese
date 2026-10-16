# 上传 GitHub

本项目的上传版本已整理到本地 `release` 分支，项目根目录就是源码根目录。该分支使用独立历史，避免包含旧实验报告及本机路径。尚未配置远程仓库或推送 GitHub。

## 推送到 GitHub

在 GitHub 创建或选择自己的仓库，将 `YOUR_REPOSITORY_URL` 替换为仓库的 HTTPS 或 SSH 地址，然后在项目根目录执行：

```sh
git switch release
git remote add origin YOUR_REPOSITORY_URL
git push -u origin release
```

无需再次 `git init`。如果已经配置 origin，先用 `git remote -v` 核对地址。如果远端已有同名 release 分支，先核对其历史并正常整合，不要强制覆盖。推送后可以在 GitHub 选择 release 分支查看代码；这不会自动发布 GitHub Release 或修改远端默认分支。

本地原 main 分支保留，切换前未提交的修改保存在 Git stash，另有本地文件备份；这些备份不属于上传内容。

## 包含内容

轴向源码、默认模型/处理配置、相对路径索引模板、训练/评估/图表脚本、测试、CPU CI、模型/数据说明以及经过整理的历史对比图。

## 未包含内容

原始或处理后的EEG、检查点、注册库、训练日志、虚拟环境、完整本地Word及私有实验元数据、机器路径和SSH配置。

默认配置既在configs供用户编辑，也随Python包放入src/eeg_siamese/configs；若修改内置默认值，应同步两处并运行测试。安装后的数据转换从包资源读取模板，不依赖当前工作目录。

许可证尚未选定，本次未替仓库所有者指定。发布时由你决定仓库可见性和许可证，并自行核对所用数据的许可范围。
