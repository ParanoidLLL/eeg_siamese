# 重新连接 GitHub 与日常提交

旧云端仓库已删除，本地已移除失效的origin关联。当前项目完整保留，本地分支仍为 `release`；它只是Git分支名，不是另一套项目文件。无需重新初始化Git，也不要删除 `.git`。

## 重新上传（VS Code）

1. 在Mac本地VS Code打开本项目根目录；如果处于SSH远程窗口，先切回本地。
2. 在源代码管理检查全部新增、修改和删除项，暂存并提交。这次删除的是旧打包清单及根目录上传说明；上传说明现在位于本文件。
3. 选择“发布分支/Publish Branch”，如果提供“发布到GitHub”，按提示创建新仓库并选择私有可见性。
4. 如果该入口没有提供新建仓库，先在GitHub新建空仓库，不自动添加README、.gitignore或许可证。在VS Code终端执行以下命令，把地址替换成新仓库地址：

```sh
git remote add origin https://github.com/YOUR_USERNAME/YOUR_NEW_REPOSITORY.git
git push -u origin release
```

不要使用已删除仓库的地址，除非已重新创建同名仓库。不要强制推送。首次推送完成后，日常只需在VS Code暂存、提交、推送或同步更改。

## 哪些文件会上传

源码、配置、索引模板、脚本、测试、文档、CI配置会上传。`artifacts/`、`reports/`、`data/`、`.venv/`被Git忽略，保留在本地。目录外归档也不属于本仓库。请以VS Code实际暂存列表为准。

## 仓库简介

在新仓库About设置中手动填写：

- Description：`Cross-session EEG identification with a quality-aware axial Transformer, hard masks, soft quality weighting and corruption consistency.`
- Topics：`eeg`、`biometrics`、`pytorch`、`transformer`、`axial-attention`、`contrastive-learning`、`cross-session`。

按所有者选择，暂不添加开源许可证；本地和GitHub的可见性与授权是不同事项。本次未创建云端仓库、提交或推送。
