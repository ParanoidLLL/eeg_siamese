# 本地提交与GitHub更新

GitHub发布线为release。1.2.0以release1.1为父版本加入W；本地main开发线有独立历史，不通过强制合并或强推覆盖发布历史。

从release创建自己的工作分支、检查变更并提交。数据、权重、注册库、运行日志和环境不应进入提交；真实EEG的处理与训练参数需要自行冻结。使用git diff --check和python -m pytest -q验证代码。再通过常规Git推送或PR更新release；不要强制推送。

现有本地W来源提交为7709dcd（W_local）；本次按文件提取通用代码和文档，未导入main的历史研究报告、机器路径CSV或运行结果目录。源码包与wheel提供W算法实现和配置，不提供新的全源20人最终权重。结果协议及局限见W_RESULTS.md。
