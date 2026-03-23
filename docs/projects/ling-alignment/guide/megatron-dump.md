---
title: Megatron-LM Dump 数据指南
---

# Megatron-LM Dump 数据指南

本文介绍如何在 Megatron-LM 项目中触发 Argus dump 以及如何找到 dump 输出。

## 通过 GitHub 页面触发 Dump

1. 打开 [Megatron-LM Actions 页面](https://github.com/primatrix/Megatron-LM/actions/workflows/train-ci.yml)
2. 点击右侧 **Run workflow**
3. 选择要运行的分支（可以选择开发分支）
4. 勾选 **Enable Argus dump and verification**
5. 点击 **Run workflow**

![GitHub Actions 手动触发 dump](/images/megatron-dump-workflow-dispatch.jpg)

> `main` 分支的 CI 会自动启用 dump，无需手动勾选。

## 通过 gh 命令触发 Dump

```bash
# 在 main 分支上触发（自动启用 dump）
gh workflow run train-ci.yml -R primatrix/Megatron-LM

# 在指定分支上触发，并启用 dump
gh workflow run train-ci.yml -R primatrix/Megatron-LM \
  --ref feat/my-dump-feature \
  -f enable_dump=true

# 查看运行状态
gh run list -R primatrix/Megatron-LM -w train-ci.yml -L 5
```

如果需要在新的开发分支上触发，先创建并推送分支：

```bash
cd Megatron-LM
git checkout main && git pull
git checkout -b feat/my-new-feature
git push -u origin feat/my-new-feature
```

然后用上面的 `gh workflow run --ref feat/my-new-feature -f enable_dump=true` 命令触发即可。

## 找到 Dump 输出

Dump 数据自动上传到 GCS，路径格式为 `megatron_ci_{TIMESTAMP}_{COMMIT_HASH_前6位}`，例如：

```text
/models/argus_dump/megatron_ci_20260322_162702_6a9d70/
  step_{N}/
    rank_0/
      metadata.yaml
      forward/   backward/   params/   grads/   optim_states/   loss/
    rank_1/ ... rank_7/
```

对应的 GCS 路径为 `gs://ant-pretrain/argus_dump/megatron_ci_20260322_162702_6a9d70/`。

验证 dump 完整性：

```bash
python ci/verify_argus_dump.py --dump-dir <path> --num-ranks <N>
```
