# 演示说明

## 推荐演示任务：demo/tasks/fix_me

`demo/tasks/fix_me` 是一个故意留下两个 bug 的小项目：`format_total` 缺少人民币符号、`apply_discount` 用减法代替百分比折扣；对应测试先失败。

在 `demo/tasks/fix_me` 目录下启动 Agent：

```powershell
cd demo/tasks/fix_me
python -m coding_agent "检查订单计算模块，修复金额格式化与折扣计算的 bug，运行测试直到全部通过"
```

Agent 会展示 `list_files`、`read_file`、`search_code`、`replace_in_file`、`run_tests` 的实际调用，测试由失败变为通过，最后输出 `final` 总结。

## 离线演示（无 API key）

```powershell
python demo/run_offline_demo.py
```

脚本把带 bug 的临时 Todo 项目放入 `demo/.runtime`，用脚本化模型响应驱动同一个 Agent 循环，验证读写、修改、测试和终止。该目录已被 .gitignore 忽略。

## 关键设计说明

工具执行完全在本地 `WorkspaceTools` 中，模型不使用任何服务端托管的文件/代码工具；消息历史由 `ContextManager` 维护并在超限时压缩；模型输出由 `parse_model_response` 解析（兼容 JSON 动作与原生 tool calls，支持一次多个工具调用）；工具错误作为结果回传，连续两次格式错误或达到最大步骤数则终止；API 429/5xx/超时自动退避重试。
