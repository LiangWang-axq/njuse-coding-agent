# 真实任务演示说明

演示目标：修复 TodoList.remove 把 Todo 编号误当作列表下标的问题，补充回归测试，并运行测试。

推荐视频流程（2 分钟内）：

1. 展示项目目录和 `.env` 未入库，说明 Agent 的工作区是当前目录。
2. 启动命令行 Agent，输入：“检查 demo Todo 项目，定位 remove 按编号删除失败的 bug，补充测试，运行测试并汇报结果。”
3. 让 Agent 展示 `list_files`、`read_file`、`search_code`、`replace_in_file`、`run_tests` 的实际调用。
4. 展示测试由失败变为通过，并让 Agent 输出 `final` 总结。

没有 API key 时可以运行 `python demo/run_offline_demo.py`。该脚本把一个带 bug 的临时 Todo 项目放入当前仓库的演示目录，用脚本化响应驱动同一个 Agent 循环，验证读写、修改、测试和终止；真实模型模式仍使用 `python -m coding_agent`。

关键设计说明：工具执行完全在本地 `WorkspaceTools` 中；模型看不到服务端托管的文件/代码工具；消息历史保存在 `CodingAgent.run` 的 `history` 列表；每次模型输出由 `parse_model_response` 解析；工具错误会作为结果回传，连续两次格式错误或达到最大步数则终止。
