# Codex multi-agent 调用失败诊断 · 2026-09-13

## 结论

`unsupported call: spawn_agent` 不是项目代码、`AGENTS.md`、任务目标或模型推理造成的。当前 Codex Desktop 会话没有注册 multi-agent 工具 namespace；路由器收到未注册的工具名后直接返回 unsupported call。

二进制本身包含 multi-agent handler，但这只证明客户端代码支持该能力，不证明当前 app-server / thread 的工具面已经把它暴露给模型。

## 证据

1. 当前会话实际工具清单没有 `spawn_agent`、`followup_task`、`list_agents` 或任何 multi-agent namespace。应以实际工具清单为准，不能只看 developer 文本或旧 session 记录。
2. 当前环境是 `CODEX_HOME=/Users/guoshaoyang/.codex-phybench`，CLI 版本为 `codex-cli 0.153.4`。
3. 安装包字符串包含 `multi_agents_v2/spawn.rs`、`followup_task`、`list_agents` 等实现，说明 handler 存在。
4. 四个本地 Codex home 已检查并启用 `multi_agent = true` 和 `multi_agent_v2 = true`，并保留了 `config.toml.bak-multi-agent-20260913` 备份。
5. 在同一 CODEX_HOME 启动全新 ephemeral CLI 进程，并在 prompt 中要求调用 `spawn_agent`，仍得到 `ERROR codex_core::tools::router: error=unsupported call: spawn_agent`，排除了仅是旧会话缓存工具面的解释。
6. 配置预检读取到 `config_v2_enabled=true`、`mode=v2`、`owner=native`，但它只验证静态配置，不代表响应 API 已注册 delegation tool。

## 为什么改配置没有立即生效

工具 namespace 在 thread/app-server 启动时由运行时编排器决定。它不是普通 prompt、skill 或仓库配置，不能由模型在当前对话中动态注入。当前 Desktop 的 app-server 进程早于本轮配置修改启动，且现有工具 surface 已固定；修改 TOML 不会热加载到这条线程。

## 修复步骤

1. 完全退出并重新打开 Codex Phybench，使 app-server 重新读取 CODEX_HOME 配置；只新建 thread 不一定会重启共享 app-server。
2. 新 thread 的第一步检查实际 tool list，必须出现 multi_agent_v2（或直接出现 `spawn_agent` / `followup_task`）。
3. 用一个最小 probe 验证：只 spawn 一个 `fork_turns=none` child，让它返回固定字符串；确认 child 的 final message 和 list_agents 状态后再投入项目任务。
4. 如果重启后仍没有 namespace，问题在 Desktop build / app-server feature gate / relay tool allowlist，而不在本地配置。应升级或切换到提供 native multi-agent tool surface 的 Codex build，并把本文件证据交给维护者。

## 当前不能做的事情

本线程不能通过编辑文件、安装 skill 或调用 MCP 自己接上缺失的原生工具。MCP skill 可以增加外部命令能力，但不能伪造 Codex 内部的 child-thread 生命周期、mailbox 和 spawn 路由。可以使用独立 `codex exec` 进程作为降级并行方案，但那不是原生 subagent，也不能冒充已建立的 agent tree。

## 诊断时应避免的误判

- `multi_agent_mode=proactive` 出现在旧 session/world-state 元数据中，只说明当时的协作策略提示，不等于工具已经注册。
- 二进制里存在 spawn handler 不等于当前线程能调用它。
- `AGENTS.md` 写了“尽可能使用 subagents”只能表达偏好和流程要求，不能开启运行时能力。
- 配置预检显示 `mode=v2` 只能说明配置解析成功，不能替代实际 tool-list probe。
