# 本地测试与 CI 验证记录

日期：2026-10-15。环境：Windows、Python 3.13.14。原始执行日志与 JUnit 留在本地，避免公开路径、运行记录与测试数据库；[verification-results.json](verification-results.json) 提供不含这些数据的汇总。

## 实际结果

| 检查 | 结果 |
| --- | --- |
| 公共 PyPI 新虚拟环境安装 | 使用独立 `.venv`，安装轻量 API + 前端，未复用项目原 `.venv`；成功。 |
| 仅运行依赖时启动 | 测试依赖安装前启动 Demo API 与 Streamlit；API 健康、UI HTTP 200，五页 AppTest 无 ImportError。 |
| 不分发词表缓存后的准备 | 在新的候选目录由 tiktoken 从官方源准备 `cl100k_base`，上游预期哈希校验成功；未调用模型 API。 |
| 最终候选启动复核 | 重新启动 Demo API/UI；`demo_mode=true`、`rag_enabled=false`、UI HTTP 200；五页组件无异常。 |
| 真实 HTTP 业务 | 递归讲解产生 8 题，客观题全对且主观题非空时 Demo 得分 97.5%；相同答案再提交返回 `replayed=true`，数据库总作答仍为 1。 |
| 全量测试 | **158 passed，1 warning，59.01 秒**。保留全部原有测试及前端测试，没有删除或跳过失败用例。 |
| 依赖一致性 | 清洁环境 `pip check` 返回 `No broken requirements found`。 |
| 生产实现保护 | 比较 106 项源码/Prompt/测试及依赖入口快照，105 项不变；唯一变化为 `tests/conftest.py` 的测试配置。生产代码、两张图、Harness、数据库实现和 Prompt 均不变。 |
| 测试夹具 lint | `ruff check tests/conftest.py` 通过；不将上游其他未改动文件的 lint 当作本轮 CI 门槛。 |
| 配置样例 | 覆盖全部 44 个类型化配置环境变量；合计 50 个示例变量，密钥为空，默认 Demo/关闭 RAG。 |
| 工作流静态校验 | YAML、三种触发器、只读权限、关闭凭据保存、完整 Action SHA、空模型密钥校验通过。 |
| 可选完整依赖 | 公共 PyPI dry-run 成功解析 152 个分发，用于许可清单；未安装/运行真实 Embedding 模型。 |

唯一自动化测试警告为继承的 `LangChainPendingDeprecationWarning`：将来检查点序列化器的 `allowed_objects` 默认值将变化。本轮未为消除警告修改检查点行为。

## 实际发现与最小修复

首次以 README 的 Demo 配置和 CI 的 `VECTOR_ENABLED=false` 执行全量测试时，**151 passed / 7 failed**。原因是测试夹具的 VectorStoreSettings 没有显式指定 enabled，会读取外部 Demo 开关，导致应使用 FakeEmbedder 的文档/RAG 用例被关闭。

修复仅为在 `tests/conftest.py` 的测试配置中添加 `enabled=True`。原 FakeEmbedder 注入、测试断言及被测生产实现不变。保持外部开关为 false 再跑全部测试，最终 158 项通过。没有通过省略文档测试、下载真实模型或修改业务逻辑获得通过。

## 本地复现

先完成主 README 的轻量安装和词表准备，再在根目录：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-test-local.txt -r requirements-frontend.txt -c requirements-verified.txt
$env:LLM_PROVIDER = 'demo'
$env:VECTOR_ENABLED = 'false'
$env:ANONYMIZED_TELEMETRY = 'False'
$env:HF_HUB_OFFLINE = '1'
$env:LANGSMITH_TRACING = 'false'
$env:LANGCHAIN_TRACING_V2 = 'false'
$env:TIKTOKEN_CACHE_DIR = Join-Path (Get-Location) 'demo/token_cache'
.\.venv\Scripts\python.exe -X utf8 -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
.\.venv\Scripts\python.exe -X utf8 -m pytest tests
.\.venv\Scripts\python.exe -X utf8 -m pip check
```

`requirements-verified.txt` 从本次清洁运行/测试环境生成，作为 `-c` 约束使用。不要直接安装整个文件来体验轻量 Demo，否则会安装只供测试的依赖。Windows 与其他平台条件依赖可能不同，CI 将负责验证配置的平台组合。

## GitHub Actions 配置与执行证据

[ci.yml](../.github/workflows/ci.yml) 配置 push、pull_request 和手动触发，矩阵为 Windows/Ubuntu + Python 3.13。安装、词表准备、pip check 和全部 tests 与本地步骤一致；不引用仓库模型 Secret，不启用真实模型。

Action 固定完整 SHA，权限为 `contents: read`，checkout 不持久化凭据。配置依据 [GitHub 官方安全参考](https://docs.github.com/en/actions/reference/security/secure-use)。

本页记录本地验证与工作流静态检查，不代表任何一次远程 CI 运行。Ubuntu 与 Windows Runner 的实际结果应查看对应的 [GitHub Actions 执行记录](https://github.com/uplaybiss/CodeMentor/actions/workflows/ci.yml)。
