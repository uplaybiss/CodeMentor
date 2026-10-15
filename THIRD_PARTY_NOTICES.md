# 第三方代码、依赖与资产说明

审查日期：2026-10-15。项目自身的 MIT 许可不替代任何第三方许可。

## EduAgent 上游代码

- 来源：[Varsha-art126/EduAgent](https://github.com/Varsha-art126/EduAgent)。
- 开发基线：`99ae1a3ea390acd202f8873a3608fab79ca0f6e3`，完整继承历史含 3 个上游提交。
- 上游许可：MIT，原 `Copyright (c) 2026 EduAgent contributors` 保留在根目录 [LICENSE](LICENSE)。CodeMentor 修改另行标注，没有删除或替换上游版权。
- 原评估/实验实现保留；上游生成物与旧总结从当前候选树排除，不作为 CodeMentor 的测量结论。继承 Git 历史中的原文件仍属于其原始版本。

## Python 依赖

本地与 CI 清单为 `requirements-local.txt`、`requirements-frontend.txt`、`requirements-test-local.txt`，具体传递版本在 `requirements-verified.txt` 约束。完整可选模型依赖继承自 `requirements.txt`。

安装分发的逐包许可与版本，以及完整可选清单 dry-run 的逐包元数据，见 [docs/dependency-licenses.md](docs/dependency-licenses.md)。该清单区分实际安装与仅解析的版本，不声称真实 Embedding 或 Provider 已验证。

重要许可事项：

- **PyMuPDF**：PDF 解析和相关测试使用；上游采用 GNU AGPL v3 或 Artifex 商业许可。轻量 Demo 安装不包含它，测试与完整 RAG 依赖会包含它。使用、组合分发或面向用户部署 PDF 功能前，需确认适用许可并履行对应义务，或取得商业许可。依据：[PyMuPDF 官方许可说明](https://pymupdf.io/licensing)。没有把该依赖重新标记为 MIT，也没有为规避许可删除 PDF 功能或测试。
- **certifi、orjson、tqdm 等**包含 MPL 或多许可条款；NumPy/SciPy 等 wheel 还可能附带其他组件许可。逐包字段不能替代 wheel 内的 LICENSE/NOTICE/COPYING。
- 当前候选只含项目源码与通知，不打包 `.venv`、wheel、模型权重或安装依赖。若后续制作完整二进制/容器分发，应保留各包实际分发中的许可、通知与适用的源码提供义务。

## tiktoken 与分词词表

Context Builder 使用 tiktoken 的 `cl100k_base` 计数。tiktoken 源码的 MIT 许可副本保存在 [licenses/tiktoken-LICENSE](licenses/tiktoken-LICENSE)，原版权归 OpenAI、Shantanu Jain。依据：[tiktoken 官方 LICENSE](https://github.com/openai/tiktoken/blob/main/LICENSE)。

词表缓存作为本地生成物排除，不随候选源码再分发。README 与 CI 的准备命令由已安装 tiktoken 从其官方地址读取，并使用上游预期哈希校验；不调用模型 API。来源和校验实现在 [tiktoken openai_public.py](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)。本地已有缓存保留，未删除。

## GitHub Actions

CI 使用 `actions/checkout` v6.0.3 与 `actions/setup-python` v6.3.0，均固定到从官方仓库验证的完整 commit SHA。Action 代码由 GitHub Runner 获取，不复制进本项目；各自许可继续适用。配置为只读仓库权限，不保存 checkout 凭据，也不接入付费模型 Secret。
