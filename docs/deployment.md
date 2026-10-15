# 测试环境部署说明

本阶段只验证本地清洁环境，没有创建云资源、登录测试服务器或发布站点。以下步骤面向有访问控制的单实例测试环境。

## 1. 运行形态

- 两个 Python 进程：FastAPI API 与 Streamlit UI。学生访问 8501，API 为 8013。
- 当前先用 Demo Provider、关闭 RAG；不需要模型密钥或模型权重。
- 使用持久磁盘保存业务 SQLite、LangGraph 检查点、上传文件与可选向量库。不要将数据库提交到 Git。
- 单 API worker。运行锁在进程内；当前实现不提供跨实例协调。不能直接扩成多个进程同时处理同一学生的写入。

## 2. 在 Linux 测试主机准备

将经过确认的源码部署到专用项目目录，安装 Python 3.13。下面 `/opt/codementor` 与 `/var/lib/codementor` 是示例路径，应由部署用户拥有对应读写权限。

```bash
cd /opt/codementor
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-local.txt -r requirements-frontend.txt -c requirements-verified.txt
.venv/bin/python -m pip check
TIKTOKEN_CACHE_DIR="$PWD/demo/token_cache" .venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
```

Demo API：

```bash
.venv/bin/python -m demo.local --port 8013 --data-dir /var/lib/codementor/demo
```

另一个进程启动 UI：

```bash
FRONTEND_API_BASE_URL=http://127.0.0.1:8013 .venv/bin/python -m streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1 --server.fileWatcherType none --server.headless true
```

两进程应使用相同项目工作目录。默认只监听回环地址，可以使用已有 VPN/SSH 隧道访问测试机。需要由平台长期托管时，分别配置两个受管理进程、重启策略和工作目录；当前仓库没有安装系统服务的脚本。

## 3. 发布到可访问的测试入口之前

现有学习档案名和 student ID 不等于认证。文档语料和部分读取接口没有完整多租户授权，运行限额也不是用户配额系统。因此测试入口应由已有网关、VPN 或访问控制保护，使用虚构的测试记录，不能直接承载真实学生隐私与付费 API 服务。

如果使用现有反向代理，确保支持 Streamlit WebSocket，配置 HTTPS 和明确的访问来源；不要通过关闭 Streamlit XSRF/CORS 检查解决代理问题。API 不需要向普通浏览器单独公开。部署平台必须能同时托管 UI 和 API，而不是只启动一个进程。

## 4. 真实 Provider 与可选 RAG

真实模型使用普通 ASGI 入口，而不是强制 Demo 的启动器：

```bash
# Secret 由平台注入。此处不填写任何真实值。
# LLM_PROVIDER=openai 或 gemini
# 对应的 LLM_*_API_KEY 与 LLM_*_MODEL 必须由部署者设置。
# DATABASE_URL 与 HARNESS_CHECKPOINT_PATH 指向持久磁盘。
VECTOR_ENABLED=false .venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8013 --workers 1
```

`.env.example` 列出全部类型化配置字段。`demo.local` 会覆盖 Provider、向量开关和数据路径，所以在 Demo 命令中修改这些环境变量不会启用付费模型。Streamlit 客户端直接读取进程环境中的 `FRONTEND_API_BASE_URL`，应显式传入。

启用检索需安装原完整 `requirements.txt` 并配置 `VECTOR_ENABLED=true`、模型与持久路径，首次启用可能下载模型。PDF 解析的 PyMuPDF 采用 AGPL/商业双许可，先按 [第三方说明](../THIRD_PARTY_NOTICES.md) 确认适用许可及对应义务。不要在轻量 Demo 部署时自动安装整个模型栈。本轮没有执行真实 Provider 或真实 Embedding 的上线验证。

## 5. 验收、备份与回退

1. API `/health` 返回成功、`demo_mode=true`、`rag_enabled=false`；UI `/_stcore/health` 返回健康响应。
2. 浏览器完成一次“递归 → 练习 → 提交 → 学习画像”；确认只有一次作答记录，断线后有中文提示。
3. 确认数据库和检查点位于持久目录；停止两个进程后再一致备份业务 SQLite、检查点及可能存在的 WAL/SHM 文件。
4. 回退时停止服务，恢复上一个经过验证的源码版本与同一时点的数据备份，再启动单 API 进程及 UI。

当前没有数据库迁移体系、集群协调或自动回滚功能。本轮没有修改数据库结构，部署升级仍应先备份并在测试副本执行现有测试。
