# CodeMentor 中文界面：Windows + VS Code

在 VS Code 打开项目根目录（含 `README.md` 和 `frontend/` 的目录），使用 PowerShell。建议 Python 3.13。

## 安装

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-local.txt -r requirements-frontend.txt -c requirements-verified.txt
.\.venv\Scripts\python.exe -X utf8 -m pip check
$env:TIKTOKEN_CACHE_DIR = Join-Path (Get-Location) 'demo/token_cache'
.\.venv\Scripts\python.exe -X utf8 -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
```

已有 `.venv` 时直接安装依赖。无需激活环境或修改执行策略。分词词表只在首次准备时从官方源校验下载，之后可离线使用；它不是模型权重或模型 API 调用。Demo 不要求密钥或 `.env`；需要配置时参考 `.env.example`，不要覆盖已有 `.env`。

## 启动两个终端

终端一：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m demo.local --port 8013
```

终端二：

```powershell
$env:FRONTEND_API_BASE_URL = "http://127.0.0.1:8013"
.\.venv\Scripts\python.exe -X utf8 -m streamlit run frontend/app.py --server.fileWatcherType none --server.port 8501
```

学生打开 **http://localhost:8501**；**http://localhost:8013/docs** 是接口文档。两个终端持续运行，停止时按 `Ctrl+C`。

## 使用路径

1. 学习首页选择知识点，阅读真实学习概览。
2. AI 辅导发送问题或阅读讲解，勾选“讲解后生成练习”。
3. 专项练习直接选择/填写答案，提交后查看评分及逐题解析。
4. 学习画像查看实际成绩曲线、当前弱项和下一步建议。
5. 学习资料在后端启用检索时开放管理；Demo 默认显示关闭说明。

默认数据位于 `data/local_demo/`。切换档案保留其他档案历史；清空界面会话不删除数据库。浏览器整页刷新或重启前端会重置当前聊天与活动测验，历史成绩保留。同一名称对应本地同一档案，这不是登录。

Demo 使用固定中文内容；客观题确定性评分，非空主观题固定得分 80%，只用于演示流程。

## 常见问题

| 现象 | 处理方法 |
| --- | --- |
| 连接不上学习服务 | 检查终端一和 8013 端口，再点击“重新连接”。更换 API 地址后需重启终端二的进程。 |
| ImportError / 缺少 Streamlit | 用命令中同一 `.venv` 解释器安装后端和前端依赖。 |
| 修改代码没有更新 | 禁用了文件监听，停止并重启 Streamlit。 |
| 端口已占用 | 停止自己之前启动的同一服务，或同时更换 API 端口与前端连接地址。 |
| 学习资料没有上传 | 本地 Demo 关闭 RAG，这是预期行为。 |
| 有错题但画像暂无弱项 | 错题描述单题表现，画像弱项依据当前整体掌握度 `<60%`。 |
| 较窄窗口导航折叠 | 使用页面上方原生展开按钮恢复侧栏。 |

实际测试记录见 [docs/testing.md](docs/testing.md)，测试部署见 [docs/deployment.md](docs/deployment.md)。
