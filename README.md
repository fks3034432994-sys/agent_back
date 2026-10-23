# Agent Backend

一个基于 **LangGraph + MCP** 的多模态 Agent 后端：流式多轮对话、知识库问答（向量召回 + 交叉编码器重排）、跨会话长期记忆，以及**视觉驱动的本机 GUI 自动化**。

整条链路是全异步的，工具以独立常驻的 MCP 服务形式运行，客户端断开时支持跨进程的协作式取消。

---

## 功能特性

| 功能 | 说明 |
|---|---|
| 流式对话 | SSE 输出 token / 工具调用 / 工具结果 / 人工确认中断 / 错误，带心跳保活 |
| 知识库问答 | 本地文档入库 → 向量召回 → 交叉编码器重排 → 阈值过滤，未命中时明确告知而非编造 |
| 长期记忆 | 跨会话记忆，按用途分「约束 / 资料」两类：约束类注入系统提示词，资料类按需语义检索 |
| 短期记忆 | 对话历史超长时自动摘要压缩，完整转录仍保留供前端展示 |
| 多模态 | 支持图片附件；图片以引用形式存检查点，发模型前才加载，避免数据库堆积 base64 |
| GUI 自动化 | YOLO 检测界面元素 + 编号标注 → 视觉模型指认 → 坐标换算点击 → 每步截图闭环校验 |
| 人工确认 | Agent 需要用户决策时通过 `interrupt()` 暂停，收到回答后从断点恢复 |
| 知识库管理 | 文档上传、按来源/片段删除、向量 PCA 可视化 |

---

## 架构

```
                        ┌─────────────────────────────┐
   客户端 ──SSE────────▶│  FastAPI 主服务  :8000       │
                        │  /api/v1/chat  /threads      │
                        │  /knowledge  /attachments    │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  LangGraph Agent            │
                        │  summarize_history          │
                        │      ↓                      │
                        │  main_model_call ⇄ ask_user │
                        └──────────────┬──────────────┘
                                       │ MCP over HTTP
        ┌────────────┬─────────────────┼─────────────────┬────────────┐
        ▼            ▼                 ▼                 ▼            ▼
   file:8761   python:8762        rag:8763         store:8764    gui:8765
   文件操作     shell/代码执行   知识库检索        长期记忆      GUI 自动化
                                                     │              │
                                            namespace 分层     YOLO + pyautogui
        └────────────┴─────────────────┴─────────────────┴────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  PostgreSQL + pgvector      │
                        │  检查点 / 长期记忆 / 向量库   │
                        └─────────────────────────────┘
```



---

## 核心设计

### 1. 跨进程协作式取消

常驻 MCP 服务是多会话共享的，不能像 stdio 时代那样「按 PID 杀进程」——那会连带杀死其它会话正在执行的任务。而 FastMCP 把同步工具派发到工作线程执行，**asyncio 的取消无法中断线程里阻塞的同步代码**，所以只能让工具自己观察信号：

1. 主进程把 `tool_call_id` 作为 `task_id` 注入长耗时工具的参数；
2. 工具开始执行时在服务端注册一个 `threading.Event`；
3. 客户端断开 / 点击停止时，主进程调用服务内的 `cancel_task`（内部工具，不暴露给模型）置位该 Event；
4. 工具在轮询或步骤边界检查 Event，主动终止——shell/python 终止整个子进程树（Windows 用 `taskkill /F /T`），GUI 在步骤边界跳出循环。

取消延迟：shell/python 实测中位数约 0.3 s（`cancel_task` 自身往返仅 22 ms，其余是等下一次轮询检查 Event，见「实测数据」），GUI 秒级（单步内的点击、截图、推理、模型调用不可中断）。

两个容易踩的时序问题，代码里都做了处理：

- 图被取消时，节点 `finally` 会立刻注销工具登记，所以调用方必须**先快照、后取消**；
- 客户端断开时 Starlette 已取消当前协程，而 anyio 的取消是 level-triggered，作用域内直接 `await` 的清理等于没做——必须丢进独立的 `asyncio.Task`。

### 2. RAG 检索链路

```
查询 → (指代消解改写) → 向量召回 top-20 → 交叉编码器重排 → 双阈值过滤 → 结果标注来源
```

- **查询改写**：`search_knowledge` 跑在独立 MCP 进程里，拿不到对话历史，所以「那这个要提前几天」这类带指代的追问由主进程先改写成可独立检索的问题。用启发式粗筛（含指代标记或提问过短）控制额外模型调用。
- **超量召回 + 重排**：向量检索是双塔模型，只能判断「语义是否接近」；交叉编码器把 query 与文档拼在一起过模型，判断的是「这段文字是否真的回答了这个问题」。实测同一批语料下，重排把正确答案与噪声拉开 4～5 倍（0.579 / 0.0001），而纯余弦相似度只差 1.2 倍（0.64 / 0.53）——所以「低于阈值即丢弃」在重排之后才真正可用。
- **未命中要说出来**：全部低于阈值时明确返回「知识库未命中」，让模型知道该声明答案不来自知识库，而不是拿噪声硬答。

### 3. 长期记忆分层

记忆按「是否要求 Agent 改变行为」分成两类，用 namespace 第二级区分：

| | namespace | 内容 | 使用方式 |
|---|---|---|---|
| 约束类 | `("user", "constraint")` | 回答语言、称呼方式、回复风格、禁忌 | 会话首轮整体注入系统提示词 |
| 资料类 | `("user", "material")` | 姓名、职位、项目背景等事实 | 不注入，按需语义检索 |

写入侧带冲突检测：同一 key 内容不一致时不覆盖，返回冲突提示让 Agent 向用户确认，用户同意后才强制覆盖。

### 4. 短期记忆（对话历史摘要）

- 只挂在**用户轮次入口**——工具循环与人工确认续答不重复摘要，避免同一轮任务中途压缩上下文；
- **双通道**：`messages` 保留完整转录（前端历史接口与图片引用依赖它），`summarized_messages` 存放「摘要 + 近期若干条」，才是实际发给模型的历史；其 reducer 是整体替换而非累加，否则摘要会被一轮轮叠加。
- 防护：未触发摘要时不写通道（避免每轮把完整历史拷一份进检查点）；摘要为空时放弃本轮更新（否则会把好摘要覆盖成空壳）。

---

## 快速开始

### 环境要求

- Python ≥ 3.14
- PostgreSQL，并安装 **pgvector** 扩展
- 一个兼容 OpenAI 协议的对话模型服务、一个嵌入模型服务（本项目默认使用火山方舟与硅基流动，可替换）

### 安装

```bash
uv sync
```

### 配置

在项目根目录创建 `.env`：

```ini
# 数据库（检查点 / 长期记忆 / 向量库共用同一个库）
DB_URI=postgresql://postgres:password@localhost:5432/agent

# 主对话模型
MAIN_MODEL_ID=your-model-id
MAIN_MODEL_URL=https://your-endpoint/v1
MAIN_MODEL_API_KEY=your-api-key

# 嵌入模型（用于知识库与长期记忆的语义检索）
EMBEDDING_MODEL_ID=your-embedding-model
EMBEDDING_MODEL_URL=https://your-embedding-endpoint/v1
EMBEDDING_MODEL_API_KEY=your-api-key

# GUI 自动化（不使用该功能可不填）
TARS_MODEL_ID=your-vision-model
TARS_MODEL_API_KEY=your-api-key
TASK_PLANNING_MODEL_ID=your-planning-model
SILI_CONFLOW_URL=https://api.siliconflow.cn/v1
SILI_CONFLOW_API_KEY=your-api-key

# HuggingFace 缓存目录（GUI 检测模型，建议绝对路径）
HF_HOME=./hf_cache
```

首次启动会自动建表，无需手工执行 DDL。

### 启动

MCP 服务是独立常驻进程，需要先启动（另开一个终端保持运行）：

```bash
python scripts/start_mcp_servers.py
# 可选：只启动部分服务、或调整就绪等待上限
# python scripts/start_mcp_servers.py --only file_server,gui_server --timeout 180
```

再启动主服务：

```bash
python src/main.py
```

健康检查：`GET http://localhost:8000/api/v1/health`

### 构建知识库（可选）

把纯文本资料放进 `src/rag/knowledge_base/`，然后：

```bash
python -m src.rag.indexer
```

也可以运行期间通过 `POST /api/v1/knowledge/documents` 上传。

---

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/chat` | 流式对话（SSE）。`resume=true` 时表示这是对中断提问的回答 |
| GET | `/api/v1/threads` | 会话列表（按最近活跃倒序，附首条用户消息预览） |
| GET | `/api/v1/threads/{thread_id}/messages` | 会话历史消息 |
| DELETE | `/api/v1/threads/{thread_id}` | 删除会话及其检查点 |
| POST | `/api/v1/attachments` | 上传附件（文本 / 图片），返回 MCP resource URI |
| GET | `/api/v1/attachments/{name}` | 按文件名读取附件原始内容 |
| GET | `/api/v1/knowledge/embeddings` | 知识库向量列表（用于可视化，含 PCA 投影） |
| GET | `/api/v1/knowledge/embeddings/{point_id}` | 单个向量片段详情 |
| DELETE | `/api/v1/knowledge/embeddings/{point_id}` | 删除单个片段 |
| DELETE | `/api/v1/knowledge/sources?source=...` | 按来源删除全部片段 |
| POST | `/api/v1/knowledge/documents` | 上传文档并向量化入库 |

### SSE 事件

`token` / `tool_call` / `tool_result` / `ask_user` / `done` / `error`，以及作为注释行的心跳。

---

## 测试

```bash
uv run pytest
```

测试套件**不依赖数据库、不启动 MCP 服务、不联网、不调用大模型**，克隆即可运行。覆盖范围包括：协作式取消的状态机、SSE 协议格式与事件流、摘要双通道契约、RAG 阈值与 fail-open 行为、重排客户端、嵌入客户端分批与重试、知识库入库路径规整、附件文件名安全、MCP 地址表。

`evals/` 下另有一份基于 LLM-as-judge 的答案质量评估脚本。

---

## 实测数据

复现：`uv run python scripts/bench_metrics.py`（可加 `roundtrip` / `cancel` / `image` / `compress` / `e2e` 只跑单项，逗号分隔可跑多项）。以下为本机实测（Windows / CPU、PostgreSQL 本地部署），样本量与测量条件一并列出，避免脱离条件引用数字：

| 指标 | 实测 | 测量条件 |
|---|---|---|
| MCP 工具调用往返 | 中位数 **71 ms** | 20 次 shell 工具调用，已预热 |
| 协作式取消延迟 | 端到端中位数 **0.27～0.33 s** | 8 次采样，两次运行的观测范围 |
| 上下文压缩 | 9,765 → 约 1,900 token（**-80%**） | 70 条消息触发摘要，真实模型生成；两次运行 1,811 / 1,959 |
| 图片引用存储 | 上下文体积 **-79%** / token **-35%** | 7 轮，每轮 1 张图 |
| 端到端延迟 | 中位数 **4.5 s** | 3 次知识库问答，含工具调用 |

几个数据背后的口径，比数字本身更值得注意：

- **取消延迟几乎全是「等下一次轮询」**：`cancel_task` 自身的 HTTP 往返只占 22 ms，其余是服务端等到下一轮检查 Event 的时间。实测服务端检测等待在 0～500 ms 上近似均匀（两次运行均值 257 / 286 ms），与 `POLL_INTERVAL=0.5` 的理论期望（250 ms）吻合——**这个理论值才是稳定的**，单次中位数会随采样在 0.27～0.33 s 之间浮动。该数字还对**测量方式很敏感**：若发出取消的时刻固定成轮询周期的整数倍，信号每次都会落在「刚进入新一轮等待」之后，只等一个零头，能把延迟系统性低估成几十毫秒；必须给等待时长加随机抖动才测得准。
- **图片引用的 token 既不能按 base64 长度折算，也不能用 `count_tokens_approximately` 估**：视觉模型按图像本身计费，图像块固定按 85 token 计（对齐 OpenAI 低分辨率开销），与 base64 字符数无关；文本侧则必须用真实分词器——`count_tokens_approximately` 按 ～4 字符/token 估，而中文在 BPE 里常占 1.5～2 token，普遍低估数倍。两者叠加会把收益抬高：同一组数据用库默认口径报 59.7%，文本换成 `tiktoken`（`cl100k_base`）真实计数后是 34.6%。所以「体积 -79%」与「token -35%」是两个口径——前者是真实字节数、与分词器无关（反映检查点与传输负担），后者反映上下文占用。文本偏低的影响在本项里格外大，因为被低估的分母是**两边共有**的中文，压低分母就等于抬高图像那部分的占比。
- **端到端延迟的主导项不是 MCP**：4.5 s 里 MCP 往返只占 71 ms，其余是模型往返 × 工具轮次——同一批问题中模型自主多调 3 个工具的那一次耗时 11.3 s。该指标应表述为「知识库问答中位数 4.5 s」，单次值没有代表性。

---

## 可调项

大部分行为可通过环境变量调整，常用的几个：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DB_POOL_MAX_SIZE` | 10 | 数据库连接池大小 |
| `MCP_TOOL_TIMEOUT` | 3600 | MCP 工具超时（GUI 任务可能长达数分钟） |
| `HISTORY_SUMMARY` | 1 | 是否启用对话历史摘要 |
| `HISTORY_SUMMARY_MAX_TOKENS` | 8000 | 摘要后给模型的历史预算 |
| `RAG_REWRITE` | 1 | 是否启用检索查询改写 |
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 重排模型，置空则关闭重排 |
| `RAG_MIN_SCORE` | 0.4 | 余弦相似度阈值 |
| `RAG_MIN_RERANK_SCORE` | 0.1 | 重排分数阈值 |

其余可调项见各模块的注释说明。

---

## 项目结构

```
src/
├── main.py                     # FastAPI 入口（lifespan 初始化持久化与 Agent）
├── asyncio_runner.py           # 统一的异步入口（Windows 上强制 SelectorEventLoop）
├── agents/
│   ├── graph/
│   │   ├── main_graph.py       # 主对话图：摘要 → 模型 ⇄ 人工确认
│   │   └── touch_graph.py      # GUI 自动化子图
│   ├── mcps/                   # 5 个 MCP 服务 + 取消/在飞工具注册表
│   ├── checkpoint/             # 检查点与长期记忆的 Postgres 持久化
│   ├── models/                 # 模型与嵌入客户端
│   └── utils/                  # 截图、YOLO 检测、坐标换算与点击
├── api/                        # 对话（SSE）、知识库、附件接口
└── rag/                        # 索引、查询改写、重排
tests/                          # 全部离线可跑的单元测试
evals/                          # LLM-as-judge 评估
scripts/                        # MCP 服务启动器、自检与指标基准脚本
```

---

## 已知限制

- **GUI 取消延迟为秒级**：单步内部的点击、截图、YOLO 推理与模型调用无法中途打断，只能在步骤之间停止。
- **GUI 元素定位不使用 OCR**：文字理解完全交给多模态视觉模型，元素定位依赖检测框编号，检测质量直接决定成功率。
- **GUI 任务串行**：鼠标键盘与截图路径是独占资源，同一时刻只允许一个任务，第二个请求直接返回「繁忙」而不排队。
- **无鉴权**：当前为开发用途，未接入认证与多租户隔离。
- **知识库仅支持纯文本**：PDF / Word 等格式需要额外的解析器。

---

