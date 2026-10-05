# 安全培训智能 Agent（Safety Training Agent）

面向建筑施工 **三级安全教育 / 岗前安全培训** 的智能体：按「工种 + 三级教育层级（公司级 / 项目级 / 班组级）」自动生成标准化**培训资料与考核试卷**（支持下载为 Word）、OCR 识别与自动判卷、输出合规培训台账，并基于内部规程 / 交底 / 事故案例 / 应急预案做检索增强（RAG）。

> 技术栈对齐项目简历：**LangChain + LangGraph** Agent / 本地 **Ollama deepseek-r1:7b（GPU 推理）** / 本地 **BGE 中文嵌入** / **Milvus**（默认，IVF / 量化 / 图索引可选）+ **FAISS** 备选向量库 / **BM25 + 向量混合检索 + BGE-reranker 重排** / FastAPI + 豆包风格前端 / Docker 部署。

---

## 一、核心特性

- **智能对话 Agent（工具调用模式）**：基于 LangGraph `create_agent` + Function-Calling 编排——`deepseek-r1:7b` 自主决定调用 5 个 LangChain `StructuredTool`（`search_safety_knowledge` / `generate_training_exam` / `generate_training_material` / `grade_paper` / `create_training_ledger`），内置多轮会话记忆（InMemorySaver）；Ollama 不可用时自动降级到离线确定性引擎（`RuleBasedChatLLM` 实现同接口），回答有据可依、稳定可靠。
- **向量库（Milvus 默认 + FAISS 备选）**：删除原 Chroma，替换为 Milvus 向量库，索引类型可配置：
  - `IVF_FLAT`：IVF 倒排索引，检索快、召回准；
  - `IVF_SQ8` / `IVF_PQ`：**量化索引**，压缩向量存储、显著降低内存；
  - `HNSW`：**图索引**，召回率最高，适合高精度场景；
  - `FLAT`：暴力检索，数据量小时保证精度。
  - Milvus 不可用时自动降级到本地 **FAISS**（同样支持 IVF / IVF_SQ8 量化索引），保证开发环境可运行。
- **本地模型**：LLM 使用 **Ollama deepseek-r1:7b**（完全本地推理，NVIDIA GPU 加速，无 GPU 自动回退 CPU）；嵌入使用 **BGE 中文嵌入模型**（`BAAI/bge-base-zh-v1.5`）；重排使用 `BAAI/bge-reranker-base`。
- **混合检索**：BGE 向量 + BM25（jieba 中文分词）Ensemble 混合 → BGE-reranker 重排 → 工种文件保底 + 来源多样性限额，回答附原文溯源。
- **Word 导出**：生成的**试卷、培训资料、台账**均可一键下载为 `.docx`（python-docx，中文字体排版）。
- **豆包风格前端**：仿豆包简洁对话界面，支持智能对话、生成试卷、生成培训资料、OCR 判卷、培训台账五个面板。
- **评估**：检索 Hit@k / MRR、出题结构合规率、判卷判分准确率、综合准确率。

---

## 二、目录结构

```
safety_training_Agent/
├── main.py                  # FastAPI 入口（路由 + 前端托管 + Word 下载）
├── app/
│   ├── config.py            # pydantic-settings 配置中心（Ollama/Milvus/FAISS/索引参数）
│   ├── schemas.py           # Pydantic 数据模型（试卷/培训资料/判卷/台账/HTTP）
│   ├── runtime_env.py       # import 重型库前的离线引导 setup_offline
│   ├── llm_factory.py       # LLM 工厂：Ollama deepseek-r1:7b + 离线确定性引擎
│   ├── document_loaders.py  # 多格式加载 + .doc/.ppt 转换（COM/LibreOffice）
│   ├── rag_retriever.py     # RAG 底座：分块/BGE/Milvus·FAISS/BM25/重排/工种保底
│   ├── exporters.py         # Word 导出：试卷 / 培训资料 / 台账 → .docx
│   ├── ocr_engine.py        # RapidOCR 引擎（CPU/ONNX）
│   ├── tools.py             # 出题/资料/判卷/台账业务服务（Agent 节点复用）
│   ├── agent.py             # SafetyTrainingAgent 核心类
│   └── evaluate.py          # AgentEvaluator 评估器
├── scripts/
│   ├── build_kb.py          # 构建知识库
│   ├── run_eval.py          # 运行评估
│   └── chat_cli.py          # 命令行对话
├── static/index.html        # 豆包风格单页前端
├── eval/                    # 评估用例 eval_questions.json / 报告 eval_report.json
├── faiss_db/                # FAISS 备选向量库持久化（运行产物）
├── data_cache/              # .doc 转换缓存 + BM25 块（运行产物）
├── logs/                    # 日志
├── requirements.txt
├── Dockerfile / docker-compose.yml / .dockerignore
└── .env / .env.example
```

## 三、环境与配置

- Python 3.12（本项目在 conda 环境 `cpu_default` 下开发；本机 NVIDIA RTX 2060 6GB + CUDA 12.6，torch 2.14.0+cu126，**GPU 推理**：Ollama 7b 与 BGE 嵌入/重排共用显存约 5.7GB）
- 安装依赖：
  ```bash
  pip install -r requirements.txt
  ```
- 配置：复制 `.env.example` 为 `.env`，关键项：
  ```ini
  LLM_BACKEND=ollama
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_MODEL=deepseek-r1:7b

  EMBEDDING_MODEL=BAAI/bge-base-zh-v1.5
  RERANKER_MODEL=BAAI/bge-reranker-base
  DEVICE=cuda        # NVIDIA GPU 推理；无 GPU 或无 CUDA 版 torch 时改为 cpu
  USE_RERANKER=true
  USE_BM25=true

  DATA_DIR=D:/2-Agent/datas
  VECTOR_DB=milvus
  MILVUS_URI=http://localhost:19530
  MILVUS_INDEX_TYPE=IVF_SQ8   # FLAT | IVF_FLAT | IVF_SQ8 | IVF_PQ | HNSW
  FAISS_DIR=./faiss_db
  ```

### 本地 Ollama 模型

```bash
# 确保已安装并拉取模型（deepseek-r1:7b；7b 约 4.4GB）
ollama pull deepseek-r1:7b
ollama serve   # 默认 11434 端口；有 NVIDIA GPU 时 Ollama 自动启用 GPU 推理
```

### Milvus 向量库（Docker）

```bash
# 一键启动 Milvus 单机版（etcd + standalone，common.storageType=local 本地存储，无需 MinIO）
docker compose up -d standalone
# 验证：http://localhost:19091/healthz 返回 {"status":"OK"}
```

> Milvus 未启动时，系统会自动降级到本地 **FAISS** 向量库（`VECTOR_DB_FALLBACK_FAISS=true`），开发调试不受影响。

## 四、快速开始（本地）

```bash
# 1) 构建知识库（首次；约 3900+ 文本块，CPU 编码约十几分钟）
python scripts/build_kb.py --backend ollama

# 2) 运行评估
python scripts/run_eval.py --backend offline      # 离线确定性
python scripts/run_eval.py --backend ollama      # 本地 Ollama

# 3) 启动服务
python main.py
# 或：uvicorn main:app --host 0.0.0.0 --port 8000
```

打开浏览器：
- 前端页面：http://localhost:8000/
- 接口文档（Swagger）：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## 五、HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/health` | 健康检查（后端 / 向量库 / 索引类型 / 模型） |
| GET  | `/` | 豆包风格前端页面 |
| POST | `/api/chat` | 自然语言对话（Agent 自动选择工具，支持多轮记忆） |
| POST | `/api/exam/generate` | 按工种 + 层级结构化生成试卷 |
| POST | `/api/exam/download` | **生成试卷并下载为 Word（.docx）** |
| POST | `/api/material/generate` | 按工种 + 层级生成标准化培训资料 |
| POST | `/api/material/download` | **生成培训资料并下载为 Word（.docx）** |
| POST | `/api/grade` | 判卷（客观题确定性比对 + 主观题辅助评分） |
| POST | `/api/ledger` | 生成培训台账 |
| POST | `/api/ledger/download` | **生成培训台账并下载为 Word（.docx）** |
| POST | `/api/ocr` | 上传试卷图片，返回 OCR 文本 |
| POST | `/api/kb/build` | （重新）构建知识库 |
| GET  | `/api/kb/stats` | 知识库统计 |

示例：生成电工班组级试卷并下载为 Word
```bash
curl -X POST http://localhost:8000/api/exam/download \
  -H "Content-Type: application/json" \
  -d '{"craft_type":"电工","edu_level":"班组级"}' \
  -o 试卷_电工_班组级.docx
```

## 六、Docker 部署（含 Milvus）

已配置国内镜像加速与阿里云 apt/pip 源。在项目根目录：

```bash
# 构建并后台启动（etcd + Milvus 本地存储模式 + 应用）
docker compose up -d --build

# 查看日志 / 停止
docker compose logs -f
docker compose down
```

`docker-compose.yml` 会挂载：
- 原始数据 `D:/2-Agent/datas` → `/data`（只读）
- 宿主机 BGE 模型缓存 → `/root/.cache/huggingface`（只读，避免重复下载）
- `faiss_db / data_cache / logs` 持久化卷
- Milvus 数据（etcd / milvus）使用命名卷持久化（`COMMON_STORAGETYPE=local` 本地存储，无需 MinIO）

> Ollama 运行在宿主机（Windows/macOS Docker Desktop 通过 `host.docker.internal` 访问）；如 Ollama 也容器化部署，将 `OLLAMA_BASE_URL` 指向对应服务名即可。

## 七、向量库索引选择建议

| 场景 | 推荐索引 | 说明 |
|---|---|---|
| 默认（数据量中等，召回/内存均衡） | `IVF_SQ8` | 量化索引，比 IVF_FLAT 节省 ~4 倍内存 |
| 高召回要求（如答案溯源严格） | `HNSW` | 图索引，召回率最高，构建/内存开销较大 |
| 数据量极大、内存受限 | `IVF_PQ` | 乘积量化，压缩率最高 |
| 小数据集 / 精度优先 | `FLAT` | 暴力精确检索 |
| 本地备选（无 Milvus 环境） | FAISS `IVF_FLAT` | 自动降级，支持 IVF_SQ8 量化 |

## 八、评估口径

- **检索层**：Hit@k（top-k 是否命中预期来源）、MRR（首个正确结果的倒数排名）、关键词覆盖率
- **出题层**：结构合规率（题型 / 选项 / 答案 / 分值完整）、工种关键词覆盖率
- **判卷层**：判分准确率（客观题确定性比对）
- **综合准确率**：检索 0.35 + 判卷 0.35 + 出题 0.30 加权
- 报告输出：`eval/eval_report.json`

## 九、主要类职责

| 类 | 职责 |
|---|---|
| `SafetyTrainingAgent` | Agent 统一入口：构建 LangGraph create_agent 工具调用 Agent、5 个工具调度、会话记忆、自动降级、暴露业务服务 |
| `SafetyKnowledgeBase` | 文档分块、BGE 向量化、Milvus/FAISS、BM25、重排、工种保底与溯源 |
| `MilvusStore` / `FaissStore` | 向量库后端（Milvus：IVF/量化/图索引；FAISS：IVF/量化） |
| `MultiFormatLoader` / `LegacyConverter` | doc/docx/pdf/pptx/xlsx/txt 加载与老式格式转换 |
| `ExamBuilderService` | 检索 + 生成结构化试卷（在线 LLM / 离线抽题） |
| `MaterialBuilderService` | 检索 + 生成标准化培训资料（在线 LLM / 离线组装） |
| `GradingService` | 客观题判分 + 主观题评分 |
| `LedgerService` | 汇总生成合规培训台账 |
| `exporters.py` | 试卷 / 培训资料 / 台账 → Word（.docx）导出 |
| `OCREngine` | 试卷图片 OCR 文本提取 |
| `AgentEvaluator` | 三层离线 / 在线评估 |
