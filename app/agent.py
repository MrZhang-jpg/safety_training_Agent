# -*- coding: utf-8 -*-
"""
安全培训智能 Agent 核心类（面向对象）。

基于 LangChain 1.x（langgraph）的 create_agent 构建，通过 Function-Calling
调度 5 个自定义工具（LangChain StructuredTool），内置多轮对话记忆（InMemorySaver）。
对接本地 Ollama deepseek-r1:7b（GPU 推理），嵌入/重排走本地 BGE（DEVICE=cuda）。

能力：
- 知识库构建（多格式文档 → BGE → Milvus/FAISS 向量库）
- 按工种 + 三级教育层级生成试卷与培训资料（工具调用）
- OCR / 文本判卷（工具调用）
- 生成培训台账（工具调用）
- 混合检索（Milvus/FAISS 向量 + BM25 + BGE-reranker 重排 + 工种保底）
- Ollama 不可用 / 断网时自动降级到离线确定性引擎

说明：7b 及以上模型具备可靠的工具调用能力，因此采用 create_agent +
Function-Calling 的 Agent 编排模式；离线引擎 RuleBasedChatLLM 实现了
BaseChatModel + bind_tools 接口，可在同一套代码下无修改运行。
"""
from __future__ import annotations

from typing import Optional

from langchain.agents import create_agent

from .config import settings
from .document_loaders import MultiFormatLoader
from .llm_factory import RuleBasedChatLLM, get_llm, strip_thinking
from .ocr_engine import OCREngine
from .rag_retriever import SafetyKnowledgeBase
from .tools import SafetyTrainingTools

SYSTEM_PROMPT = """你是施工现场 EHS（环境、健康、安全）安全培训智能 Agent，服务于三级安全教育与岗前安全培训。

你可以调用以下工具：
1. search_safety_knowledge：查询内部安全操作规程、安全技术交底、事故案例、应急预案（参数：query 问题、craft_type 工种、edu_level 三级教育层级）；
2. generate_training_exam：按工种与三级教育层级生成考核试卷（含答案解析）；
3. generate_training_material：按工种与三级教育层级生成标准化培训资料（风险/规程/防护/应急章节）；
4. grade_paper：对 OCR 识别或粘贴的考生作答判分；
5. create_training_ledger：生成合规培训台账。

请按以下原则工作（先思考再行动）：
1. 先识别用户给出的工种与三级教育层级（公司级/项目级/班组级）；信息缺失时，工种按“通用工种”、层级按“班组级”处理；
2. **强制工具调用**：回答专业问题、出题、生成资料前，**必须先调用 search_safety_knowledge 检索**该工种相关的规程、交底、题库与事故案例资料；未拿到检索结果就直接作答属于违规行为。请使用可用工具，不要跳过工具直接回答；
3. 判卷时客观题（单选/多选/判断）以标准答案严格比对，主观题对照得分点给分并说明理由；
4. 严禁杜撰规范中的数值、距离、型号；所有结论须基于检索到的内部资料；
5. **最终回答只输出对用户有价值的结果本身**：不得复述工具调用过程、工具参数或原始 JSON，不得输出"我调用了某某工具"之类的过程描述；试卷、资料、判分、台账以工具返回的结构化内容为准，不要编造与资料冲突的内容。"""


class SafetyTrainingAgent:
    """安全培训智能 Agent 对外统一入口。"""

    def __init__(self, backend: Optional[str] = None):
        self.backend = (backend or settings.llm_backend or "ollama").lower()
        # 共享的 RAG 底座与 OCR
        self.kb = SafetyKnowledgeBase()
        self.ocr = OCREngine()
        # 主 LLM 与 Agent
        self.llm = get_llm(self.backend)
        self.agent = self._build_agent(self.llm, self.backend)
        self._fallback_agent = None

    # ---------- 构建 langgraph agent（工具调用模式）----------
    def _build_agent(self, llm, backend: str):
        checkpointer = self._create_checkpointer()
        tools = SafetyTrainingTools(self.kb, llm, self.ocr, backend).tools
        return create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )

    @staticmethod
    def _create_checkpointer():
        try:
            from langgraph.checkpoint.memory import InMemorySaver

            return InMemorySaver()
        except ImportError:  # 兼容旧版本命名
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()

    # ---------- 知识库 ----------
    def build_knowledge_base(self) -> dict:
        loader = MultiFormatLoader()
        docs = loader.load_all()
        if not docs:
            return {"chunks": 0, "message": "未加载到任何文档，请检查 DATA_DIR。"}
        n = self.kb.build(docs)
        return {"chunks": n, "loader_stats": loader.stats}

    # ---------- 降级 ----------
    def _get_fallback_agent(self):
        if self._fallback_agent is None:
            # 离线确定性引擎：可稳定调用工具、基于真实题库，零网络依赖
            fb_backend, fb_llm = "offline", RuleBasedChatLLM()
            self._fallback_agent = self._build_agent(fb_llm, fb_backend)
            print(f"[INFO] Ollama 不可用，已降级到 {fb_backend} 后端。")
        return self._fallback_agent

    # ---------- 对话执行 ----------
    def _invoke(self, agent, query: str, thread_id: str):
        """执行 agent 并返回 (最终文本, 是否发生过工具调用)。"""
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        final = result["messages"][-1]
        tool_called = any(
            type(m).__name__ == "ToolMessage" for m in result["messages"]
        )
        return strip_thinking(final.content), tool_called

    def run(self, query: str, thread_id: str = "default") -> str:
        try:
            text, tool_called = self._invoke(self.agent, query, thread_id)
            if tool_called:
                return text
            # 模型未发起工具调用（7b 的 Function-Calling 存在概率性）：
            # 强制走 RAG 兜底，保证回答始终有据可依，而非模型自由发挥。
            print("[INFO] 模型本轮未调用工具，触发强制检索兜底（RAG fallback）。")
            return self._rag_fallback(query)
        except Exception as exc:
            # 云端限流 / 网络错误时自动降级（offline 本身不会抛这类错误）
            if self.backend in ("auto", "ollama"):
                print(f"[WARN] {self.backend} 调用失败（{exc!r}），尝试降级。")
                return self._invoke(self._get_fallback_agent(), query, thread_id)[0]
            raise

    # ---------- 强制 RAG 兜底（无工具调用时保证有据回答）----------
    def _rag_fallback(self, query: str, k: Optional[int] = None) -> str:
        from .config import settings as _s

        k = k or _s.retrieve_k
        docs = self.kb.search(query, k=k)  # 内部自动识别工种 + 混合检索 + 重排
        ctx = "\n\n".join(
            f"【资料{i}·{d.source}】\n{d.content}" for i, d in enumerate(docs, 1)
        )
        prompt = (
            "你是施工现场 EHS（环境、健康、安全）安全培训智能 Agent。"
            "请严格基于以下【内部资料】回答用户问题，资料中没有的信息不要编造；"
            "回答应简洁、专业、分点。\n\n"
            f"【内部资料】\n{ctx}\n\n【用户问题】\n{query}"
        )
        resp = self.llm.invoke(prompt)
        return strip_thinking(str(resp.content))

    # ---------- 直接暴露业务服务（供 FastAPI 结构化接口复用）----------
    def get_exam(self, craft_type: str, edu_level: str,
                 counts: Optional[dict] = None):
        from .tools import ExamBuilderService

        return ExamBuilderService(self.kb).build(
            self.llm, self.backend, craft_type, edu_level, counts
        )

    def get_material(self, craft_type: str, edu_level: str):
        from .tools import MaterialBuilderService

        return MaterialBuilderService(self.kb).build(
            self.llm, self.backend, craft_type, edu_level
        )

    def grade(self, exam, paper):
        from .tools import GradingService

        return GradingService().grade(exam, paper, self.llm, self.backend)

    def make_ledger(self, craft_type, edu_level, trainee_names,
                    grade_results, **kwargs):
        from .tools import LedgerService

        return LedgerService().build(
            craft_type, edu_level, trainee_names, grade_results, **kwargs
        )
