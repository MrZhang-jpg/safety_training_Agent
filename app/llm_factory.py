# -*- coding: utf-8 -*-
"""
LLM 工厂（面向对象 + 可插拔后端）。

支持后端：
- ollama : 本地 Ollama deepseek-r1:7b（默认，完全本地推理 + GPU 加速，符合项目简历）
- offline: 内置确定性离线引擎 RuleBasedChatLLM（零网络、零额度，保证流水线永远可跑、可评测）
- auto   : 优先 ollama，运行时不可用则自动降级 offline

设计说明：
    在线后端（ollama）由 LangChain 原生提供；offline 引擎实现了最小的
    BaseChatModel 接口并支持 bind_tools，因此同一套 langgraph Agent 代码
    可以在两种后端上无修改运行，便于模型未启动时继续开发、联调与评测。

deepseek-r1 为推理模型，输出可能包含 响应（思考）块；提供
strip_thinking() 统一剥离，避免思考内容混入最终答案与 JSON 解析。
"""
from __future__ import annotations

import re
import uuid
from typing import Any, ClassVar, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import settings

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_thinking(text: str) -> str:
    """剥离 deepseek-r1 等推理模型的 <think> 思考块。"""
    if not text:
        return ""
    return _THINK_RE.sub("", str(text)).strip()


# =====================================================================
# 本地 Ollama 后端（默认）
# =====================================================================
def create_ollama_llm() -> BaseChatModel:
    """对接本地 Ollama deepseek-r1:7b（完全本地推理，GPU 加速）。"""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_predict=2048,
        timeout=180,
        # 抑制小参数模型输出重复（deepseek-r1 系易循环重复）
        repeat_penalty=1.3,
    )


# =====================================================================
# 内置离线引擎（确定性，支持工具调用）
# =====================================================================
# 工种关键词（长词优先，命中即映射为标准工种名）
_CRAFT_KEYWORDS: list[tuple[str, str]] = [
    ("人工挖孔桩", "人工挖孔桩工"), ("挖孔桩", "人工挖孔桩工"),
    ("施工升降机", "施工升降机司机"), ("门式起重机", "门式起重机司机"),
    ("司索信号", "司索信号工"), ("信号工", "司索信号工"), ("司索", "司索信号工"),
    ("电焊", "电焊工"), ("焊工", "电焊工"),
    ("架子", "架子工"), ("脚手架", "架子工"),
    ("钢筋", "钢筋工"),
    ("塔吊", "塔吊司机"), ("塔司", "塔吊司机"),
    ("起重", "起重吊装工"), ("吊装", "起重吊装工"), ("吊车", "起重吊装工"),
    ("挖机", "挖掘机司机"), ("挖掘机", "挖掘机司机"),
    ("钻孔桩", "钻孔桩工"), ("破桩", "破桩工"), ("桩基", "桩基工"), ("桩工", "桩基工"),
    ("挂篮", "挂篮安装工"), ("张拉", "张拉压浆工"), ("压浆", "张拉压浆工"),
    ("高处", "高处作业工"), ("高空", "高处作业工"),
    ("基坑", "基坑作业工"), ("钢板桩", "钢板桩工"), ("钢支撑", "钢支撑安装工"),
    ("钢立柱", "钢立柱安装工"), ("斜拉索", "斜拉索工"),
    ("混凝土", "混凝土工"), ("浇砼", "混凝土工"), ("浇筑", "混凝土工"),
    ("拌合站", "拌合站操作员"), ("搅拌机", "搅拌机操作手"), ("搅拌", "搅拌机操作手"),
    ("罐车", "罐车司机"), ("装载机", "装载机司机"), ("渣土", "渣土车司机"),
    ("车辆驾驶", "车辆驾驶员"), ("机动车", "厂内机动车司机"),
    ("爆破", "爆破工"), ("隧道", "隧道作业工"),
    ("桥涵", "桥涵作业工"), ("桥梁", "桥涵作业工"),
    ("桥面系", "桥面系作业工"), ("连续梁", "连续梁作业工"),
    ("路基", "路基作业工"), ("电工", "电工"),
    ("机械", "机械设备操作员"),
]

_EDU_KEYWORDS: list[tuple[str, str]] = [
    ("公司级", "公司级"), ("公司", "公司级"), ("一级", "公司级"),
    ("项目级", "项目级"), ("项目部", "项目级"), ("工程级", "项目级"), ("二级", "项目级"),
    ("班组级", "班组级"), ("班组", "班组级"), ("岗前", "班组级"), ("三级", "班组级"),
]


class RuleBasedChatLLM(BaseChatModel):
    """确定性离线对话引擎：用规则做意图识别并发起工具调用。

    仅用于无网络 / Ollama 不可用时的联调与评测，不替代线上模型。
    """

    craft_keywords: ClassVar[list[tuple[str, str]]] = _CRAFT_KEYWORDS
    edu_keywords: ClassVar[list[tuple[str, str]]] = _EDU_KEYWORDS

    # 让 LangChain 把工具列表通过 kwargs 传入
    def bind_tools(self, tools: list, *, tool_choice: Optional[Any] = None, **kwargs):
        return self.bind(bound_tools=tools)

    @property
    def _llm_type(self) -> str:
        return "rule_based_offline"

    @staticmethod
    def _extract_craft(text: str) -> str:
        for kw, name in _CRAFT_KEYWORDS:
            if kw in text:
                return name
        return "通用工种"

    @staticmethod
    def _extract_edu(text: str) -> str:
        for kw, name in _EDU_KEYWORDS:
            if kw in text:
                return name
        return "班组级"

    def _plan(self, query: str) -> tuple[str, dict]:
        """根据用户文本决定调用哪个工具。"""
        craft = self._extract_craft(query)
        edu = self._extract_edu(query)

        if re.search(r"试卷|试题|考题|出题|出一套|考卷|题目|考核题", query):
            return "generate_training_exam", {"craft_type": craft, "edu_level": edu}
        if re.search(r"培训资料|讲义|学习资料|培训材料|教材|交底资料", query):
            return "generate_training_material", {"craft_type": craft, "edu_level": edu}
        if re.search(r"批改|判分|阅卷|打分|评分|改卷|对答案|ocr|识别", query, re.I):
            return "grade_paper", {"student_text": query, "craft_type": craft, "edu_level": edu}
        if re.search(r"台账|登记|留痕|培训记录", query):
            return "create_training_ledger", {
                "craft_type": craft, "edu_level": edu, "trainee_names": []
            }
        return "search_safety_knowledge", {"query": query, "craft_type": craft}

    def _generate(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        bound_tools = kwargs.get("bound_tools", [])
        available = {getattr(t, "name", ""): t for t in bound_tools}
        last = messages[-1]

        # 工具已执行 → 直接整理工具结果为最终自然语言回复
        if isinstance(last, ToolMessage):
            prefix = ""
            if last.name == "generate_training_exam":
                prefix = "已根据工种与三级教育层级生成考核试卷（含参考答案与解析）：\n\n"
            elif last.name == "generate_training_material":
                prefix = "已生成标准化安全培训资料：\n\n"
            elif last.name == "create_training_ledger":
                prefix = "已生成安全培训台账初稿：\n\n"
            elif last.name == "grade_paper":
                prefix = "已完成试卷判分：\n\n"
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=prefix + str(last.content)))]
            )

        # 找到最后一条人类消息做规划
        human_text = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                human_text = str(m.content)
                break

        tool_name, args = self._plan(human_text)
        if tool_name not in available:
            tool_name = "search_safety_knowledge"
            args = {"query": human_text}
            if tool_name not in available:
                return ChatResult(generations=[
                    ChatGeneration(message=AIMessage(content="当前未挂载任何工具，无法处理该请求。"))
                ])

        tool_call = {
            "name": tool_name,
            "args": args,
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        }
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[tool_call]))]
        )


# =====================================================================
# 工厂入口
# =====================================================================
def _ollama_alive() -> bool:
    """探测本地 Ollama 服务是否可用。"""
    import requests

    try:
        r = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_llm(backend: Optional[str] = None) -> BaseChatModel:
    """按指定后端创建 LLM；默认读取配置 LLM_BACKEND（ollama）。"""
    backend = (backend or settings.llm_backend or "ollama").lower()

    if backend == "ollama":
        return create_ollama_llm()
    if backend == "offline":
        return RuleBasedChatLLM()
    if backend == "auto":
        # ollama 优先；本地 ollama 在线则用 ollama；否则离线引擎
        if _ollama_alive():
            return create_ollama_llm()
        return RuleBasedChatLLM()

    raise ValueError(f"未知 LLM 后端: {backend}（可选 ollama/offline/auto）")
