# -*- coding: utf-8 -*-
"""
Agent 自定义工具与业务服务（面向对象）。

业务服务：
- ExamBuilderService : 检索 + LLM 生成结构化试卷（含答案解析）；离线时用确定性真题/要点工厂
- GradingService     : 客观题（单选/多选/判断）程序确定性比对，主观题 LLM 辅助评分（离线用要点覆盖率）
- LedgerService      : 生成合规培训台账

工具封装：
- SafetyTrainingTools 把上述服务暴露为 4 个 LangChain StructuredTool：
  search_safety_knowledge / generate_training_exam / grade_paper / create_training_ledger
"""
from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.tools import StructuredTool

from .config import settings
from .llm_factory import strip_thinking
from .rag_retriever import SafetyKnowledgeBase, ScoredDoc
from .schemas import (
    Exam, GradeResult, GradedQuestion, MaterialSection, Question, SubmittedPaper,
    AnsweredQuestion, TrainingLedger, TrainingMaterial,
)


# =====================================================================
# 通用工具
# =====================================================================
def _parse_json_lenient(text: str) -> dict:
    """从容错文本中提取 JSON（去掉 markdown 代码块、前后说明、<think> 思考块）。"""
    if not text:
        return {}
    text = strip_thinking(text).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    # 截取最外层花括号
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


# =====================================================================
# 出题服务
# =====================================================================
_DEFAULT_COUNTS = {"单选题": 5, "多选题": 3, "判断题": 3, "简答题": 2}

_SAFETY_HINT = re.compile(r"(必须|严禁|不得|应当|应该|禁止|务必|需要|不准)")


class ExamBuilderService:
    """根据工种 + 三级教育层级生成试卷。"""

    def __init__(self, kb: SafetyKnowledgeBase):
        self.kb = kb

    def build(self, llm, backend: str, craft_type: str, edu_level: str,
              counts: Optional[dict] = None) -> Exam:
        counts = counts or _DEFAULT_COUNTS
        docs = self.kb.search(
            f"{craft_type} {edu_level} 三级安全教育 安全操作规程 试题",
            k=settings.retrieve_k, craft_type=craft_type,
        )
        context = SafetyKnowledgeBase.format_context(docs)
        references = sorted({d.source for d in docs})

        if backend in ("ollama",):
            try:
                return self._build_online(llm, craft_type, edu_level, counts,
                                          context, references)
            except Exception as exc:
                print(f"[WARN] 在线出题失败，降级为离线要点出题: {exc}")
        return self._build_offline(craft_type, edu_level, counts, docs)

    # ---------- 在线：LLM 命题 ----------
    def _build_online(self, llm, craft_type, edu_level, counts, context,
                      references) -> Exam:
        cnt = "、".join(f"{k}{v}道" for k, v in counts.items() if v)
        prompt = f"""你是建筑施工安全培训命题专家。请仅依据下面检索到的本工种安全资料，
为「{craft_type}」的「{edu_level}」三级安全教育命制考核试卷。

题型与数量：{cnt}。
要求：
1. 题目必须基于给定资料，严禁杜撰规范数值、型号、距离；
2. 单选/多选给出 A/B/C/D 选项，多选至少两个正确项；判断题陈述明确；
3. 每题给出参考答案，选择题答案为选项字母，判断题答案为“对/错”；
4. 每题给出解析，简答题在答案中列出得分点。

仅输出 JSON，结构：
{{"title":"...","craft_type":"{craft_type}","edu_level":"{edu_level}",
"total_score":100,
"questions":[{{"index":1,"q_type":"单选题","content":"...","options":["A. ...","B. ..."],
"answer":"A","analysis":"...","score":5}}],
"references":[]}}

检索资料：
{context}"""
        resp = llm.invoke(prompt)
        data = _parse_json_lenient(resp.content if hasattr(resp, "content") else str(resp))
        return self._exam_from_dict(data, references)

    @staticmethod
    def _exam_from_dict(data: dict, references: list[str]) -> Exam:
        questions = [
            Question(
                index=q.get("index", i + 1),
                q_type=q.get("q_type", "简答题"),
                content=q.get("content", ""),
                options=q.get("options", []) or [],
                answer=str(q.get("answer", "")).strip(),
                analysis=q.get("analysis", ""),
                score=float(q.get("score", 5)),
            )
            for i, q in enumerate(data.get("questions", []))
            if q.get("content")
        ]
        if not questions:
            raise ValueError("LLM 未返回有效题目。")
        total = sum(q.score for q in questions)
        return Exam(
            title=data.get("title", "安全培训考核试卷"),
            craft_type=data.get("craft_type", ""),
            edu_level=data.get("edu_level", ""),
            questions=questions,
            total_score=total or 100.0,
            references=data.get("references") or references,
        )

    # ---------- 离线：基于真实资料要点的确定性命题 ----------
    def _build_offline(self, craft_type, edu_level, counts, docs: list[ScoredDoc]) -> Exam:
        sentences: list[tuple[str, str]] = []
        for d in docs:
            for s in re.split(r"[。\n]", d.content):
                s = s.strip()
                if len(s) >= 8 and _SAFETY_HINT.search(s):
                    sentences.append((s, d.source))

        questions: list[Question] = []
        idx = 1

        # 判断题：要点句为“对”，翻转情态词制造“错”项
        n_judge = counts.get("判断题", 0)
        for i in range(n_judge):
            if i >= len(sentences):
                break
            s, src = sentences[i]
            if i % 2 == 0:
                content, answer = s, "对"
                analysis = f"符合安全规程要求。来源：{src}"
            else:
                flipped = s.replace("必须", "可以不必").replace("严禁", "可以") \
                           .replace("不得", "可以").replace("禁止", "允许") \
                           .replace("应当", "无需").replace("不准", "可以")
                content, answer = flipped, "错"
                analysis = f"正确要求为：{s}。来源：{src}"
            questions.append(Question(index=idx, q_type="判断题", content=content,
                                      answer=answer, analysis=analysis, score=5))
            idx += 1

        # 单选题：取要点为题干，正确项为关键要求，其余为通用干扰
        n_single = counts.get("单选题", 0)
        pool = sentences[len(questions):]
        distractors = ["可根据现场情况自行决定", "为提高效率可省略", "无需采取防护措施"]
        for i in range(n_single):
            if i >= len(pool):
                break
            s, src = pool[i]
            m = _SAFETY_HINT.search(s)
            tail = s[m.start():] if m else s
            options = [f"{chr(65)}. {tail}"] + [
                f"{chr(66+j)}. {d}" for j, d in enumerate(distractors)]
            questions.append(Question(
                index=idx, q_type="单选题",
                content=f"关于{craft_type}作业，下列说法正确的是？",
                options=options, answer="A",
                analysis=f"依据规程：{s}。来源：{src}", score=5))
            idx += 1

        # 多选题离线难度大，用要点组合
        n_multi = counts.get("多选题", 0)
        for i in range(n_multi):
            base = len(questions)
            chosen = sentences[base:base + 3]
            if len(chosen) < 2:
                break
            opts = [f"{chr(65+j)}. {c[0][:40]}" for j, c in enumerate(chosen)]
            opts.append(f"{chr(65+len(chosen))}. 以上要求均无关紧要")
            letters = "".join(chr(65 + j) for j in range(len(chosen)))
            questions.append(Question(
                index=idx, q_type="多选题",
                content=f"{craft_type}作业时，下列做法正确的有？",
                options=opts, answer=letters,
                analysis="以上均为规程明确要求，最后一项错误。", score=6))
            idx += 1

        # 简答题：答案为检索到的要点清单
        n_essay = counts.get("简答题", 0)
        if n_essay and sentences:
            points = [s for s, _ in sentences[:6]]
            questions.append(Question(
                index=idx, q_type="简答题",
                content=f"简述{craft_type}{edu_level}安全作业的主要要求。",
                answer="；".join(points),
                analysis="得分点：每答出一项安全要求得相应分数。", score=15))
            idx += 1

        if not questions:
            # 极端兜底：给一道简答
            questions.append(Question(
                index=1, q_type="简答题",
                content=f"简述{craft_type}安全作业注意事项。",
                answer="遵守安全操作规程，正确佩戴劳动防护用品，服从现场安全管理。",
                analysis="无检索要点时的通用安全要求。", score=100))

        total = sum(q.score for q in questions)
        title = f"{craft_type}{edu_level}三级安全教育考核试卷"
        return Exam(title=title, craft_type=craft_type, edu_level=edu_level,
                    questions=questions, total_score=float(total),
                    references=sorted({src for _, src in sentences}))


# =====================================================================
# 培训资料服务
# =====================================================================
class MaterialBuilderService:
    """根据工种 + 三级教育层级生成标准化安全培训资料（章节化）。"""

    def __init__(self, kb: SafetyKnowledgeBase):
        self.kb = kb

    def build(self, llm, backend: str, craft_type: str, edu_level: str) -> TrainingMaterial:
        docs = self.kb.search(
            f"{craft_type} {edu_level} 安全培训 安全操作规程 应急处置 事故案例",
            k=8, craft_type=craft_type,
        )
        context = SafetyKnowledgeBase.format_context(docs)
        references = sorted({d.source for d in docs})

        if backend in ("ollama",):
            try:
                return self._build_online(llm, craft_type, edu_level, context, references)
            except Exception as exc:
                print(f"[WARN] 在线生成培训资料失败，降级为离线资料组装: {exc}")
        return self._build_offline(craft_type, edu_level, docs)

    # ---------- 在线：LLM 生成 ----------
    def _build_online(self, llm, craft_type, edu_level, context, references) -> TrainingMaterial:
        prompt = f"""你是建筑施工安全培训专家。请仅依据下面检索到的本工种安全资料，
为「{craft_type}」的「{edu_level}」三级安全教育编写一份标准化培训资料。

资料章节建议（可微调）：
1. 作业主要风险与危害因素
2. 安全操作规程要点（含禁止/必须类要求）
3. 劳动防护用品配备要求
4. 应急处置与事故案例警示

要求：
1. 内容必须基于给定资料，严禁杜撰规范数值、型号、距离；
2. 条理清晰，每章若干条要点，便于一线人员学习；
3. summary 给出 1-2 句培训目的与适用范围概述。

仅输出 JSON，结构：
{{"title":"...","craft_type":"{craft_type}","edu_level":"{edu_level}",
"summary":"...",
"sections":[{{"heading":"...","content":"1. ...\\n2. ..."}}],
"references":[]}}

检索资料：
{context}"""
        resp = llm.invoke(prompt)
        data = _parse_json_lenient(resp.content if hasattr(resp, "content") else str(resp))
        sections = [
            MaterialSection(heading=s.get("heading", ""), content=s.get("content", ""))
            for s in data.get("sections", []) if s.get("heading")
        ]
        if not sections:
            raise ValueError("LLM 未返回有效章节。")
        return TrainingMaterial(
            title=data.get("title", f"{craft_type}{edu_level}安全培训资料"),
            craft_type=data.get("craft_type") or craft_type,
            edu_level=data.get("edu_level") or edu_level,
            summary=data.get("summary", ""),
            sections=sections,
            references=data.get("references") or references,
        )

    # ---------- 离线：基于检索资料确定性组装 ----------
    def _build_offline(self, craft_type, edu_level, docs: list[ScoredDoc]) -> TrainingMaterial:
        sections: list[MaterialSection] = []

        # 一、作业主要风险与危害因素（含“风险/危险/危害/伤害/事故”的句子）
        risk_sent = [s.strip() for d in docs for s in re.split(r"[。\n]", d.content)
                     if len(s.strip()) >= 6 and re.search(r"风险|危险|危害|伤害|事故|坠落|触电|中毒|坍塌|机械伤害", s)]
        if risk_sent:
            uniq = list(dict.fromkeys(risk_sent[:10]))
            sections.append(MaterialSection(
                heading="作业主要风险与危害因素",
                content="\n".join(f"{i}. {s}。" for i, s in enumerate(uniq, 1)),
            ))

        # 二、安全操作规程要点（含“必须/严禁/不得/应当/禁止”的句子）
        rule_sent = [s.strip() for d in docs for s in re.split(r"[。\n]", d.content)
                     if len(s.strip()) >= 6 and _SAFETY_HINT.search(s)]
        if rule_sent:
            uniq = list(dict.fromkeys(rule_sent[:14]))
            sections.append(MaterialSection(
                heading="安全操作规程要点",
                content="\n".join(f"{i}. {s}。" for i, s in enumerate(uniq, 1)),
            ))

        # 三、劳动防护用品配备要求
        ppe_sent = [s.strip() for d in docs for s in re.split(r"[。\n]", d.content)
                    if len(s.strip()) >= 6 and re.search(r"防护|安全帽|安全带|手套|护目镜|防毒|口罩|劳保|PPE", s)]
        if ppe_sent:
            uniq = list(dict.fromkeys(ppe_sent[:8]))
            sections.append(MaterialSection(
                heading="劳动防护用品配备要求",
                content="\n".join(f"{i}. {s}。" for i, s in enumerate(uniq, 1)),
            ))

        # 四、应急处置要求
        emg_sent = [s.strip() for d in docs for s in re.split(r"[。\n]", d.content)
                    if len(s.strip()) >= 6 and re.search(r"应急|急救|报警|撤离|救援|处置|预案|火灾", s)]
        if emg_sent:
            uniq = list(dict.fromkeys(emg_sent[:8]))
            sections.append(MaterialSection(
                heading="应急处置与报警要求",
                content="\n".join(f"{i}. {s}。" for i, s in enumerate(uniq, 1)),
            ))

        if not sections:
            sections.append(MaterialSection(
                heading="基本安全要求",
                content="遵守安全操作规程，正确佩戴劳动防护用品，服从现场安全管理，"
                        "发现隐患及时报告。",
            ))

        title = f"{craft_type}{edu_level}安全培训资料"
        summary = (f"本资料面向{craft_type}作业人员{edu_level}三级安全教育，"
                   f"涵盖作业风险、操作规程、劳动防护与应急处置要点。")
        return TrainingMaterial(
            title=title, craft_type=craft_type, edu_level=edu_level,
            summary=summary, sections=sections,
            references=sorted({d.source for d in docs}),
        )


# =====================================================================
# 判卷服务
# =====================================================================
_JUDGE_TRUE = {"对", "正确", "是", "√", "t", "true", "1", "对的", "正确的"}
_JUDGE_FALSE = {"错", "错误", "否", "×", "x", "f", "false", "0", "错的", "错误的"}


def _norm_letters(ans: str) -> str:
    letters = sorted(re.findall(r"[A-Za-z]", str(ans)))
    return "".join(letters).upper()


def _norm_judge(ans: str) -> Optional[str]:
    a = str(ans).strip().lower().replace(" ", "")
    if a in _JUDGE_TRUE:
        return "对"
    if a in _JUDGE_FALSE:
        return "错"
    return None


class GradingService:
    """试卷判分：客观题确定性比对 + 主观题 LLM/规则评分。"""

    @staticmethod
    def _is_correct(q: Question, student: str) -> bool:
        student = str(student).strip()
        if q.q_type == "判断题":
            return _norm_judge(student) == _norm_judge(q.answer)
        if q.q_type == "多选题":
            return _norm_letters(student) == _norm_letters(q.answer)
        if q.q_type == "单选题":
            return bool(_norm_letters(student)) and _norm_letters(student) == _norm_letters(q.answer)
        return False

    @staticmethod
    def _subjective_offline(q: Question, student: str) -> tuple[float, str]:
        points = [p for p in re.split(r"[。；;、]", q.answer) if len(p.strip()) >= 2]
        if not points:
            return 0.0, "缺少评分要点。"
        hit = sum(1 for p in points if p.strip()[:6] in student or any(
            w in student for w in re.findall(r"[\u4e00-\u9fa5]{2,}", p)[:2]))
        ratio = min(hit / len(points), 1.0)
        return round(q.score * ratio, 1), f"命中 {hit}/{len(points)} 个得分点。"

    def _subjective_online(self, llm, q: Question, student: str,
                           ) -> tuple[float, str]:
        prompt = f"""你是安全培训阅卷老师。请按得分点给简答题评分，满分 {q.score} 分。
题目：{q.content}
参考答案/得分点：{q.answer}
考生作答：{student}
仅输出 JSON：{{"score": 数字, "reason": "简述扣分/给分理由"}}"""
        try:
            resp = llm.invoke(prompt)
            data = _parse_json_lenient(resp.content if hasattr(resp, "content") else str(resp))
            score = float(data.get("score", 0))
            score = max(0.0, min(q.score, score))
            return round(score, 1), data.get("reason", "")
        except Exception:
            return self._subjective_offline(q, student)

    def grade(self, exam: Exam, paper: SubmittedPaper, llm, backend: str) -> GradeResult:
        by_index = {a.index: a.student_answer for a in paper.answers}
        details: list[GradedQuestion] = []
        obj_score = sub_score = 0.0

        for q in exam.questions:
            student = by_index.get(q.index, "")
            if q.q_type in ("单选题", "多选题", "判断题"):
                ok = self._is_correct(q, student)
                score = q.score if ok else 0.0
                obj_score += score
                comment = "回答正确" if ok else f"回答错误，正确答案：{q.answer}"
                details.append(GradedQuestion(
                    index=q.index, q_type=q.q_type, standard_answer=q.answer,
                    student_answer=student, full_score=q.score, score=score,
                    correct=ok, comment=comment))
            else:
                if backend in ("ollama",):
                    score, comment = self._subjective_online(llm, q, student)
                else:
                    score, comment = self._subjective_offline(q, student)
                sub_score += score
                details.append(GradedQuestion(
                    index=q.index, q_type=q.q_type, standard_answer=q.answer,
                    student_answer=student, full_score=q.score, score=score,
                    correct=score >= q.score * 0.6, comment=comment))

        total = round(obj_score + sub_score, 1)
        full = sum(q.score for q in exam.questions)
        return GradeResult(
            trainee_name=paper.trainee_name, total_score=total, full_score=float(full),
            objective_score=round(obj_score, 1), subjective_score=round(sub_score, 1),
            passed=total >= 60.0, pass_line=60.0,
            details=details,
        )


# =====================================================================
# 台账服务
# =====================================================================
class LedgerService:
    """生成安全培训台账。"""

    def build(self, craft_type: str, edu_level: str,
              trainee_names: list[str], grade_results: list[GradeResult],
              **kwargs) -> TrainingLedger:
        n = len(trainee_names)
        if grade_results:
            scores = [g.total_score for g in grade_results]
            avg = round(sum(scores) / len(scores), 1)
            passed = sum(1 for g in grade_results if g.passed)
            summary = (f"本次培训共 {n or len(grade_results)} 人参加，"
                       f"平均分 {avg}，合格 {passed} 人。")
        else:
            avg, passed, summary = 0.0, 0, f"本次培训共 {n} 人参加，考试结果待登记。"

        return TrainingLedger(
            craft_type=craft_type, edu_level=edu_level,
            training_topic=kwargs.get("training_topic", f"{craft_type}{edu_level}三级安全教育"),
            trainer=kwargs.get("trainer", ""), location=kwargs.get("location", ""),
            training_date=kwargs.get("training_date", ""),
            trainee_names=trainee_names, trainee_count=n,
            average_score=avg, pass_count=passed, grade_summary=summary,
            grade_results=grade_results,
        )


# =====================================================================
# Agent 工具封装
# =====================================================================
class SafetyTrainingTools:
    """把业务服务封装为 Agent 可调用的 LangChain 工具。"""

    def __init__(self, kb: SafetyKnowledgeBase, llm, ocr=None, backend: str = ""):
        self.kb = kb
        self.llm = llm
        self.ocr = ocr
        self.backend = (backend or settings.llm_backend).lower()
        self.exam_service = ExamBuilderService(kb)
        self.material_service = MaterialBuilderService(kb)
        self.grade_service = GradingService()
        self.ledger_service = LedgerService()
        self.tools = self._build_tools()

    def _build_tools(self) -> list:
        return [
            StructuredTool.from_function(
                self.search_knowledge, name="search_safety_knowledge",
                description=(
                    "查询企业内部安全操作规程、安全技术交底、事故案例、应急预案等。"
                    "当用户询问安全要求、规范、做法，或需要为出题/判卷准备资料时调用。"
                    "输入 query 为自然语言问题，craft_type 为工种（可空）。"),
            ),
            StructuredTool.from_function(
                self.generate_exam, name="generate_training_exam",
                description=(
                    "按工种(craft_type)与三级教育层级(edu_level：公司级/项目级/班组级)"
                    "自动生成标准化考核试卷，含单选、多选、判断、简答题及参考答案、解析。"
                    "用户要求出题、出试卷、生成试题时调用。"),
            ),
            StructuredTool.from_function(
                self.generate_material, name="generate_training_material",
                description=(
                    "按工种(craft_type)与三级教育层级(edu_level：公司级/项目级/班组级)"
                    "自动生成标准化安全培训资料（作业风险/操作规程/劳动防护/应急处置章节）。"
                    "用户要求生成培训资料、讲义、培训材料、学习资料时调用。"),
            ),
            StructuredTool.from_function(
                self.grade_paper, name="grade_paper",
                description=(
                    "对 OCR 识别或粘贴的考生作答文本(student_text)自动判分："
                    "客观题程序比对标准答案，主观题按评分规则辅助评分。"
                    "用户要求批改、判卷、评分、阅卷时调用。"),
            ),
            StructuredTool.from_function(
                self.create_ledger, name="create_training_ledger",
                description=(
                    "根据工种、教育层级、参训人员名单(trainee_names)及考试结果，"
                    "生成合规的安全培训台账。用户要求生成台账、培训记录时调用。"),
            ),
        ]

    # ---------- 工具实现 ----------
    def search_knowledge(self, query: str, craft_type: str = "") -> str:
        docs = self.kb.search(query, k=settings.retrieve_k, craft_type=craft_type or None)
        if not docs:
            return "未在安全知识库中检索到相关内容，请补充资料或换一种表述。"
        return SafetyKnowledgeBase.format_context(docs)

    def generate_exam(self, craft_type: str, edu_level: str) -> str:
        exam = self.exam_service.build(self.llm, self.backend, craft_type, edu_level)
        return json.dumps(exam.model_dump(), ensure_ascii=False, indent=2)

    def generate_material(self, craft_type: str, edu_level: str) -> str:
        material = self.material_service.build(self.llm, self.backend, craft_type, edu_level)
        return json.dumps(material.model_dump(), ensure_ascii=False, indent=2)

    def grade_paper(self, student_text: str = "", craft_type: str = "",
                    edu_level: str = "班组级") -> str:
        if not student_text or not student_text.strip():
            return "请上传试卷照片（OCR 识别）或粘贴考生作答内容后再进行判分。"

        # 1) 生成/获取标准试卷
        exam = self.exam_service.build(self.llm, self.backend,
                                       craft_type or "通用工种", edu_level)

        # 2) 从文本中抽取学生答案，形如 “1.A / 1、A / 第1题 A / 1 对”
        answers: list[AnsweredQuestion] = []
        # 先按“题号”切分
        pattern = re.compile(
            r"(?:第?\s*)?(\d{1,2})\s*[\.、:：]?\s*"
            r"([A-Za-z]{1,6}|对|错|正确|错误|√|×|x|X|是|否)")
        seen_idx = set()
        for m in pattern.finditer(student_text):
            idx = int(m.group(1))
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            q = next((qq for qq in exam.questions if qq.index == idx), None)
            qtype = q.q_type if q else "单选题"
            answers.append(AnsweredQuestion(index=idx, q_type=qtype,
                                            student_answer=m.group(2).strip()))

        if not answers:
            return ("已识别试卷文本，但未能解析出“题号-答案”，请按如 “1.A、2.对、3.BCD” "
                    "的格式提供作答内容。")

        paper = SubmittedPaper(craft_type=craft_type, edu_level=edu_level,
                               answers=answers)
        result = self.grade_service.grade(exam, paper, self.llm, self.backend)
        return json.dumps(result.model_dump(), ensure_ascii=False, indent=2)

    def create_ledger(self, craft_type: str, edu_level: str,
                      trainee_names: list = None) -> str:
        names = trainee_names or []
        ledger = self.ledger_service.build(craft_type, edu_level, names, [])
        return json.dumps(ledger.model_dump(), ensure_ascii=False, indent=2)
