# -*- coding: utf-8 -*-
"""
安全培训智能 Agent 评估器（面向对象）。

三个评估层面：
1. 检索层 retrieval：Hit@k（来源命中）、MRR、答案关键词覆盖率 —— 本地可真实测量；
2. 出题主 exam：试卷结构合规率（题量/答案/选项完整）、工种层级与关键词命中；
3. 判卷层 grading：用已知标准答案与预期分数的用例，校验客观题判分准确性。

评估结果写入 eval/eval_report.json，并在控制台打印汇总。
在线后端（ollama deepseek-r1:7b）跑完本脚本，即可得到线上生成质量指标；
离线后端（offline）下检索与判卷指标同样真实有效。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .agent import SafetyTrainingAgent
from .config import BASE_DIR, settings
from .schemas import (
    AnsweredQuestion, Exam, Question, SubmittedPaper,
)


class AgentEvaluator:
    """聚合检索 / 出题 / 判卷三类评测。"""

    def __init__(self, backend: str | None = None):
        self.agent = SafetyTrainingAgent(backend)
        self.backend = self.agent.backend
        self.eval_dir = BASE_DIR / "eval"
        with open(self.eval_dir / "eval_questions.json", encoding="utf-8") as f:
            self.dataset = json.load(f)

    # ============================ 检索评估 ============================
    def eval_retrieval(self) -> dict:
        rows = self.dataset["retrieval"]
        hits, rr_sum, term_sum = 0, 0.0, 0.0
        details = []

        for row in rows:
            docs = self.agent.kb.search(row["query"], k=settings.retrieve_k)
            sources = [d.source for d in docs]

            candidates = str(row["expected_source"]).split("|")
            rank = next((i + 1 for i, s in enumerate(sources)
                         if any(c in s for c in candidates)), 0)
            hit = rank > 0
            hits += int(hit)
            rr_sum += (1.0 / rank) if hit else 0.0

            joined = "\n".join(d.content for d in docs)
            terms = row["expected_terms"]
            term_hit = sum(1 for t in terms if t in joined) / len(terms)
            term_sum += term_hit

            details.append({
                "query": row["query"], "expected_source": row["expected_source"],
                "hit": hit, "rank": rank, "term_coverage": round(term_hit, 2),
                "top_sources": sources[:3],
            })

        n = len(rows)
        return {
            "samples": n,
            "hit_rate": round(hits / n, 4),
            "mrr": round(rr_sum / n, 4),
            "term_coverage": round(term_sum / n, 4),
            "details": details,
        }

    # ============================ 出题评估 ============================
    def eval_exam(self) -> dict:
        rows = self.dataset["exam"]
        valid_cnt, term_sum = 0, 0.0
        details = []

        for row in rows:
            exam = self.agent.get_exam(row["craft_type"], row["edu_level"])
            text = json.dumps(exam.model_dump(), ensure_ascii=False)

            structure_ok = (
                len(exam.questions) > 0
                and all(q.answer.strip() for q in exam.questions)
                and all(
                    q.options for q in exam.questions
                    if q.q_type in ("单选题", "多选题")
                )
                and exam.craft_type == row["craft_type"]
                and exam.edu_level == row["edu_level"]
            )
            valid_cnt += int(structure_ok)

            terms = row["must_contain"]
            term_hit = sum(1 for t in terms if t in text) / len(terms)
            term_sum += term_hit

            details.append({
                "craft_type": row["craft_type"], "edu_level": row["edu_level"],
                "structure_valid": structure_ok,
                "question_count": len(exam.questions),
                "q_type_dist": self._qtype_dist(exam),
                "term_coverage": round(term_hit, 2),
            })

        n = len(rows)
        return {
            "samples": n,
            "valid_rate": round(valid_cnt / n, 4),
            "term_coverage": round(term_sum / n, 4),
            "details": details,
        }

    @staticmethod
    def _qtype_dist(exam: Exam) -> dict:
        dist: dict[str, int] = {}
        for q in exam.questions:
            dist[q.q_type] = dist.get(q.q_type, 0) + 1
        return dist

    # ============================ 判卷评估 ============================
    def _standard_exam(self) -> Exam:
        qs = [
            Question(index=1, q_type="单选题", content="单选1",
                     options=["A. 正确项", "B. 干扰", "C. 干扰", "D. 干扰"],
                     answer="A", score=5),
            Question(index=2, q_type="单选题", content="单选2",
                     options=["A. 干扰", "B. 正确项", "C. 干扰", "D. 干扰"],
                     answer="B", score=5),
            Question(index=3, q_type="判断题", content="判断1（应为对）",
                     answer="对", score=5),
            Question(index=4, q_type="判断题", content="判断2（应为错）",
                     answer="错", score=5),
            Question(index=5, q_type="多选题", content="多选1",
                     options=["A. 项", "B. 项", "C. 项", "D. 项"],
                     answer="ABD", score=10),
        ]
        return Exam(title="判卷准确性测试卷", craft_type="通用工种",
                    edu_level="班组级", questions=qs, total_score=30)

    def eval_grading(self) -> dict:
        exam = self._standard_exam()
        # (描述, 各题学生答案, 预期总分)
        cases = [
            ("全部答对", ["A", "B", "对", "错", "ABD"], 30),
            ("第1题单选答错", ["C", "B", "对", "错", "ABD"], 25),
            ("第3题判断答错", ["A", "B", "错", "错", "ABD"], 25),
            ("第5题多选漏选判错", ["A", "B", "对", "错", "AB"], 20),
            ("多选答案乱序仍算对", ["A", "B", "对", "错", "dba"], 30),
        ]
        correct_cases = 0
        details = []

        for name, ans, expected in cases:
            paper = SubmittedPaper(
                trainee_name=name,
                answers=[
                    AnsweredQuestion(index=i + 1, q_type=q.q_type, student_answer=a)
                    for i, (q, a) in enumerate(zip(exam.questions, ans))
                ],
            )
            result = self.agent.grade(exam, paper)
            ok = abs(result.total_score - expected) < 0.01
            correct_cases += int(ok)
            details.append({
                "case": name, "expected_score": expected,
                "actual_score": result.total_score, "correct": ok,
            })

        n = len(cases)
        return {
            "samples": n,
            "score_accuracy": round(correct_cases / n, 4),
            "details": details,
        }

    # ============================ 汇总 ============================
    def run_all(self) -> dict:
        print(f"\n========== 开始评估（后端：{self.backend}）==========")
        print("[1/3] 检索层评估 ...")
        retrieval = self.eval_retrieval()
        print(f"  Hit@{settings.retrieve_k}={retrieval['hit_rate']:.1%}  "
              f"MRR={retrieval['mrr']:.3f}  关键词覆盖={retrieval['term_coverage']:.1%}")

        print("[2/3] 出题层评估 ...")
        exam = self.eval_exam()
        print(f"  结构合规率={exam['valid_rate']:.1%}  关键词覆盖={exam['term_coverage']:.1%}")

        print("[3/3] 判卷层评估 ...")
        grading = self.eval_grading()
        print(f"  判分准确率={grading['score_accuracy']:.1%}")

        # 综合准确率口径：检索 0.35 + 判卷 0.35 + 出题 0.30
        overall = (
            retrieval["hit_rate"] * 0.35
            + grading["score_accuracy"] * 0.35
            + exam["valid_rate"] * 0.30
        )
        report = {
            "backend": self.backend,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "retrieval": retrieval,
            "exam": exam,
            "grading": grading,
            "overall_accuracy": round(overall, 4),
        }

        out = self.eval_dir / "eval_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n========== 评估汇总 ==========")
        print(f"检索 Hit@k   : {retrieval['hit_rate']:.1%}")
        print(f"检索 MRR      : {retrieval['murr'] if False else retrieval['mrr']:.3f}")
        print(f"出题结构合规率: {exam['valid_rate']:.1%}")
        print(f"判卷判分准确率: {grading['score_accuracy']:.1%}")
        print(f"综合准确率    : {overall:.1%}")
        print(f"报告已保存    : {out}")
        return report
