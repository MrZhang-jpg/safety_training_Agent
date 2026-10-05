# -*- coding: utf-8 -*-
"""
安全培训智能 Agent —— 严格评估器（Strict Evaluator）。

在 app/evaluate.py 原有三层评测（检索 / 出题 / 判卷）基础上，按「加严评测方案」升级：

1. 检索层
   - 测试集由 12 条扩到 80 条，含口语化、错别字、跨工种、场景症状式、数值条款式、制度流程式对抗 query；
   - 指标从单一 Hit@6 扩展为 Hit@1 / Hit@3 / Hit@6 + MRR；
   - 关键词覆盖拆为两种口径：
       · term_coverage_topk ：原口径，术语出现在 top-k 任意文本中（易被通用大文件摊薄命中）；
       · term_coverage_gold ：严口径，术语必须出现在 top-k 中「命中的期望来源文件」分块内（真正有据可依）；
   - 按 query 类别分组报告，暴露「开卷式」满分背后的类别短板。

2. 出题层
   - 工种由 5 个扩到 8 个；
   - 除「结构合规率」外新增：
       · schema_rate    ：确定性一致性校验（答案字母是否在选项范围内、单选是否恰一项、
                          多选是否≥2 项、判断答案是否∈{对,错}、简答是否有≥2 得分点、
                          题干/选项是否重复、解析是否缺失、题型分布是否等于要求）；
       · fact_coverage  ：知识点（expected_facts）覆盖率；
       · 事实核查（LLM-as-judge）：逐题判定「题干有效性 / 答案正确性 / 依据充分性」，
                          报告 fact_ok_rate（三项全过）。
   - 记录 online_rate（在线 LLM 出题成功率，未静默降级为离线兜底的比例）。

3. 判卷层
   - 标准卷由 5 道客观题扩为「3 道客观题 + 2 道简答题（各含 6 个得分点、各 30 分）」；
   - 用例由 5 条扩到 14 条，每条给出**人工期望分数**；
   - 指标：客观题判分准确率、主观题 MAE/RMSE、总分 MAE、偏差（bias）、
           以及「机器评分 vs 人工评分」的 Pearson r 与 Spearman ρ（评分一致性）。

4. 稳定性
   - 随机性指标（出题、主观题评分）支持多轮重复运行，报告均值 ± 标准差与极差。

结果写入 eval/eval_report_strict.json，并可渲染为 eval/eval_report_strict.md。
"""
from __future__ import annotations

import io
import json
import math
import re
import statistics
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .agent import SafetyTrainingAgent
from .config import BASE_DIR, settings
from .schemas import AnsweredQuestion, Exam, Question, SubmittedPaper

STRICT_DATASET = "eval/eval_questions_strict.json"

# 出题请求的题型分布（与 app/tools.py._DEFAULT_COUNTS 一致）
REQUIRED_COUNTS = {"单选题": 5, "多选题": 3, "判断题": 3, "简答题": 2}

_JUDGE_PROMPT = """你是建筑施工安全培训考核试卷的独立审核专家。请逐题审核下面这份试卷，判断每道题是否合格。

审核标准（三项分别判断）：
1. stem_ok 题干有效性：题干表述完整、无自相矛盾、无错别字、有明确且唯一的考点；
2. answer_ok 答案正确性：参考答案与题干、选项自洽。单选题只能有一个正确项且答案字母与其一致；
   多选题至少两个正确项且答案字母与正确项一致；判断题答案只能是"对"或"错"且与题干陈述相符；
   简答题答案须列出可核查的得分点，不得空泛；选项中不得出现两个互相冲突的"正确项"；
3. grounded_ok 依据充分性：题目内容属于建筑施工安全常识或规程要求，未编造明显不存在的数据、距离、型号或标准号。

试卷（JSON）：
{exam}
{context}

只输出一个 JSON 数组，不要输出任何其他文字，每个元素形如：
{{"index": 1, "stem_ok": true, "answer_ok": true, "grounded_ok": true, "reason": "不超过20字"}}
数组中每个题号都必须出现一次，顺序与试卷一致。"""


def _parse_json_loose(text: str) -> Any:
    """从模型输出中尽力提取 JSON（容忍 markdown 代码块与前后说明）。"""
    if not text:
        raise ValueError("空响应")
    s = str(text).strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        i, j = s.find(open_ch), s.rfind(close_ch)
        if i != -1 and j > i:
            try:
                return json.loads(s[i:j + 1])
            except Exception:
                continue
    raise ValueError(f"无法解析 JSON：{s[:120]!r}")


def _rankdata(xs: list[float]) -> list[float]:
    """平均秩（处理并列），用于 Spearman。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 4)


def _spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _mean_std(xs: list[float]) -> dict:
    xs = [float(x) for x in xs]
    if not xs:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.fmean(xs), 4),
        "std": round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0,
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
        "n": len(xs),
    }


def aggregate_retrieval(details: list[dict], k: int) -> dict:
    """把逐条检索明细汇总为整体/分类指标（供在线跑与中间结果复用两条路径共用）。"""
    def agg(rows_: list[dict]) -> dict:
        n = len(rows_)
        if not n:
            return {}
        return {
            "samples": n,
            "hit@1": round(sum(r["hit@1"] for r in rows_) / n, 4),
            "hit@3": round(sum(r["hit@3"] for r in rows_) / n, 4),
            "hit@k": round(sum(r["hit@k"] for r in rows_) / n, 4),
            "mrr": round(sum(r["rr"] for r in rows_) / n, 4),
            "term_coverage_topk": round(
                sum(r["term_coverage_topk"] for r in rows_) / n, 4),
            "term_coverage_gold": round(
                sum(r["term_coverage_gold"] for r in rows_) / n, 4),
        }

    by_cat = {cat: agg([r for r in details if r["category"] == cat])
              for cat in sorted({r["category"] for r in details})}

    misses = [
        {"id": r["id"], "category": r["category"], "query": r["query"],
         "expected_source": r["expected_source"], "top1_source": r["top1_source"]}
        for r in details if not r["hit@k"]
    ]
    gold_cov_misses = [
        {"id": r["id"], "category": r["category"], "query": r["query"],
         "term_coverage_gold": r["term_coverage_gold"]}
        for r in details if r["hit@k"] and r["term_coverage_gold"] < 1.0
    ]

    return {
        "k": k,
        "overall": agg(details),
        "by_category": by_cat,
        "misses": misses,
        "gold_coverage_gaps": gold_cov_misses,
        "details": details,
    }


class _CountingLLM:
    """极简 LLM 代理：统计真实调用次数与失败次数，其余属性透传。

    必要性：出题与判卷分别由 app/tools.py 直接调用 llm.invoke，
    若不代理则无法区分「模型真的参与了」还是「静默降级到规则引擎」。
    """

    def __init__(self, inner, box: dict):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_box", box)

    def invoke(self, *args, **kwargs):
        self._box["calls"] += 1
        try:
            out = self._inner.invoke(*args, **kwargs)
            self._box["ok"] += 1
            return out
        except Exception:
            self._box["failed"] += 1
            raise

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


class StrictAgentEvaluator:
    """严格评测：检索对抗集 / 出题事实核查 / 含简答题的判卷一致性。"""

    def __init__(self, backend: Optional[str] = None,
                 dataset: str = STRICT_DATASET,
                 call_interval: float = 0.6,
                 max_retry: int = 4,
                 report_stem: str = "eval_report_strict"):
        self.agent = SafetyTrainingAgent(backend)
        self.backend = self.agent.backend
        raw_llm = self.agent.llm
        self.llm_box = {"calls": 0, "ok": 0, "failed": 0}
        # 用计数代理替换 agent 侧的 LLM，使 tools.py 内部调用也可观测
        self.llm = _CountingLLM(raw_llm, self.llm_box)
        self.agent.llm = self.llm
        self.eval_dir = BASE_DIR / "eval"
        self.report_stem = report_stem
        self.dataset = json.loads(
            (BASE_DIR / dataset).read_text(encoding="utf-8")
        )
        self.call_interval = call_interval
        self.max_retry = max_retry
        self._calls = 0
        self._quota_exhausted = False      # 触发后所有 LLM 调用快速失败，避免空跑
        self.llm_errors: list[str] = []

    # ------------------------------------------------------------------
    # LLM 调用（带重试与节流；额度耗尽时快速失败）
    # ------------------------------------------------------------------
    @staticmethod
    def _is_quota_error(msg: str) -> bool:
        keys = ("free-models-per-day", "Rate limit exceeded", "insufficient_quota",
                "quota", "429")
        return any(k.lower() in msg.lower() for k in keys)

    def _invoke_llm(self, prompt: str) -> str:
        if self._quota_exhausted:
            raise RuntimeError("在线模型额度已耗尽（本轮已标记），跳过后续 LLM 调用")
        delay = 2.0
        last = None
        for attempt in range(self.max_retry):
            try:
                resp = self.llm.invoke(prompt)
                self._calls += 1
                time.sleep(self.call_interval)
                return resp.content if hasattr(resp, "content") else str(resp)
            except Exception as exc:  # noqa: BLE001
                last = exc
                msg = str(exc)
                if self._is_quota_error(msg):
                    # 免费额度按日重置，短退避重试毫无意义：立即熔断
                    self._quota_exhausted = True
                    note = f"额度耗尽(429): {msg[:220]}"
                    if note not in self.llm_errors:
                        self.llm_errors.append(note)
                    print("    [FATAL] 在线模型额度已耗尽（429），后续 LLM 调用将快速失败。")
                    raise RuntimeError(note) from exc
                wait = delay * (attempt + 1)
                print(f"    [WARN] LLM 调用失败（{type(exc).__name__}），{wait:.0f}s 后重试 "
                      f"({attempt + 1}/{self.max_retry})")
                time.sleep(wait)
        raise RuntimeError(f"LLM 连续 {self.max_retry} 次调用失败: {last!r}")

    # ==================================================================
    # 1. 检索层
    # ==================================================================
    def eval_retrieval(self, save_cb=None) -> dict:
        rows = self.dataset["retrieval"]
        k = settings.retrieve_k
        details = []
        t_start = time.time()

        for i, row in enumerate(rows):
            docs = self.agent.kb.search(row["query"], k=k)
            sources = [d.source for d in docs]
            cands = [c for c in str(row["expected_source"]).split("|") if c]

            rank = next((i + 1 for i, s in enumerate(sources)
                         if any(c in s for c in cands)), 0)
            is_gold = [any(c in s for c in cands) for s in sources]
            topk_text = "\n".join(d.content for d in docs)
            gold_text = "\n".join(d.content for d, g in zip(docs, is_gold) if g)

            terms = row["expected_terms"]
            cov_topk = sum(1 for t in terms if t in topk_text) / len(terms)
            cov_gold = sum(1 for t in terms if t in gold_text) / len(terms)

            el = time.time() - t_start
            eta = el / (i + 1) * (len(rows) - i - 1)
            print(f"    [{i + 1}/{len(rows)}] #{row['id']:<3} {row['category']:<12} "
                  f"rank={rank or '-':<3} top1={sources[0].split('/')[-1][:34] if sources else '-':<34}"
                  f" 已用{el:5.0f}s 预计剩余{eta:5.0f}s")

            details.append({
                "id": row["id"], "category": row["category"], "query": row["query"],
                "expected_source": row["expected_source"],
                "expected_terms": terms,
                "rank": rank,
                "hit@1": rank == 1,
                "hit@3": 1 <= rank <= 3,
                "hit@k": rank > 0,
                "rr": round(1.0 / rank, 4) if rank else 0.0,
                "term_coverage_topk": round(cov_topk, 4),
                "term_coverage_gold": round(cov_gold, 4),
                "gold_in_topk": sum(is_gold),
                "top1_source": sources[0] if sources else "",
                "top_sources": sources,
            })
            if save_cb is not None:
                save_cb(details)

        return aggregate_retrieval(details, k)

    def retrieval_from_partial(self) -> Optional[dict]:
        """从落盘的检索明细重建报告（避免长耗时检索重跑）。

        优先用本轮中间文件；否则回落到跨报告复用的明细缓存（检索层与后端无关，
        换后端/换报告名时无需重跑）。
        """
        cands = [self.eval_dir / ".strict_retrieval_partial.json",
                 self.eval_dir / "eval_retrieval_details.json"]
        details = None
        for p in cands:
            if not p.exists():
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if len(d) == len(self.dataset["retrieval"]):
                details = d
                print(f"  [INFO] 复用检索明细（{len(d)} 条，来自 {p.name}），跳过重跑。")
                break
            print(f"  [INFO] {p.name} 条数 {len(d)} ≠ 测试集 "
                  f"{len(self.dataset['retrieval'])}，忽略。")
        if details is None:
            return None
        return aggregate_retrieval(details, settings.retrieve_k)

    # ==================================================================
    # 2. 出题层
    # ==================================================================
    @staticmethod
    def _schema_checks(exam: Exam) -> dict:
        """确定性一致性校验（不依赖任何模型判断）。"""
        issues: list[str] = []
        contents: list[str] = []

        for q in exam.questions:
            tag = f"第{q.index}题"
            stem = (q.content or "").strip()
            ans = (q.answer or "").strip()
            contents.append(stem)

            if not stem:
                issues.append(f"{tag} 题干为空")
            if not (q.analysis or "").strip():
                issues.append(f"{tag} 缺少解析")

            letters = sorted(set(re.findall(r"[A-Za-z]", ans.upper())))
            n_options = len([o for o in (q.options or []) if str(o).strip()])

            if q.q_type in ("单选题", "多选题"):
                if n_options < 2:
                    issues.append(f"{tag} 选项不足 2 个")
                if not letters:
                    issues.append(f"{tag} 答案无选项字母")
                else:
                    max_letter = chr(64 + max(1, n_options))
                    if any(l > max_letter for l in letters):
                        issues.append(f"{tag} 答案 {ans} 超出选项范围（共{n_options}项）")
                if q.q_type == "单选题" and len(letters) != 1:
                    issues.append(f"{tag} 单选题答案应为 1 个字母，实为 {ans}")
                if q.q_type == "多选题" and len(letters) < 2:
                    issues.append(f"{tag} 多选题答案应≥2 个字母，实为 {ans}")
                opts = [str(o).strip() for o in (q.options or []) if str(o).strip()]
                if len(opts) != len(set(opts)):
                    issues.append(f"{tag} 选项存在重复")
            elif q.q_type == "判断题":
                if ans not in ("对", "错", "正确", "错误", "√", "×"):
                    issues.append(f"{tag} 判断题答案不规范：{ans}")
            elif q.q_type == "简答题":
                pts = [p for p in re.split(r"[。；;、\n]", ans) if len(p.strip()) >= 2]
                if len(pts) < 2:
                    issues.append(f"{tag} 简答题得分点不足 2 个")

        if len(contents) != len(set(contents)):
            issues.append("存在重复题干")

        dist = {}
        for q in exam.questions:
            dist[q.q_type] = dist.get(q.q_type, 0) + 1
        dist_ok = all(dist.get(t, 0) == c for t, c in REQUIRED_COUNTS.items())

        total_consistent = abs(sum(q.score for q in exam.questions) - exam.total_score) < 0.01

        return {
            "schema_ok": not issues,
            "issues": issues,
            "q_type_dist": dist,
            "dist_match": dist_ok,
            "total_score": exam.total_score,
            "total_consistent": total_consistent,
        }

    def _fact_check(self, exam: Exam, context: str) -> dict:
        """LLM-as-judge 事实核查。"""
        payload = {
            "title": exam.title,
            "craft_type": exam.craft_type,
            "edu_level": exam.edu_level,
            "questions": [
                {"index": q.index, "q_type": q.q_type, "content": q.content,
                 "options": q.options, "answer": q.answer, "score": q.score}
                for q in exam.questions
            ],
        }
        prompt = _JUDGE_PROMPT.format(
            exam=json.dumps(payload, ensure_ascii=False, indent=1),
            context=("\n命题所依据的检索资料（供判断依据充分性参考）：\n" + context[:4000]
                     if context else ""),
        )
        verdicts = {}
        err = None
        for _ in range(2):  # 解析失败再试一次
            try:
                raw = self._invoke_llm(prompt)
                arr = _parse_json_loose(raw)
                if isinstance(arr, dict):
                    arr = arr.get("results") or arr.get("data") or [arr]
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    idx = int(item.get("index", 0) or 0)
                    verdicts[idx] = {
                        "stem_ok": bool(item.get("stem_ok")),
                        "answer_ok": bool(item.get("answer_ok")),
                        "grounded_ok": bool(item.get("grounded_ok")),
                        "reason": str(item.get("reason", ""))[:60],
                    }
                err = None
                break
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
        return {"verdicts": verdicts, "error": err}

    def _eval_exam_once(self, row: dict, fact_check: bool) -> dict:
        craft, level = row["craft_type"], row["edu_level"]

        buf = io.StringIO()
        t0 = time.time()
        with redirect_stdout(buf):
            exam = self.agent.get_exam(craft, level)
        elapsed = time.time() - t0
        online = (self.backend in ("ollama",)
                  and "[WARN] 在线出题失败" not in buf.getvalue())

        text = json.dumps(exam.model_dump(), ensure_ascii=False)
        structure_ok = (
            len(exam.questions) > 0
            and all((q.answer or "").strip() for q in exam.questions)
            and all(q.options for q in exam.questions
                    if q.q_type in ("单选题", "多选题"))
            and exam.craft_type == craft
            and exam.edu_level == level
        )
        facts = row["expected_facts"]
        fact_cov = sum(1 for f in facts if f in text) / len(facts)
        must = row["must_contain"]
        must_cov = sum(1 for f in must if f in text) / len(must)

        schema = self._schema_checks(exam)

        out = {
            "craft_type": craft, "edu_level": level,
            "online": online, "elapsed_s": round(elapsed, 1),
            "structure_ok": structure_ok,
            "must_contain_coverage": round(must_cov, 4),
            "fact_coverage": round(fact_cov, 4),
            "question_count": len(exam.questions),
            **schema,
        }

        if fact_check:
            fc = self._fact_check(exam, "")
            vd = fc["verdicts"]
            judged = [vd.get(q.index) for q in exam.questions]
            judged = [j for j in judged if j]
            if not judged:
                # 裁判不可用（额度耗尽 / 解析失败）：如实标记为「未判定」，
                # 绝不用 0% 冒充「事实全部不合格」。
                out["judge"] = {
                    "judged_questions": 0,
                    "total_questions": len(exam.questions),
                    "unavailable": True,
                    "error": fc["error"],
                    "failed_examples": [],
                }
                return out
            n = len(judged)
            out["judge"] = {
                "judged_questions": len(judged),
                "total_questions": len(exam.questions),
                "unavailable": False,
                "stem_valid_rate": round(sum(j["stem_ok"] for j in judged) / n, 4),
                "answer_valid_rate": round(sum(j["answer_ok"] for j in judged) / n, 4),
                "grounded_rate": round(sum(j["grounded_ok"] for j in judged) / n, 4),
                "fact_ok_rate": round(
                    sum(1 for j in judged
                        if j["stem_ok"] and j["answer_ok"] and j["grounded_ok"]) / n, 4),
                "error": fc["error"],
                "verdicts": {str(k): v for k, v in vd.items()},
            }
            # 保留失败样例，便于人工复核
            out["judge"]["failed_examples"] = [
                {"index": q.index, "content": q.content[:60],
                 "answer": (q.answer or "")[:40], "reason": vd.get(q.index, {}).get("reason", "")}
                for q in exam.questions
                if q.index in vd and not (vd[q.index]["stem_ok"]
                                          and vd[q.index]["answer_ok"]
                                          and vd[q.index]["grounded_ok"])
            ][:4]
        return out

    def eval_exam(self, runs: int = 3, fact_check: bool = True) -> dict:
        rows = self.dataset["exam"]
        if fact_check and self.backend not in ("ollama",):
            print("  [INFO] 当前后端非在线模型，LLM 事实核查不可用，已自动关闭。")
            fact_check = False
        per_run: list[dict] = []

        for run in range(1, runs + 1):
            print(f"  [出题] 第 {run}/{runs} 轮（{len(rows)} 个工种）...")
            run_rows = []
            for row in rows:
                r = self._eval_exam_once(row, fact_check)
                r["run"] = run
                run_rows.append(r)
                flag = "在线" if r["online"] else "离线兜底"
                extra = ""
                if fact_check and r.get("judge"):
                    if r["judge"].get("unavailable"):
                        extra = " 事实核查=不可用"
                    else:
                        extra = f" 事实合格率={r['judge']['fact_ok_rate']:.0%}"
                print(f"    {row['craft_type']}/{row['edu_level']} [{flag}] "
                      f"题量={r['question_count']} 结构={'OK' if r['structure_ok'] else 'NG'} "
                      f"一致性={'OK' if r['schema_ok'] else 'NG'} "
                      f"知识点覆盖={r['fact_coverage']:.0%}{extra} ({r['elapsed_s']}s)",
                      flush=True)
                per_run.append(r)

        def agg(rows_: list[dict]) -> dict:
            n = len(rows_)
            if not n:
                return {}
            d = {
                "samples": n,
                "online_rate": round(sum(r["online"] for r in rows_) / n, 4),
                "structure_valid_rate": round(sum(r["structure_ok"] for r in rows_) / n, 4),
                "schema_rate": round(sum(r["schema_ok"] for r in rows_) / n, 4),
                "dist_match_rate": round(sum(r["dist_match"] for r in rows_) / n, 4),
                "must_contain_coverage": round(
                    statistics.fmean([r["must_contain_coverage"] for r in rows_]), 4),
                "fact_coverage": round(
                    statistics.fmean([r["fact_coverage"] for r in rows_]), 4),
                "avg_questions": round(
                    statistics.fmean([r["question_count"] for r in rows_]), 2),
                "avg_elapsed_s": round(
                    statistics.fmean([r["elapsed_s"] for r in rows_]), 1),
            }
            if fact_check:
                allj = [r["judge"] for r in rows_ if r.get("judge")]
                js = [j for j in allj if not j.get("unavailable")]
                d["judge_coverage"] = round(len(js) / len(allj), 4) if allj else 0.0
                if js:
                    d.update({
                        "judge_total_questions": sum(j["total_questions"] for j in js),
                        "judge_judged": sum(j["judged_questions"] for j in js),
                        "stem_valid_rate": round(
                            statistics.fmean([j["stem_valid_rate"] for j in js]), 4),
                        "answer_valid_rate": round(
                            statistics.fmean([j["answer_valid_rate"] for j in js]), 4),
                        "grounded_rate": round(
                            statistics.fmean([j["grounded_rate"] for j in js]), 4),
                        "fact_ok_rate": round(
                            statistics.fmean([j["fact_ok_rate"] for j in js]), 4),
                    })
                else:
                    d["fact_check_unavailable"] = True
            return d

        run_summaries = [agg([r for r in per_run if r["run"] == i])
                         for i in range(1, runs + 1)]

        stability_keys = ["online_rate", "structure_valid_rate", "schema_rate",
                          "dist_match_rate", "fact_coverage"]
        if fact_check:
            stability_keys += ["stem_valid_rate", "answer_valid_rate",
                               "grounded_rate", "fact_ok_rate"]
        stability = {k: _mean_std([s[k] for s in run_summaries if k in s])
                     for k in stability_keys}

        return {
            "runs": runs,
            "fact_check": fact_check,
            "overall": agg(per_run),
            "per_run": run_summaries,
            "stability": stability,
            "details": per_run,
        }

    # ==================================================================
    # 3. 判卷层
    # ==================================================================
    def _build_standard_exam(self) -> Exam:
        g = self.dataset["grading"]["exam"]
        qs = [
            Question(index=q["index"], q_type=q["q_type"], content=q["content"],
                     options=q.get("options", []) or [], answer=q["answer"],
                     score=float(q["score"]))
            for q in g["questions"]
        ]
        return Exam(title=g["title"], craft_type=g["craft_type"],
                    edu_level=g["edu_level"], questions=qs,
                    total_score=float(sum(q.score for q in qs)))

    def _eval_grading_once(self) -> dict:
        exam = self._build_standard_exam()
        qmap = {q.index: q for q in exam.questions}
        obj_idx = [q.index for q in exam.questions
                   if q.q_type in ("单选题", "多选题", "判断题")]
        essay_idx = [q.index for q in exam.questions if q.q_type == "简答题"]
        cases = self.dataset["grading"]["cases"]

        per_case = []
        for c in cases:
            answers = []
            for qi, q in qmap.items():
                a = c["objective"].get(str(qi), "") if q.q_type != "简答题" \
                    else c["essay"].get(str(qi), "")
                answers.append(AnsweredQuestion(index=qi, q_type=q.q_type,
                                                student_answer=a))
            paper = SubmittedPaper(craft_type=exam.craft_type,
                                   edu_level=exam.edu_level,
                                   trainee_name=c["name"], answers=answers)

            buf = io.StringIO()
            with redirect_stdout(buf):
                res = self.agent.grade(exam, paper)
            got = {d.index: d.score for d in res.details}
            human = {int(k): float(v) for k, v in c["human"].items()}

            obj_ok = sum(1 for i in obj_idx if abs(got.get(i, 0) - human[i]) < 0.01)
            essay_diff = [abs(got.get(i, 0) - human[i]) for i in essay_idx]

            # 判定每条简答究竟由「LLM 评分」还是「规则兜底」产生：
            # _subjective_offline 的评语固定为「命中 x/y 个得分点。」
            def _scorer(i: int) -> str:
                cmt = next((d.comment for d in res.details if d.index == i), "")
                return "offline_rule" if re.match(r"^命中 \d+/\d+ 个得分点", cmt.strip()) \
                    else "llm"

            scorers = {str(i): _scorer(i) for i in essay_idx}
            per_case.append({
                "case": c["name"],
                "human_total": sum(human.values()),
                "actual_total": res.total_score,
                "total_diff": round(res.total_score - sum(human.values()), 1),
                "objective_correct": obj_ok,
                "objective_total": len(obj_idx),
                "essay_diff": [round(d, 1) for d in essay_diff],
                "essay_got": [got.get(i, 0) for i in essay_idx],
                "essay_human": [human[i] for i in essay_idx],
                "essay_scorers": scorers,
                "per_question": {str(i): {"got": got.get(i, 0), "human": human[i],
                                          "comment": next((d.comment for d in res.details
                                                           if d.index == i), "")[:50]}
                                 for i in qmap},
            })

        n = len(per_case)
        obj_correct = sum(c["objective_correct"] for c in per_case)
        obj_total = sum(c["objective_total"] for c in per_case)
        essay_pairs = [(g, h) for c in per_case
                       for g, h in zip(c["essay_got"], c["essay_human"])]
        total_pairs = [(c["actual_total"], c["human_total"]) for c in per_case]

        essay_err = [g - h for g, h in essay_pairs]
        total_err = [g - h for g, h in total_pairs]

        all_scorers = [s for c in per_case for s in c["essay_scorers"].values()]
        llm_share = (round(sum(1 for s in all_scorers if s == "llm") / len(all_scorers), 4)
                     if all_scorers else None)
        # 仅统计「确实由 LLM 评分」的简答样本，避免规则兜底混入 LLM 质量结论
        llm_pairs = [(g, h) for c in per_case
                     for g, h, s in zip(c["essay_got"], c["essay_human"],
                                        c["essay_scorers"].values()) if s == "llm"]

        return {
            "cases": n,
            "essay_llm_share": llm_share,
            "essay_llm_samples": len(llm_pairs),
            "objective_accuracy": round(obj_correct / obj_total, 4) if obj_total else None,
            "objective_correct": obj_correct,
            "objective_total": obj_total,
            "essay_mae": round(sum(abs(e) for e in essay_err) / len(essay_err), 2),
            "essay_rmse": round(math.sqrt(sum(e * e for e in essay_err) / len(essay_err)), 2),
            "essay_bias": round(sum(essay_err) / len(essay_err), 2),
            "essay_within_10pct": round(
                sum(1 for g, h in essay_pairs
                    if abs(g - h) <= 0.1 * (h if h else 30)) / len(essay_pairs), 4),
            "total_mae": round(sum(abs(e) for e in total_err) / len(total_err), 2),
            "total_bias": round(sum(total_err) / len(total_err), 2),
            "total_within_5pts": round(
                sum(1 for e in total_err if abs(e) <= 5) / len(total_err), 4),
            "pearson_total": _pearson([g for g, _ in total_pairs], [h for _, h in total_pairs]),
            "spearman_total": _spearman([g for g, _ in total_pairs], [h for _, h in total_pairs]),
            "pearson_essay": _pearson([g for g, _ in essay_pairs], [h for _, h in essay_pairs]),
            "spearman_essay": _spearman([g for g, _ in essay_pairs], [h for _, h in essay_pairs]),
            "llm_pearson_essay": (_pearson([g for g, _ in llm_pairs], [h for _, h in llm_pairs])
                                  if len(llm_pairs) >= 3 else None),
            "llm_spearman_essay": (_spearman([g for g, _ in llm_pairs], [h for _, h in llm_pairs])
                                   if len(llm_pairs) >= 3 else None),
            "llm_essay_mae": (round(sum(abs(g - h) for g, h in llm_pairs) / len(llm_pairs), 2)
                              if llm_pairs else None),
            "per_case": per_case,
        }

    def eval_grading(self, runs: int = 3) -> dict:
        per_run = []
        for run in range(1, runs + 1):
            print(f"  [判卷] 第 {run}/{runs} 轮（{len(self.dataset['grading']['cases'])} 条用例）...",
                  flush=True)
            r = self._eval_grading_once()
            r["run"] = run
            per_run.append(r)
            print(f"    客观题准确率={r['objective_accuracy']:.1%} "
                  f"主观题MAE={r['essay_mae']} 总分MAE={r['total_mae']} "
                  f"Pearson={r['pearson_total']} Spearman={r['spearman_total']} "
                  f"LLM评分占比={r['essay_llm_share']:.0%}", flush=True)

        keys = ["objective_accuracy", "essay_mae", "essay_rmse", "essay_bias",
                "essay_within_10pct", "total_mae", "total_bias", "total_within_5pts",
                "pearson_total", "spearman_total", "pearson_essay", "spearman_essay",
                "essay_llm_share", "essay_llm_samples", "llm_essay_mae",
                "llm_pearson_essay", "llm_spearman_essay"]

        def agg(run_: dict) -> dict:
            return {k: run_[k] for k in keys if run_.get(k) is not None}

        return {
            "runs": runs,
            "overall": agg(per_run[0]),
            "per_run": [agg(r) for r in per_run],
            "stability": {k: _mean_std([r[k] for r in per_run if r.get(k) is not None])
                          for k in keys},
            "details": per_run,
        }

    # ==================================================================
    # 汇总
    # ==================================================================
    def _load_existing(self) -> dict:
        p = self.eval_dir / f"{self.report_stem}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _write(self, report: dict) -> Path:
        out = self.eval_dir / f"{self.report_stem}.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return out

    def _model_name(self) -> str:
        if self.backend == "ollama":
            return settings.ollama_model
        return "内置规则引擎 RuleBasedChatLLM（零网络）"

    def run_all(self, runs_exam: int = 3, runs_grading: int = 3,
                fact_check: bool = True, skip_exam: bool = False,
                skip_grading: bool = False, stage: str = "all") -> dict:
        """stage: all | retrieval | exam | grading —— 非 all 时只跑该阶段并与已有报告合并落盘。"""
        if stage != "all":
            skip_exam = stage != "exam"
            skip_grading = stage != "grading"
            if stage == "retrieval":
                skip_exam = skip_grading = True

        t0 = time.time()
        print(f"\n========== 严格评测开始（后端：{self.backend}，阶段：{stage}）==========")
        prev = self._load_existing() if stage != "all" else {}
        partial = self.eval_dir / ".strict_retrieval_partial.json"

        def _save_partial(details):
            partial.write_text(json.dumps(details, ensure_ascii=False),
                               encoding="utf-8")

        if stage in ("all", "retrieval"):
            cached = self.retrieval_from_partial()
            if cached is not None:
                retrieval = cached
            else:
                print(f"[检索层] {len(self.dataset['retrieval'])} 条对抗 query ...")
                retrieval = self.eval_retrieval(save_cb=_save_partial)
                # 跨报告复用缓存（检索层与 LLM 后端无关）
                (self.eval_dir / "eval_retrieval_details.json").write_text(
                    json.dumps(retrieval["details"], ensure_ascii=False),
                    encoding="utf-8")
            o = retrieval["overall"]
            print(f"  Hit@1={o['hit@1']:.1%} Hit@3={o['hit@3']:.1%} "
                  f"Hit@{retrieval['k']}={o['hit@k']:.1%} MRR={o['mrr']:.3f} "
                  f"术语覆盖(top-k)={o['term_coverage_topk']:.1%} "
                  f"术语覆盖(gold)={o['term_coverage_gold']:.1%}")
            retrieval["_backend"] = self.backend   # 检索层实际与后端无关，仅记录
        else:
            retrieval = prev.get("retrieval", {"skipped": True})

        exam = prev.get("exam", {"skipped": True})
        if not skip_exam:
            print(f"[出题层] {len(self.dataset['exam'])} 工种 × {runs_exam} 轮"
                  f"{'，含 LLM 事实核查' if fact_check else ''} ...")
            exam = self.eval_exam(runs=runs_exam, fact_check=fact_check)
            exam["_backend"] = self.backend
            exam["_llm_model"] = self._model_name()
            exam["_llm_calls"] = dict(self.llm_box)
            e = exam["overall"]
            print(f"  在线率={e['online_rate']:.1%} 结构合规率={e['structure_valid_rate']:.1%} "
                  f"一致性校验={e['schema_rate']:.1%} 知识点覆盖={e['fact_coverage']:.1%}")
            if exam.get("fact_check") and e.get("fact_ok_rate") is not None:
                print(f"  事实合格率={e['fact_ok_rate']:.1%} "
                      f"（题干{e['stem_valid_rate']:.1%} / 答案{e['answer_valid_rate']:.1%} / "
                      f"依据{e['grounded_rate']:.1%}；覆盖率={e.get('judge_coverage', 0):.0%}）")
            elif exam.get("fact_check"):
                print("  [WARN] 事实核查不可用（LLM 额度耗尽或解析失败），"
                      "该指标已从综合分中剔除，未以 0% 冒充。")
            if e.get("online_rate", 1) == 0:
                print("  [WARN] 在线出题成功率 0%：本层结果全部来自离线兜底引擎，"
                      "不能代表线上模型质量。")

        grading = prev.get("grading", {"skipped": True})
        if not skip_grading:
            print(f"[判卷层] 3 客观题 + 2 简答题，"
                  f"{len(self.dataset['grading']['cases'])} 条含人工期望分数的用例 × "
                  f"{runs_grading} 轮 ...")
            grading = self.eval_grading(runs=runs_grading)
            grading["_backend"] = self.backend
            grading["_llm_model"] = self._model_name()
            grading["_llm_calls"] = dict(self.llm_box)

        # ---- 按阶段归集后端/模型/调用次数，避免「最后一次运行」覆盖整份报告 ----
        prev_meta = prev.get("meta", {}) if isinstance(prev, dict) else {}
        stage_meta = dict(prev_meta.get("stage_meta", {}))
        for name, blk in (("retrieval", retrieval), ("exam", exam), ("grading", grading)):
            if blk.get("_backend"):
                stage_meta[name] = {
                    "backend": blk.pop("_backend"),
                    "llm_model": blk.pop("_llm_model", None),
                    "llm_calls": blk.pop("_llm_calls", None),
                }
            elif name in stage_meta:
                continue
        llm_stages = {k: v for k, v in stage_meta.items()
                      if k in ("exam", "grading") and v.get("backend")}
        backends = sorted({v["backend"] for v in llm_stages.values()})
        models = sorted({v["llm_model"] for v in llm_stages.values() if v.get("llm_model")})
        sum_ok = sum((v.get("llm_calls") or {}).get("ok", 0) for v in llm_stages.values())
        sum_all = sum((v.get("llm_calls") or {}).get("calls", 0) for v in llm_stages.values())
        sum_fail = sum((v.get("llm_calls") or {}).get("failed", 0) for v in llm_stages.values())

        effective_fact_check = (exam.get("fact_check", fact_check)
                                if not exam.get("skipped") else fact_check)
        # 未在本轮运行的层，其元信息沿用在先报告中的真实值，避免被 CLI 默认值覆盖
        if skip_exam and not exam.get("skipped"):
            effective_fact_check = prev_meta.get("fact_check", effective_fact_check)
            runs_exam = prev_meta.get("runs_exam", runs_exam)
        if skip_grading and not grading.get("skipped"):
            runs_grading = prev_meta.get("runs_grading", runs_grading)

        components: list[tuple[str, float, float]] = []
        if not retrieval.get("skipped"):
            ro = retrieval["overall"]
            components += [("hit@k", ro["hit@k"], 0.30), ("mrr", ro["mrr"], 0.10)]
        if not exam.get("skipped"):
            eo = exam["overall"]
            components.append(("schema_rate", eo["schema_rate"], 0.15))
            if eo.get("fact_ok_rate") is not None and not eo.get("fact_check_unavailable"):
                components.append(("fact_ok_rate", eo["fact_ok_rate"], 0.20))
        if not grading.get("skipped"):
            go = grading["overall"]
            components.append(("objective_accuracy", go["objective_accuracy"], 0.10))
            if go.get("essay_within_10pct") is not None:
                components.append(("essay_within_10pct", go["essay_within_10pct"], 0.15))

        wsum = sum(w for _, _, w in components)
        strict_overall = (round(sum(v * w for _, v, w in components) / wsum, 4)
                          if wsum > 0 else None)

        report = {
            "meta": {
                "evaluator": "StrictAgentEvaluator v1.0",
                "backend": (backends[0] if len(backends) == 1
                            else ("mixed:" + "+".join(backends) if backends else self.backend)),
                "backend_by_stage": {k: v.get("backend") for k, v in stage_meta.items()},
                "stage_meta": stage_meta,
                "llm_model": (models[0] if len(models) == 1
                              else ("mixed" if models else self._model_name())),
                "llm_models_used": models,
                "dataset": STRICT_DATASET,
                "retrieval_samples": len(self.dataset["retrieval"]),
                "exam_crafts": len(self.dataset["exam"]),
                "grading_cases": len(self.dataset["grading"]["cases"]),
                "runs_exam": runs_exam,
                "runs_grading": runs_grading,
                "fact_check": effective_fact_check,
                "stages_done": [s for s, present in
                                (("retrieval", not retrieval.get("skipped")),
                                 ("exam", not exam.get("skipped")),
                                 ("grading", not grading.get("skipped"))) if present],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "llm_calls": self._calls,
                "llm_calls_total": sum_all,
                "llm_calls_ok": sum_ok,
                "llm_calls_failed": sum_fail,
                "llm_errors": self.llm_errors,
                "quota_exhausted": self._quota_exhausted,
                "wall_clock_s": round(time.time() - t0, 1),
            },
            "retrieval": retrieval,
            "exam": exam,
            "grading": grading,
            "strict_overall_score": strict_overall,
            "strict_overall_components": [
                {"metric": name, "value": val, "weight": w} for name, val, w in components
            ],
            "strict_overall_formula": (
                "hit@k*0.30 + mrr*0.10 + schema_rate*0.15 + fact_ok_rate*0.20 "
                "+ objective_accuracy*0.10 + essay_within_10pct*0.15"
                "（缺失子项按剩余权重归一化，fact_ok_rate 不可用时自动剔除）"
            ),
        }

        out = self._write(report)
        if partial.exists():
            partial.unlink()
        print(f"\n========== 阶段完成（{stage}）==========")
        print(f"LLM 调用次数 : {self._calls}")
        print(f"本阶段耗时   : {round(time.time() - t0, 1)}s")
        if strict_overall is not None:
            print(f"严格综合分   : {strict_overall:.1%}")
        print(f"报告已保存   : {out}")
        return report
