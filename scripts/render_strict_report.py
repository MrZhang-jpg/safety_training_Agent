# -*- coding: utf-8 -*-
"""把 eval/eval_report_strict*.json 渲染为可读的 Markdown 报告（并对比原始基线报告）。

用法：
    python scripts/render_strict_report.py                          # eval_report_strict
    python scripts/render_strict_report.py eval_report_strict_ollama
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"


def pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def num(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def ms(d):
    if not d or d.get("mean") is None:
        return "—"
    return f"{d['mean']:.3f} ± {d['std']:.3f}（{d['min']:.2f}~{d['max']:.2f}）"


def pctms(d):
    if not d or d.get("mean") is None:
        return "—"
    return (f"{d['mean'] * 100:.1f}% ± {d['std'] * 100:.1f}%"
            f"（{d['min'] * 100:.1f}~{d['max'] * 100:.1f}）")


def main():
    stem = sys.argv[1] if len(sys.argv) > 1 else "eval_report_strict"
    rep = json.loads((EVAL / f"{stem}.json").read_text(encoding="utf-8"))
    base = None
    bp = EVAL / "eval_report.json"
    if bp.exists():
        try:
            base = json.loads(bp.read_text(encoding="utf-8"))
        except Exception:
            base = None

    m = rep["meta"]
    L = []
    A = L.append

    A("# 安全培训智能 Agent —— 严格评测报告\n")
    A(f"- 评测时间：{m['timestamp']}")
    A(f"- 后端 / 模型：`{m['backend']}` / `{m['llm_model']}`")
    A(f"- 测试集：`{m['dataset']}`（检索 {m['retrieval_samples']} 条、"
      f"出题 {m['exam_crafts']} 工种、判卷 {m['grading_cases']} 条用例）")
    A(f"- 重复轮数：出题 {m['runs_exam']} 轮 / 判卷 {m['runs_grading']} 轮"
      f"（随机性指标报告均值 ± 标准差）")
    A(f"- LLM 事实核查：{'开启' if m['fact_check'] else '关闭'}")
    A(f"- LLM 调用：共 {m.get('llm_calls_total', m['llm_calls'])} 次"
      f"（成功 {m.get('llm_calls_ok', '—')} / 失败 {m.get('llm_calls_failed', '—')}），"
      f"总耗时 {m['wall_clock_s']} 秒")
    if rep.get("strict_overall_score") is not None:
        A(f"- **严格综合分：{pct(rep['strict_overall_score'])}**")
        comps = rep.get("strict_overall_components") or []
        if comps:
            A("  - 计入子项：" + "、".join(
                f"{c['metric']}={pct(c['value'])}×{c['weight']}" for c in comps))
    A("")
    if m.get("llm_errors"):
        A("> ⚠️ **本轮存在 LLM 调用故障，相关指标已按「不可用」处理，未用 0% 冒充：**")
        for err in m["llm_errors"]:
            A(f"> - `{err[:300]}`")
        A("")

    # ---------------- 一、检索层 ----------------
    A("## 一、检索层（80 条对抗 query）\n")
    if rep["retrieval"].get("skipped"):
        A("_本轮未运行检索层。_\n")
        r = {"k": 6, "by_category": {}, "misses": [], "gold_coverage_gaps": []}
    else:
        r = rep["retrieval"]
        o = r["overall"]
        A("| 指标 | 严格评测 | 原始基线（12 条 query） |")
        A("|---|---|---|")
        bh = base["retrieval"] if base else {}
        A(f"| 样本数 | {o['samples']} | {bh.get('samples', '—')} |")
        A(f"| Hit@1 | {pct(o['hit@1'])} | 未统计 |")
        A(f"| Hit@3 | {pct(o['hit@3'])} | 未统计 |")
        A(f"| Hit@{r['k']} | {pct(o['hit@k'])} | {pct(bh.get('hit_rate'))} |")
        A(f"| MRR | {num(o['mrr'])} | {num(bh.get('mrr'))} |")
        A(f"| 关键词覆盖（top-k 口径，宽） | {pct(o['term_coverage_topk'])} | "
          f"{pct(bh.get('term_coverage'))} |")
        A(f"| 关键词覆盖（命中来源口径，严） | {pct(o['term_coverage_gold'])} | 未统计 |")
        A("")
        A("### 按 query 类别拆分（暴露「开卷式满分」背后的短板）\n")
        A("| 类别 | 条数 | Hit@1 | Hit@3 | Hit@k | MRR | 术语覆盖(严) |")
        A("|---|---|---|---|---|---|---|")
        label = {"literal": "直述式（含工种词，基线同类）", "colloquial": "口语化/现场土话",
                 "typo": "错别字/变体", "cross_craft": "跨工种易混淆",
                 "scenario": "场景症状式（不给工种名）", "numeric_spec": "数值条款式",
                 "policy": "制度流程式"}
        for cat, v in r["by_category"].items():
            A(f"| {label.get(cat, cat)} | {v['samples']} | {pct(v['hit@1'])} | "
              f"{pct(v['hit@3'])} | {pct(v['hit@k'])} | {num(v['mrr'])} | "
              f"{pct(v['term_coverage_gold'])} |")
        A("")

    if r.get("misses"):
        A(f"### 完全未命中的 query（{len(r['misses'])} 条）\n")
        A("| id | 类别 | query | 期望来源 | 实际 Top1 |")
        A("|---|---|---|---|---|")
        for x in r["misses"]:
            A(f"| {x['id']} | {x['category']} | {x['query']} | "
              f"`{x['expected_source']}` | `{x['top1_source']}` |")
        A("")

    if r.get("gold_coverage_gaps"):
        A(f"### 命中来源但术语覆盖不全的 query（{len(r['gold_coverage_gaps'])} 条）\n")
        A("| id | 类别 | query | 术语覆盖(严) |")
        A("|---|---|---|---|")
        for x in r["gold_coverage_gaps"]:
            A(f"| {x['id']} | {x['category']} | {x['query']} | "
              f"{pct(x['term_coverage_gold'])} |")
        A("")

    # ---------------- 二、出题层 ----------------
    A("## 二、出题层（8 个工种）\n")
    e = rep.get("exam", {})
    if e.get("skipped"):
        A("_本轮未运行出题层。_\n")
    else:
        eo = e["overall"]
        be = base["exam"] if base else {}
        A("| 指标 | 严格评测 | 原始基线（5 工种 / 单轮） |")
        A("|---|---|---|")
        A(f"| 在线 LLM 出题成功率 | {pct(eo['online_rate'])} | 100%（人工确认） |")
        A(f"| 结构合规率 | {pct(eo['structure_valid_rate'])} | {pct(be.get('valid_rate'))} |")
        A(f"| 确定性一致性校验通过率 | {pct(eo['schema_rate'])} | 未统计 |")
        A(f"| 题型分布符合要求率 | {pct(eo['dist_match_rate'])} | 未统计 |")
        A(f"| 工种/层级关键词覆盖 | {pct(eo['must_contain_coverage'])} | "
          f"{pct(be.get('term_coverage'))} |")
        A(f"| 知识点覆盖率 | {pct(eo['fact_coverage'])} | 未统计 |")
        if eo.get("fact_check_unavailable"):
            A("| **事实合格率（LLM 核查三项全过）** | 不可用（裁判模型额度耗尽/解析失败） | 未统计 |")
        elif "fact_ok_rate" in eo:
            A(f"| **事实合格率（LLM 核查三项全过）** | {pct(eo['fact_ok_rate'])} | 未统计 |")
            A(f"| ├ 题干有效性 | {pct(eo['stem_valid_rate'])} | 未统计 |")
            A(f"| ├ 答案正确性 | {pct(eo['answer_valid_rate'])} | 未统计 |")
            A(f"| ├ 依据充分性 | {pct(eo['grounded_rate'])} | 未统计 |")
            A(f"| └ 裁判覆盖率 | {pct(eo.get('judge_coverage'))} | — |")
        A(f"| 平均题量 | {eo['avg_questions']} | 13（基线记录） |")
        A(f"| 平均单工种耗时 | {eo['avg_elapsed_s']} 秒 | — |")
        A("")

        A("### 多轮稳定性（均值 ± 标准差）\n")
        A("| 指标 | 表现 |")
        A("|---|---|")
        st = e["stability"]
        A(f"| 在线率 | {pctms(st.get('online_rate'))} |")
        A(f"| 结构合规率 | {pctms(st.get('structure_valid_rate'))} |")
        A(f"| 一致性校验通过率 | {pctms(st.get('schema_rate'))} |")
        A(f"| 知识点覆盖率 | {pctms(st.get('fact_coverage'))} |")
        if "fact_ok_rate" in st:
            A(f"| 事实合格率 | {pctms(st.get('fact_ok_rate'))} |")
        A("")

        A("### 逐工种明细（各轮）\n")
        A("| 轮 | 工种/层级 | 在线 | 题量 | 结构 | 一致性 | 分布 | 知识点覆盖 | 事实合格率 |")
        A("|---|---|---|---|---|---|---|---|---|")
        for d in e["details"]:
            j = d.get("judge")
            if not j:
                fk = "—"
            elif j.get("unavailable"):
                fk = "不可用"
            else:
                fk = pct(j["fact_ok_rate"])
            A(f"| {d['run']} | {d['craft_type']}/{d['edu_level']} | "
              f"{'✅' if d['online'] else '❌'} | {d['question_count']} | "
              f"{'✅' if d['structure_ok'] else '❌'} | "
              f"{'✅' if d['schema_ok'] else '❌'} | "
              f"{'✅' if d['dist_match'] else '❌'} | {pct(d['fact_coverage'])} | {fk} |")
        A("")

        issues = [(d["run"], d["craft_type"], i) for d in e["details"] for i in d["issues"]]
        if issues:
            A(f"### 一致性校验发现的问题（{len(issues)} 条，去重展示）\n")
            seen = set()
            for run, craft, it in issues:
                if it in seen:
                    continue
                seen.add(it)
                A(f"- [{craft}] {it}")
            A("")

        fails = [(d["craft_type"], fx) for d in e["details"]
                 for fx in (d.get("judge", {}) or {}).get("failed_examples", [])]
        if fails:
            A("### 事实核查判为不合格的题目样例（最多 12 条）\n")
            A("| 工种 | 题号 | 题干 | 参考答案 | 判为不合格理由 |")
            A("|---|---|---|---|---|")
            for craft, fx in fails[:12]:
                A(f"| {craft} | {fx['index']} | {fx['content']} | {fx['answer']} | "
                  f"{fx['reason']} |")
            A("")

    # ---------------- 三、判卷层 ----------------
    A("## 三、判卷层（3 客观题 + 2 简答题 × 14 条用例）\n")
    g = rep.get("grading", {})
    if g.get("skipped"):
        A("_本轮未运行判卷层。_\n")
    else:
        go = g["overall"]
        bg = base["grading"] if base else {}
        A("| 指标 | 严格评测 | 原始基线（5 条纯客观题用例） |")
        A("|---|---|---|")
        A(f"| 用例数 | {m['grading_cases']} | {bg.get('samples', '—')} |")
        A(f"| 客观题判分准确率 | {pct(go['objective_accuracy'])} | "
          f"{pct(bg.get('score_accuracy'))} |")
        if go.get("essay_llm_share") is not None:
            A(f"| 简答由 LLM 实际评分的比例 | {pct(go['essay_llm_share'])}"
              f"（{go.get('essay_llm_samples', 0)} 个样本；其余为规则兜底） | 未覆盖 |")
        A(f"| 主观题（简答）MAE | {go['essay_mae']} 分 | 未覆盖 |")
        A(f"| 主观题 RMSE | {go['essay_rmse']} 分 | 未覆盖 |")
        A(f"| 主观题偏差（机器-人工） | {go['essay_bias']:+.2f} 分 | 未覆盖 |")
        A(f"| 主观题落在人工分 ±10% 内 | {pct(go['essay_within_10pct'])} | 未覆盖 |")
        A(f"| 总分 MAE | {go['total_mae']} 分 | 未覆盖 |")
        A(f"| 总分偏差 | {go['total_bias']:+.2f} 分 | 未覆盖 |")
        A(f"| 总分误差 ≤5 分 | {pct(go['total_within_5pts'])} | 未覆盖 |")
        A(f"| **Pearson r（总分 vs 人工）** | {num(go['pearson_total'])} | 未统计 |")
        A(f"| **Spearman ρ（总分 vs 人工）** | {num(go['spearman_total'])} | 未统计 |")
        A(f"| Pearson r（单题简答） | {num(go['pearson_essay'])} | 未统计 |")
        A(f"| Spearman ρ（单题简答） | {num(go['spearman_essay'])} | 未统计 |")
        if go.get("llm_essay_mae") is not None:
            A(f"| 仅计 LLM 评分样本的简答 MAE | {go['llm_essay_mae']} 分 | 未覆盖 |")
            A(f"| 仅计 LLM 评分样本的 Pearson r | {num(go.get('llm_pearson_essay'))} | 未统计 |")
        A("")

        A("### 多轮稳定性\n")
        A("| 指标 | 均值 ± 标准差 |")
        A("|---|---|")
        rate_keys = {"objective_accuracy", "essay_within_10pct", "total_within_5pts",
                     "essay_llm_share"}
        for k, lbl in [("objective_accuracy", "客观题准确率"),
                       ("essay_llm_share", "LLM 实际评分占比"),
                       ("essay_mae", "主观题 MAE（分）"),
                       ("essay_rmse", "主观题 RMSE（分）"),
                       ("essay_bias", "主观题偏差（分）"),
                       ("essay_within_10pct", "主观题 ±10% 命中率"),
                       ("total_mae", "总分 MAE（分）"),
                       ("total_bias", "总分偏差（分）"),
                       ("total_within_5pts", "总分误差 ≤5 分率"),
                       ("pearson_total", "Pearson r（总分）"),
                       ("spearman_total", "Spearman ρ（总分）")]:
            v = g["stability"].get(k)
            A(f"| {lbl} | {pctms(v) if k in rate_keys else ms(v)} |")
        A("")

        A("### 逐用例明细（第 1 轮）\n")
        A("| 用例 | 人工总分 | 机器总分 | 差值 | 客观题对/总 | 简答[机器] | 简答[人工] |")
        A("|---|---|---|---|---|---|---|")
        for c in g["details"][0]["per_case"]:
            A(f"| {c['case']} | {c['human_total']:.0f} | {c['actual_total']:.0f} | "
              f"{c['total_diff']:+.0f} | {c['objective_correct']}/{c['objective_total']} | "
              f"{c['essay_got']} | {c['essay_human']} |")
        A("")

    # ---------------- 四、跨后端对照 ----------------
    A("## 四、跨后端对照与外部依赖说明\n")
    rows = []
    for p in sorted(EVAL.glob("eval_report_strict*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        dm = d.get("meta", {})
        dv, de, dg = d.get("retrieval", {}), d.get("exam", {}), d.get("grading", {})
        rows.append({
            "file": p.name,
            "backend": dm.get("backend"),
            "stages": "/".join(dm.get("stages_done", [])),
            "hitk": None if dv.get("skipped") else dv.get("overall", {}).get("hit@k"),
            "mrr": None if dv.get("skipped") else dv.get("overall", {}).get("mrr"),
            "online": None if de.get("skipped") else de.get("overall", {}).get("online_rate"),
            "schema": None if de.get("skipped") else de.get("overall", {}).get("schema_rate"),
            "factcov": None if de.get("skipped") else de.get("overall", {}).get("fact_coverage"),
            "essay_mae": None if dg.get("skipped") else dg.get("overall", {}).get("essay_mae"),
            "total_mae": None if dg.get("skipped") else dg.get("overall", {}).get("total_mae"),
            "pearson": None if dg.get("skipped") else dg.get("overall", {}).get("pearson_total"),
            "llm_share": None if dg.get("skipped") else dg.get("overall", {}).get("essay_llm_share"),
            "calls": f"{dm.get('llm_calls_ok', '—')}/{dm.get('llm_calls_total', '—')}",
        })
    if rows:
        A("| 报告 | 后端 | 已完成阶段 | 检索 Hit@6 | MRR | 出题在线率 | 一致性校验 | "
          "知识点覆盖 | 简答 MAE | 总分 MAE | Pearson r | LLM 评分占比 | LLM 成功/总调用 |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for x in rows:
            A(f"| `{x['file']}` | {x['backend']} | {x['stages']} | {pct(x['hitk'])} | "
              f"{num(x['mrr'])} | {pct(x['online'])} | {pct(x['schema'])} | {pct(x['factcov'])} | "
              f"{'—' if x['essay_mae'] is None else x['essay_mae']} | "
              f"{'—' if x['total_mae'] is None else x['total_mae']} | {num(x['pearson'])} | "
              f"{pct(x['llm_share'])} | {x['calls']} |")
        A("")
        A("> 注：检索层为本地向量 + BM25 + BGE 重排，与 LLM 后端无关，各报告数值应一致；"
          "不一致说明知识库或检索配置被改动过。")
        A("")

    A("### 外部依赖限制（本轮实测）\n")
    A("1. **LLM 为本地 Ollama `deepseek-r1:7b`**（7.6B, Q4_K_M），完全本地推理 + GPU 加速，"
      "无云端额度限制；出题与判卷指标均由此模型或离线引擎给出，并已明确标注后端。")
    A("2. **嵌入/重排为本地模型**：`BAAI/bge-base-zh-v1.5` 嵌入 + `BAAI/bge-reranker-base` 重排，"
      "向量库为 Milvus（IVF_SQ8 量化索引，备选 FAISS），检索层确定性可复现。")
    A("3. 7B 模型具备可靠的工具调用能力（create_agent + Function-Calling）："
      "可自主调用 search_safety_knowledge 检索后作答、按工种/层级生成结构化试卷，回答有据可依。")
    A("4. 实测本地 1.5B 模型**担任事实核查裁判能力有限**："
      "要求输出逐题 JSON 数组时可能回吐原文（解析失败），故 `fact_ok_rate` 可能标记为「不可用」，"
      "不以 0% 冒充。")
    A("")
    A("**补测命令**（重跑在线层并合并进同一报告）：")
    A("```bash")
    A("python scripts/run_eval_strict.py --backend ollama --stage exam --runs-exam 3 --tag ollama")
    A("python scripts/run_eval_strict.py --backend ollama --stage grading --runs-grading 3 --tag ollama")
    A("python scripts/render_strict_report.py eval_report_strict_ollama")
    A("```")
    A("")

    # ---------------- 五、口径与局限 ----------------
    A("## 五、口径与局限\n")
    A("1. **术语覆盖严口径**要求关键词出现在「命中的期望来源文件」分块内，"
      "而宽口径只要求出现在 top-k 任意文本中；两者差值即「被通用大文件摊薄」的程度。")
    A("2. **事实核查由同款模型兼任裁判**（LLM-as-judge），存在同源偏差与批次噪声，"
      "故按多轮报告均值 ± 标准差；结论应结合 `failed_examples` 人工复核。")
    A("3. **判卷主观题的人工期望分**按标准答案的 6 个得分点等权折算，"
      "属规则化人工标注；相关系数反映的是「与规则化人工评分的一致性」，"
      "不等同于与多位资深阅卷人的组内一致性。")
    A("4. 检索层为确定性指标（固定向量 + BM25 + 重排模型），单轮即可复现；"
      "出题与主观题判分涉及模型采样，故多轮统计。")
    A("5. 标注可信度：`expected_source` / `expected_terms` 已用脚本逐条校验"
      "（来源可匹配、术语确实出现在该来源的分块文本中），避免臆造标注。")
    A("")

    # ---------------- 六、如何复现 ----------------
    A("## 六、如何复现\n")
    A("```bash")
    A("# 全量严格评测（检索 + 出题 + 判卷）")
    A("python scripts/run_eval_strict.py --backend ollama --runs-exam 3 --runs-grading 3")
    A("")
    A("# 分阶段执行（推荐：长任务可分段，各阶段结果自动合并落盘，可断点续跑）")
    A("python scripts/run_eval_strict.py --stage retrieval            # 纯本地，约 10 分钟")
    A("python scripts/run_eval_strict.py --stage exam                 # 出题层")
    A("python scripts/run_eval_strict.py --stage grading --tag ollama # 判卷层（可指定后端与报告名）")
    A("")
    A("# 本地/离线对照")
    A("python scripts/run_eval_strict.py --backend ollama --stage grading --tag ollama")
    A("python scripts/run_eval_strict.py --backend offline --stage grading --tag offline --no-fact-check")
    A("")
    A("# 渲染报告")
    A(f"python scripts/render_strict_report.py {stem}")
    A("```")
    A("")
    A("改动清单（原 `app/evaluate.py` 与 `eval/eval_questions.json` 未改动，基线仍可复现）：")
    A("- 新增测试集 `eval/eval_questions_strict.json`：检索 80 条 / 出题 8 工种 / 判卷 14 用例；")
    A("- 新增 `app/evaluate_strict.py`（严格评估器）、`scripts/run_eval_strict.py`（分阶段运行）、"
      "`scripts/render_strict_report.py`（渲染本报告）；")
    A("- 产物：`eval/eval_report_strict*.json`（机器可读）与同名 `.md`（人读）；"
      "检索明细缓存 `eval/eval_retrieval_details.json` 供换后端时复用。")
    A("")

    out = EVAL / f"{stem}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"已生成 {out}")


if __name__ == "__main__":
    main()
