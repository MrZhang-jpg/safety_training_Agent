# -*- coding: utf-8 -*-
"""
运行严格评估：检索对抗集 + 出题事实核查 + 含简答题的判卷一致性。

用法示例：
    python scripts/run_eval_strict.py --backend ollama --runs-exam 3 --runs-grading 3
    python scripts/run_eval_strict.py --skip-exam --skip-grading        # 只跑检索层（本地、快）
    python scripts/run_eval_strict.py --backend offline --skip-exam     # 离线对照
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.runtime_env import setup_offline  # noqa: E402

setup_offline(settings.embedding_model, settings.reranker_model, settings.use_reranker)

from app.evaluate_strict import StrictAgentEvaluator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="安全培训 Agent 严格评估")
    parser.add_argument("--backend", default=None,
                        help="LLM 后端：ollama|offline|auto，默认读 .env")
    parser.add_argument("--runs-exam", type=int, default=3, help="出题层重复轮数")
    parser.add_argument("--runs-grading", type=int, default=3, help="判卷层重复轮数")
    parser.add_argument("--no-fact-check", action="store_true", help="跳过 LLM 事实核查")
    parser.add_argument("--skip-exam", action="store_true", help="只跑检索层")
    parser.add_argument("--skip-grading", action="store_true", help="跳过判卷层")
    parser.add_argument("--stage", default="all",
                        choices=["all", "retrieval", "exam", "grading"],
                        help="分阶段执行：非 all 时只跑该阶段，并与已有报告合并落盘")
    parser.add_argument("--interval", type=float, default=0.6,
                        help="每次 LLM 调用后的节流间隔（秒），应对免费额度限流")
    parser.add_argument("--tag", default="",
                        help="报告文件名后缀，如 ollama → eval_report_strict_ollama.json")
    args = parser.parse_args()

    stem = f"eval_report_strict_{args.tag}" if args.tag else "eval_report_strict"
    ev = StrictAgentEvaluator(args.backend, call_interval=args.interval,
                              report_stem=stem)
    ev.run_all(runs_exam=args.runs_exam, runs_grading=args.runs_grading,
               fact_check=not args.no_fact_check,
               skip_exam=args.skip_exam, skip_grading=args.skip_grading,
               stage=args.stage)


if __name__ == "__main__":
    main()
