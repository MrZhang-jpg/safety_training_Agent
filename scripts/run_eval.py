# -*- coding: utf-8 -*-
"""运行评估：检索命中率/MRR、出题合规率、判卷准确率，输出 eval/eval_report.json。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.runtime_env import setup_offline  # noqa: E402

setup_offline(settings.embedding_model, settings.reranker_model, settings.use_reranker)

from app.evaluate import AgentEvaluator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="安全培训 Agent 评估")
    parser.add_argument(
        "--backend", default=None,
        help="LLM 后端：ollama|offline|auto，默认读 .env",
    )
    args = parser.parse_args()
    AgentEvaluator(args.backend).run_all()


if __name__ == "__main__":
    main()
