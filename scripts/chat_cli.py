# -*- coding: utf-8 -*-
"""命令行多轮对话客户端。输入 exit/quit 退出。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.runtime_env import setup_offline  # noqa: E402

setup_offline(settings.embedding_model, settings.reranker_model, settings.use_reranker)

from app.agent import SafetyTrainingAgent  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="安全培训 Agent 命令行对话")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--thread", default="cli")
    args = parser.parse_args()

    agent = SafetyTrainingAgent(args.backend)
    print("安全培训智能 Agent（输入 exit 退出，后端：{}）".format(agent.backend))
    while True:
        try:
            query = input("\n我> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("exit", "quit", "q"):
            break
        if not query:
            continue
        try:
            print("\nAgent> " + agent.run(query, thread_id=args.thread))
        except Exception as exc:
            print("调用失败：", exc)


if __name__ == "__main__":
    main()
