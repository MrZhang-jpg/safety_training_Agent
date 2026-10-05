# -*- coding: utf-8 -*-
"""构建安全知识库：加载 DATA_DIR 下全部 EHS 文档 → BGE 向量化 → Milvus/FAISS 持久化。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须在 import app.agent（间接 import transformers/huggingface_hub）之前启用离线
from app.config import settings  # noqa: E402
from app.runtime_env import setup_offline  # noqa: E402

setup_offline(settings.embedding_model, settings.reranker_model, settings.use_reranker)

from app.agent import SafetyTrainingAgent  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="构建安全培训知识库")
    parser.add_argument("--backend", default=None, help="LLM 后端，默认读 .env")
    args = parser.parse_args()

    agent = SafetyTrainingAgent(args.backend)
    result = agent.build_knowledge_base()
    print("\n构建结果：")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("知识库统计：", agent.kb.stats())


if __name__ == "__main__":
    main()
