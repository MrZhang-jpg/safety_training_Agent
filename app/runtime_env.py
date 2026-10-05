# -*- coding: utf-8 -*-
"""
运行时环境引导。

必须在 import 任何会间接导入 huggingface_hub / transformers 的模块
（例如 app.agent、langchain_huggingface、sentence_transformers）之前调用
setup_offline()。

原因：huggingface_hub / transformers 在「模块导入时」就读取 HF_HUB_OFFLINE /
TRANSFORMERS_OFFLINE 并固化为模块常量；导入之后再设置 os.environ 不会生效。
本模块只用标准库 pathlib 扫描本地缓存，自身不导入任何重型依赖。

当检测到嵌入与重排模型均已在本地缓存时，启用离线模式，避免 CPU 环境联网卡顿；
若缓存不存在（例如容器内首次运行需要下载），则保持在线，由 HF 正常下载。
"""
from __future__ import annotations

import os
from pathlib import Path


def _model_cached(repo_id: str) -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
    snap = hub / "snapshots"
    if not snap.exists():
        return False
    # 至少一个快照目录内含有配置文件
    for p in snap.iterdir():
        if p.is_dir() and any(x.name == "config.json" for x in p.iterdir()):
            return True
    return False


def setup_offline(embedding_model: str = "BAAI/bge-base-zh-v1.5",
                  reranker_model: str = "BAAI/bge-reranker-base",
                  use_reranker: bool = True) -> bool:
    """检测到本地模型缓存则启用 HF 离线模式，返回是否启用。"""
    need = [embedding_model] + ([reranker_model] if use_reranker else [])
    if all(_model_cached(m) for m in need):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return True
    return False
