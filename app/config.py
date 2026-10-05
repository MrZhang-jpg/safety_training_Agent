# -*- coding: utf-8 -*-
"""
全局配置中心。

所有参数均可通过环境变量或项目根目录下的 .env 文件覆盖，
便于在本地 Windows / Docker / 内网服务器之间切换，而无需改动代码。

技术栈（对齐项目简历）：
- LLM：本地 Ollama deepseek-r1:7b（默认，GPU 推理），可降级到内置离线确定性引擎
- 嵌入：本地 BGE 中文嵌入模型（BAAI/bge-base-zh-v1.5）
- 向量库：Milvus（默认，可选 IVF / 量化 / 图索引），FAISS 作为本地备选向量库
- 检索：向量 + BM25 混合 → BGE-reranker 重排 → 工种保底
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（app/ 的上一级）
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置。"""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- LLM（默认本地 Ollama deepseek-r1:7b）----------
    llm_backend: str = Field(
        default="ollama",
        description="LLM 后端：ollama | offline | auto",
    )
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="deepseek-r1:7b")

    temperature: float = 0.1
    max_agent_iterations: int = 8

    # ---------- 嵌入 / 重排（本地 BGE）----------
    embedding_model: str = Field(default="BAAI/bge-base-zh-v1.5")
    embedding_dim: int = Field(default=768, description="嵌入向量维度（bge-base-zh-v1.5 为 768）")
    reranker_model: str = Field(default="BAAI/bge-reranker-base")
    device: str = Field(default="cpu")
    use_reranker: bool = True
    use_bm25: bool = True

    # ---------- 向量库 ----------
    # 可选：milvus（默认，需 Milvus 服务）| faiss（本地备选）
    vector_db: str = Field(
        default="milvus",
        description="向量库后端：milvus | faiss",
    )
    # Milvus 不可用时是否自动降级到 FAISS（便于本地无 Milvus 环境开发）
    vector_db_fallback_faiss: bool = True

    # Milvus 连接：milvus_uri 优先（如 http://localhost:19530 或 Milvus Lite 文件路径）；
    # 为空时使用 host + port
    milvus_uri: str = Field(default="", description="Milvus 连接 URI，为空则用 host/port")
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)
    milvus_collection: str = Field(default="ehs_safety_training")
    # 索引类型：IVF_FLAT（IVF 倒排）、IVF_SQ8 / IVF_PQ（量化索引）、HNSW（图索引）、FLAT（暴力检索）
    milvus_index_type: str = Field(
        default="IVF_SQ8",
        description="Milvus 索引：FLAT | IVF_FLAT | IVF_SQ8 | IVF_PQ | HNSW",
    )
    milvus_metric_type: str = Field(default="COSINE")
    milvus_nlist: int = Field(default=128, description="IVF 系列索引的聚类数 nlist")
    milvus_nprobe: int = Field(default=16, description="IVF 系列检索探测数 nprobe")
    milvus_pq_m: int = Field(default=32, description="IVF_PQ 量化子空间数 m（需整除维度）")
    milvus_hnsw_m: int = Field(default=16, description="HNSW 图索引每个节点的最大连接数")
    milvus_hnsw_ef_construction: int = Field(default=200, description="HNSW 构建时的动态列表大小")
    milvus_hnsw_ef_search: int = Field(default=64, description="HNSW 检索时的动态列表大小")

    # FAISS 备选向量库
    faiss_dir: str = Field(default="./faiss_db")
    faiss_index_type: str = Field(
        default="IVF_FLAT",
        description="FAISS 索引：FLAT | IVF_FLAT | IVF_SQ8",
    )
    faiss_nlist: int = Field(default=100)
    faiss_nprobe: int = Field(default=16)

    # ---------- 路径 ----------
    data_dir: str = Field(default=r"D:\2-Agent\datas", description="知识库原始数据目录")
    cache_dir: str = Field(default="./data_cache")
    log_dir: str = Field(default="./logs")

    # ---------- 分块 / 检索 ----------
    chunk_size: int = 500
    chunk_overlap: int = 80
    retrieve_k: int = 6

    # ---------- 服务 ----------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # ---------- 文档纳入规则 ----------
    include_dirs: tuple[str, ...] = ()
    excluded_extensions: tuple[str, ...] = (".ppt",)
    supported_extensions: tuple[str, ...] = (
        ".docx", ".doc", ".pdf", ".pptx", ".xlsx", ".xls", ".txt",
    )

    # ---------- 路径解析为绝对路径 ----------
    def resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def faiss_path(self) -> Path:
        p = self.resolve_path(self.faiss_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_path(self) -> Path:
        return self.resolve_path(self.cache_dir)

    @property
    def converted_path(self) -> Path:
        p = self.resolve_path(self.cache_dir) / "converted"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def log_path(self) -> Path:
        p = self.resolve_path(self.log_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例配置，避免重复读取 .env。"""
    return Settings()


# 便于 `from app.config import settings`
settings = get_settings()
