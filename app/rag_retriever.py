# -*- coding: utf-8 -*-
"""
RAG 检索底座（面向对象）。

流程：多格式文档 → 中文语义分块 → BGE 本地向量化 → 向量库持久化；
检索：向量检索 + BM25 关键词检索（Ensemble 混合）→ BGE-reranker 重排 → 带来源返回。

向量库支持（可配置，对齐项目简历）：
- Milvus（默认）：支持 IVF_FLAT（IVF 倒排）、IVF_SQ8 / IVF_PQ（量化索引）、
  HNSW（图索引）、FLAT（暴力检索），度量 COSINE，索引参数全部可配置；
- FAISS（本地备选向量库）：支持 FLAT / IVF_FLAT / IVF_SQ8（量化）索引。

全部在 CPU 上运行；模型默认已缓存于本地 HuggingFace 目录，
检测到完整缓存后自动切换为离线模式，避免联网卡顿。
"""
from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

COLLECTION_NAME = "ehs_safety_training"


def _model_cached(repo_id: str) -> bool:
    """检测 HuggingFace 默认缓存中是否已存在指定模型快照。"""
    hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
    snap = hub / "snapshots"
    return snap.exists() and any(p.is_dir() and any(p.iterdir()) for p in snap.iterdir())


def _enable_hf_offline_if_cached() -> None:
    """嵌入与重排模型都已缓存时，启用 HF 离线模式，避免 CPU 环境联网等待。"""
    need = [settings.embedding_model]
    if settings.use_reranker:
        need.append(settings.reranker_model)
    if all(_model_cached(m) for m in need):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _cn_preprocess(text: str) -> list[str]:
    """中文分词：优先 jieba；未安装则用字级 unigram + bigram（零依赖，短词命中好）。"""
    text = str(text)
    try:
        import jieba

        toks = jieba.lcut(text)
        out = [t.strip().lower() for t in toks if t.strip()]
        if out:
            return out
    except Exception:
        pass
    han = re.findall(r"[\u4e00-\u9fa5]", text)
    bigrams = [han[i] + han[i + 1] for i in range(len(han) - 1)]
    alnum = re.findall(r"[a-z0-9]+", text.lower())
    return han + bigrams + alnum


# 标准工种名 → 文件名关键词（用于按工种确定性召回专属资料，避免被通用大文件淹没）
CRAFT_FILE_HINTS: dict[str, list[str]] = {
    "电工": ["电工"],
    "电焊工": ["电焊"],
    "架子工": ["架子", "脚手架", "四知卡"],
    "钢筋工": ["钢筋"],
    "塔吊司机": ["塔吊"],
    "起重吊装工": ["起重", "吊装"],
    "司索信号工": ["司索", "信号", "起重", "吊装"],
    "高处作业工": ["高空", "高处"],
    "基坑作业工": ["基坑"],
    "人工挖孔桩工": ["人工挖孔桩", "挖孔桩"],
    "钻孔桩工": ["钻孔桩", "桩工"],
    "破桩工": ["破桩"],
    "桩基工": ["桩基", "钻孔桩", "桩工", "破桩"],
    "钢板桩工": ["钢板桩"],
    "钢支撑安装工": ["钢支撑"],
    "钢立柱安装工": ["钢立柱"],
    "挂篮安装工": ["挂篮"],
    "张拉压浆工": ["张拉", "压浆"],
    "斜拉索工": ["斜拉索"],
    "混凝土工": ["浇砼", "混凝土", "砼"],
    "拌合站操作员": ["拌合站"],
    "搅拌机操作手": ["搅拌机"],
    "罐车司机": ["罐车"],
    "装载机司机": ["装载机"],
    "渣土车司机": ["渣土"],
    "车辆驾驶员": ["车辆", "机动车"],
    "厂内机动车司机": ["机动车", "车辆"],
    "爆破工": ["爆破"],
    "隧道作业工": ["隧道"],
    "桥涵作业工": ["桥涵", "桥梁"],
    "桥面系作业工": ["桥面系", "桥面"],
    "连续梁作业工": ["连续梁"],
    "路基作业工": ["路基"],
    "挖掘机司机": ["挖机", "挖掘机"],
    "施工升降机司机": ["升降机", "电梯"],
    "门式起重机司机": ["门式起重机", "起重机"],
    "机械设备操作员": ["机械"],
    "通用工种": [],
}


@dataclass
class ScoredDoc:
    """带分数与来源的检索结果。"""

    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0

    @property
    def source(self) -> str:
        return self.metadata.get("source", "未知来源")


# =====================================================================
# 向量库后端（可插拔）：Milvus（默认）/ FAISS（本地备选）
# =====================================================================
class MilvusStore:
    """Milvus 向量库后端（pymilvus MilvusClient）。

    支持索引：FLAT / IVF_FLAT / IVF_SQ8 / IVF_PQ（量化）/ HNSW（图索引），
    度量 COSINE；索引参数（nlist / nprobe / M / ef / pq_m）均可通过配置控制。
    """

    name = "milvus"

    def __init__(self, embeddings, dim: Optional[int] = None):
        self.embeddings = embeddings
        self.dim = dim or settings.embedding_dim
        self.collection_name = settings.milvus_collection
        self._client = self._connect()
        self._ensure_collection()

    # ---------- 连接 ----------
    def _connect(self):
        from pymilvus import MilvusClient

        uri = settings.milvus_uri.strip()
        if not uri:
            uri = f"http://{settings.milvus_host}:{settings.milvus_port}"
        return MilvusClient(uri=uri)

    # ---------- 索引参数 ----------
    def _index_params(self) -> dict:
        idx = settings.milvus_index_type.upper()
        metric = settings.milvus_metric_type.upper()
        params: dict = {}
        if idx in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
            params["nlist"] = settings.milvus_nlist
            if idx == "IVF_PQ":
                params["m"] = settings.milvus_pq_m
        elif idx == "HNSW":
            params["M"] = settings.milvus_hnsw_m
            params["efConstruction"] = settings.milvus_hnsw_ef_construction
        return {"index_type": idx, "metric_type": metric, "params": params}

    @staticmethod
    def _wrap_index_params(params: dict):
        """兼容 pymilvus 3.x（要求 IndexParams 对象）与 2.x（接受 dict）。"""
        try:
            from pymilvus.milvus_client.index import IndexParams

            ip = IndexParams()
            ip.add_index(field_name="vector",
                         index_type=params["index_type"],
                         metric_type=params["metric_type"],
                         params=params.get("params", {}))
            return ip
        except Exception:
            return params

    def _search_params(self) -> dict:
        idx = settings.milvus_index_type.upper()
        params: dict = {}
        if idx in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
            params["nprobe"] = settings.milvus_nprobe
        elif idx == "HNSW":
            params["ef"] = settings.milvus_hnsw_ef_search
        return {"metric_type": settings.milvus_metric_type.upper(), "params": params}

    # ---------- 集合 ----------
    def _ensure_collection(self):
        from pymilvus import CollectionSchema, DataType, FieldSchema

        if not self._client.has_collection(self.collection_name):
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
                    FieldSchema(name="page", dtype=DataType.VARCHAR, max_length=128),
                ],
                description="EHS 安全培训知识库（BGE 中文嵌入）",
            )
            self._client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=self._wrap_index_params(self._index_params()),
            )
        else:
            # 已存在：按当前配置重建索引（自动替换旧索引，保证索引类型/参数与配置一致）
            self._client.create_index(
                collection_name=self.collection_name,
                index_params=self._wrap_index_params(self._index_params()),
            )

    # ---------- 写入 ----------
    def add(self, texts: list[str], metadatas: list[dict]) -> None:
        if not texts:
            return
        vecs = self.embeddings.embed_documents(texts)
        rows = []
        for text, meta, vec in zip(texts, metadatas, vecs):
            rows.append({
                "vector": [float(x) for x in vec],
                "text": str(text),
                "source": str(meta.get("source", "")),
                "page": str(meta.get("page", "") or ""),
            })
        self._client.insert(collection_name=self.collection_name, data=rows)
        # Milvus 写入为异步可见，显式 flush 保证后续检索/统计立即可见
        self._client.flush(collection_name=self.collection_name)

    # ---------- 检索 ----------
    def search(self, query: str, k: int) -> list[Document]:
        vec = self.embeddings.embed_query(query)
        results = self._client.search(
            collection_name=self.collection_name,
            data=[[float(x) for x in vec]],
            limit=k,
            output_fields=["text", "source", "page"],
            search_params=self._search_params(),
        )
        docs: list[Document] = []
        if results and results[0]:
            for hit in results[0]:
                entity = hit.get("entity", {})
                docs.append(Document(
                    page_content=entity.get("text", ""),
                    metadata={
                        "source": entity.get("source", ""),
                        "page": entity.get("page", ""),
                        "score": float(hit.get("distance", 0.0)),
                    },
                ))
        return docs

    # ---------- 统计 / 重置 ----------
    def count(self) -> int:
        try:
            stats = self._client.get_collection_stats(self.collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            # 降级：count(*) 查询
            try:
                res = self._client.query(
                    collection_name=self.collection_name,
                    filter="pk >= 0",
                    output_fields=["count(*)"],
                )
                return int(res[0]["count(*)"] if res else 0)
            except Exception:
                return -1

    def reset(self) -> None:
        if self._client.has_collection(self.collection_name):
            self._client.drop_collection(self.collection_name)
        self._ensure_collection()


class FaissStore:
    """FAISS 本地备选向量库（faiss-cpu）。

    支持索引：FLAT（暴力） / IVF_FLAT（IVF 倒排） / IVF_SQ8（标量量化），
    度量内积（向量已归一化，等价余弦相似度）；索引参数 nlist / nprobe 可配置。
    """

    name = "faiss"

    def __init__(self, embeddings, dim: Optional[int] = None):
        self.embeddings = embeddings
        self.dim = dim or settings.embedding_dim
        self.path = settings.faiss_path
        self._index_path = self.path / "index.bin"
        self._meta_path = self.path / "meta.pkl"
        self._vecs: list[np.ndarray] = []
        self._meta: list[dict] = []
        self._index = None
        self._dirty = True
        self._load()

    # ---------- 持久化 ----------
    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            try:
                self._index = self._read_index()
                with open(self._meta_path, "rb") as f:
                    data = pickle.load(f)
                self._vecs = [np.asarray(v, dtype=np.float32) for v in data.get("vecs", [])]
                self._meta = data.get("meta", [])
                self._dirty = False
            except Exception as exc:
                print(f"[WARN] FAISS 索引加载失败，将重建: {exc}")

    def _read_index(self):
        import faiss

        return faiss.read_index(str(self._index_path))

    def _save(self) -> None:
        try:
            with open(self._meta_path, "wb") as f:
                pickle.dump(
                    {"vecs": [v.tolist() for v in self._vecs], "meta": self._meta}, f
                )
        except Exception as exc:
            print(f"[WARN] FAISS 元数据保存失败: {exc}")

    # ---------- 写入 ----------
    def add(self, texts: list[str], metadatas: list[dict]) -> None:
        if not texts:
            return
        vecs = self.embeddings.embed_documents(texts)
        for text, meta, vec in zip(texts, metadatas, vecs):
            self._vecs.append(np.asarray(vec, dtype=np.float32))
            self._meta.append({
                "text": str(text),
                "source": str(meta.get("source", "")),
                "page": str(meta.get("page", "") or ""),
            })
        self._dirty = True
        self._save()

    # ---------- 索引构建（懒构建 + 缓存）----------
    def _build_index(self):
        import faiss

        if not self._dirty and self._index is not None:
            return self._index

        mat = (np.vstack(self._vecs).astype(np.float32)
               if self._vecs else np.zeros((0, self.dim), dtype=np.float32))
        idx_type = settings.faiss_index_type.upper()
        nlist = settings.faiss_nlist

        if idx_type == "IVF_SQ8":
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFScalarQuantizer(
                quantizer, self.dim, nlist, faiss.ScalarQuantizer.QT_8bit)
        elif idx_type == "IVF_FLAT":
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(self.dim), self.dim, nlist)
        else:
            index = faiss.IndexFlatIP(self.dim)

        if getattr(index, "is_trained", False) is False:
            if mat.shape[0] >= nlist * 5:
                faiss.normalize_L2(mat)
                index.train(mat)
            else:
                # 数据不足无法训练 IVF 时退化为暴力检索，保证可用
                print(f"[WARN] 数据量不足（{mat.shape[0]}）无法训练 IVF（nlist={nlist}），"
                      f"本次检索退化为 FLAT 暴力检索。")
                index = faiss.IndexFlatIP(self.dim)
        if hasattr(index, "nprobe"):
            index.nprobe = settings.faiss_nprobe
        if mat.shape[0]:
            faiss.normalize_L2(mat)
            index.add(mat)

        self._index = index
        self._dirty = False
        self._persist_index(index)
        return index

    def _persist_index(self, index) -> None:
        import faiss

        try:
            faiss.write_index(index, str(self._index_path))
        except Exception as exc:
            print(f"[WARN] FAISS 索引写入失败: {exc}")

    # ---------- 检索 ----------
    def search(self, query: str, k: int) -> list[Document]:
        if not self._vecs:
            return []
        vec = np.asarray(self.embeddings.embed_query(query), dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        index = self._build_index()
        distances, indices = index.search(vec, k)
        docs: list[Document] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            m = self._meta[int(idx)]
            docs.append(Document(
                page_content=m["text"],
                metadata={"source": m["source"], "page": m["page"], "score": float(dist)},
            ))
        return docs

    # ---------- 统计 / 重置 ----------
    def count(self) -> int:
        return len(self._vecs)

    def reset(self) -> None:
        self._vecs.clear()
        self._meta.clear()
        self._index = None
        self._dirty = True
        for p in (self._index_path, self._meta_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass


def create_vector_store(embeddings):
    """按配置创建向量库后端：milvus（默认）/ faiss；Milvus 不可用时自动降级 FAISS。"""
    backend = (settings.vector_db or "milvus").lower()
    if backend == "milvus":
        try:
            return MilvusStore(embeddings)
        except Exception as exc:
            if settings.vector_db_fallback_faiss:
                print(f"[WARN] Milvus 连接失败（{exc!r}），自动降级到 FAISS 本地向量库。")
                return FaissStore(embeddings)
            raise
    if backend == "faiss":
        return FaissStore(embeddings)
    raise ValueError(f"未知向量库后端: {backend}（可选 milvus/faiss）")


# =====================================================================
# 安全知识库
# =====================================================================
class SafetyKnowledgeBase:
    """安全知识库：构建 + 混合检索 + 重排。"""

    def __init__(self):
        _enable_hf_offline_if_cached()
        self._bm25_path = settings.cache_path / "bm25_chunks.pkl"

        # 中文嵌入（本地 BGE）
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 中文语义分块
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "：", "，", " ", ""],
            keep_separator=True,
        )

        # 向量库（Milvus 默认 / FAISS 备选）
        self.store = create_vector_store(self.embeddings)
        # 兼容旧引用
        self.vector_db = self.store

        self._reranker = None
        self._bm25_retriever = None
        self._chunks: list[Document] = []

    # ---------- 重排（懒加载）----------
    def _get_reranker(self):
        if self._reranker is None and settings.use_reranker:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                settings.reranker_model, device=settings.device
            )
        return self._reranker

    # ---------- 构建 ----------
    def build(self, documents: list[Document], rebuild: bool = True) -> int:
        if rebuild:
            self.store.reset()

        chunks = self.splitter.split_documents(documents)
        # 去除内容完全重复的块
        seen, unique = set(), []
        for c in chunks:
            key = c.page_content.strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(c)
        chunks = unique

        if not chunks:
            print("[WARN] 没有可入库的文本块。")
            return 0

        # 分批写入向量库，降低 CPU 峰值内存
        batch = 128
        for i in range(0, len(chunks), batch):
            sub = chunks[i:i + batch]
            self.store.add(
                [c.page_content for c in sub],
                [dict(c.metadata) for c in sub],
            )
            print(f"  向量入库 {min(i + batch, len(chunks))}/{len(chunks)}")

        # 持久化分块，供 BM25 使用
        self._chunks = chunks
        self._dump_bm25_chunks(chunks)
        self._bm25_retriever = self._build_bm25(chunks)

        print(f"知识库构建完成，共 {len(chunks)} 个文本块。")
        return len(chunks)

    # ---------- BM25 ----------
    def _dump_bm25_chunks(self, chunks: list[Document]) -> None:
        data = [{"content": c.page_content, "metadata": c.metadata} for c in chunks]
        with open(self._bm25_path, "wb") as f:
            pickle.dump(data, f)

    def _load_bm25_chunks(self) -> list[Document]:
        if not self._bm25_path.exists():
            return []
        with open(self._bm25_path, "rb") as f:
            data = pickle.load(f)
        return [Document(page_content=d["content"], metadata=d["metadata"]) for d in data]

    def _build_bm25(self, chunks: list[Document]):
        if not settings.use_bm25 or not chunks:
            return None
        try:
            from langchain_community.retrievers import BM25Retriever

            retriever = BM25Retriever.from_documents(
                chunks, preprocess_func=_cn_preprocess
            )
            return retriever
        except Exception as exc:
            print(f"[WARN] BM25 初始化失败，将仅使用向量检索: {exc}")
            return None

    def _ensure_bm25(self) -> None:
        if self._bm25_retriever is None and settings.use_bm25:
            chunks = self._load_bm25_chunks()
            self._chunks = chunks
            self._bm25_retriever = self._build_bm25(chunks)

    # ---------- 检索 ----------
    def search(self, query: str, k: Optional[int] = None,
               craft_type: Optional[str] = None) -> list[ScoredDoc]:
        k = k or settings.retrieve_k
        self._ensure_bm25()

        # 自动识别工种（显式传入优先）
        craft = craft_type if (craft_type and craft_type != "通用工种") \
            else self._auto_craft(query)

        # 1) 语义候选（向量 + BM25），按来源多样性限额
        fetch = max(k * 4, 24)
        sem = self._diversify_by_source(
            self._gather_candidates(query, fetch), per_file=2
        )

        # 2) 按文件名确定性召回工种专属块
        craft_docs = self._docs_by_craft_files(craft) if craft else []

        # 3) 合并（工种块在前，按 source+内容去重）
        def _key(d: Document):
            return (d.metadata.get("source"), d.page_content[:80])

        seen = {_key(d) for d in craft_docs}
        merged = list(craft_docs) + [
            d for d in sem if _key(d) not in seen
        ]

        # 4) 重排
        ranked = self._rerank(query, merged, limit=None)

        # 5) 保底组装：工种专属文件稳定进入 top-k，同时来源多样
        hints = CRAFT_FILE_HINTS.get(craft, []) if craft else []
        return self._assemble_with_guarantee(
            ranked, hints, k, per_file=2, guarantee=3
        )

    # ---------- 自动工种识别 ----------
    def _auto_craft(self, query: str) -> Optional[str]:
        from .llm_factory import RuleBasedChatLLM

        name = RuleBasedChatLLM._extract_craft(query)
        return None if name == "通用工种" else name

    # ---------- 按文件名召回工种专属块 ----------
    def _docs_by_craft_files(self, craft: str, per_file: int = 2) -> list[Document]:
        hints = CRAFT_FILE_HINTS.get(craft, [])
        if not hints:
            return []
        from collections import defaultdict

        buckets: dict = defaultdict(list)
        order: list[str] = []
        for d in self._chunks:
            src = d.metadata.get("source", "")
            fname = src.split("/")[-1]
            if any(h in fname for h in hints):
                if src not in buckets:
                    order.append(src)
                buckets[src].append(d)

        out: list[Document] = []
        for s in order:
            out.extend(buckets[s][:per_file])
        return out

    # ---------- 保底组装 ----------
    @staticmethod
    def _assemble_with_guarantee(ranked: list[ScoredDoc], hints: list[str],
                                 k: int, per_file: int = 2,
                                 guarantee: int = 3) -> list[ScoredDoc]:
        def src(d: ScoredDoc) -> str:
            return d.metadata.get("source", "")

        def is_craft(d: ScoredDoc) -> bool:
            fname = src(d).split("/")[-1]
            return any(h in fname for h in hints)

        counts: dict[str, int] = {}
        out: list[ScoredDoc] = []

        def add(d: ScoredDoc) -> bool:
            s = src(d)
            if counts.get(s, 0) < per_file:
                out.append(d)
                counts[s] = counts.get(s, 0) + 1
                return True
            return False

        # 先保证一定数量的工种专属块
        if hints:
            for d in ranked:
                if is_craft(d) and sum(1 for x in out if is_craft(x)) < guarantee:
                    add(d)
        # 再按重排分数顺序填满
        for d in ranked:
            if len(out) >= k:
                break
            add(d)
        return out[:k]

    def _gather_candidates(self, query: str, fetch: int) -> list[Document]:
        vec_docs = self.store.search(query, k=fetch)
        merged: list[Document] = list(vec_docs)

        if self._bm25_retriever is not None:
            self._bm25_retriever.k = fetch
            bm_docs = self._bm25_retriever.invoke(query)
            seen = {(d.metadata.get("source"), d.page_content[:80]) for d in merged}
            for d in bm_docs:
                key = (d.metadata.get("source"), d.page_content[:80])
                if key not in seen:
                    seen.add(key)
                    merged.append(d)
        return merged

    @staticmethod
    def _diversify_by_source(docs: list[Document], per_file: int) -> list[Document]:
        """轮转从各来源文件取块，每个文件最多 per_file 个，提升来源多样性。"""
        from collections import defaultdict

        buckets: dict = defaultdict(list)
        order: list[str] = []
        for d in docs:
            s = d.metadata.get("source")
            if s not in buckets:
                order.append(s)
            buckets[s].append(d)

        out: list[Document] = []
        progress = True
        while progress:
            progress = False
            for s in order:
                have = sum(1 for x in out if x.metadata.get("source") == s)
                if have < per_file and buckets[s]:
                    out.append(buckets[s].pop(0))
                    progress = True
        return out

    @staticmethod
    def _boost_craft(docs: list[Document], craft: str) -> list[Document]:
        key = craft[:2]
        hit = [d for d in docs
               if key in d.page_content or key in str(d.metadata.get("source", ""))]
        rest = [d for d in docs if d not in hit]
        return hit + rest

    @staticmethod
    def _cap_sources(docs: list[ScoredDoc], k: int, per_file: int = 2) -> list[ScoredDoc]:
        counts: dict[str, int] = {}
        out: list[ScoredDoc] = []
        for d in docs:
            s = d.metadata.get("source")
            if counts.get(s, 0) < per_file:
                out.append(d)
                counts[s] = counts.get(s, 0) + 1
            if len(out) >= k:
                break
        return out

    def _rerank(self, query: str, docs: list[Document],
                limit: Optional[int] = None) -> list[ScoredDoc]:
        if not docs:
            return []
        reranker = self._get_reranker() if settings.use_reranker else None
        if reranker is not None:
            pairs = [(query, d.page_content) for d in docs]
            try:
                scores = reranker.predict(pairs)
                ranked = sorted(zip(docs, scores),
                                key=lambda x: float(x[1]), reverse=True)
                result = [
                    ScoredDoc(content=d.page_content, metadata=d.metadata, score=float(s))
                    for d, s in ranked
                ]
                return result[:limit] if limit else result
            except Exception as exc:
                print(f"[WARN] 重排失败，使用原始顺序: {exc}")
        result = [ScoredDoc(content=d.page_content, metadata=d.metadata, score=0.0)
                  for d in docs]
        return result[:limit] if limit else result

    # ---------- 上下文格式化（带来源溯源）----------
    @staticmethod
    def format_context(docs: list[ScoredDoc]) -> str:
        blocks = []
        for i, d in enumerate(docs, start=1):
            src = d.source
            page = d.metadata.get("page") or d.metadata.get("sheet")
            loc = f"（{page}）" if page else ""
            blocks.append(f"[资料{i}] 来源：{src}{loc}\n{d.content.strip()}")
        return "\n\n".join(blocks)

    def stats(self) -> dict:
        return {
            "vector_db": self.store.name,
            "vector_chunks": self.store.count(),
            "bm25_chunks": len(self._load_bm25_chunks()),
            "index_type": (settings.milvus_index_type if self.store.name == "milvus"
                           else settings.faiss_index_type),
        }
