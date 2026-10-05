# -*- coding: utf-8 -*-
"""
安全培训智能 Agent —— FastAPI 服务入口。

技术栈：LangChain + LangGraph Agent / 本地 Ollama deepseek-r1:7b /
本地 BGE 中文嵌入 / Milvus（默认）· FAISS（备选）向量库 / 混合检索 + Rerank。

接口：
- GET  /health                      健康检查（后端 / 向量库 / 索引类型）
- GET  /                            豆包风格前端页面
- POST /api/chat                    自然语言对话（Agent 自动调度工具，含多轮记忆）
- POST /api/kb/build                构建/重建知识库
- GET  /api/kb/stats                知识库统计
- POST /api/exam/generate           按工种+层级结构化生成试卷
- POST /api/exam/download           生成试卷并下载为 Word（.docx）
- POST /api/material/generate       按工种+层级生成标准化培训资料
- POST /api/material/download       生成培训资料并下载为 Word（.docx）
- POST /api/grade                   结构化判卷（客观题确定性比对 + 主观题辅助评分）
- POST /api/ledger                  生成培训台账
- POST /api/ledger/download         生成培训台账并下载为 Word（.docx）
- POST /api/ocr                     上传试卷图片，返回 OCR 识别文本

启动：python main.py ；接口文档：/docs
"""
from __future__ import annotations

import io
import json
from urllib.parse import quote

# 必须在 import app.agent（间接 import transformers/huggingface_hub）之前启用离线
from app.config import BASE_DIR, settings
from app.runtime_env import setup_offline

setup_offline(settings.embedding_model, settings.reranker_model, settings.use_reranker)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.agent import SafetyTrainingAgent
from app.exporters import exam_to_docx, ledger_to_docx, material_to_docx
from app.schemas import (
    AgentChatRequest, ExamRequest, GradeRequest, LedgerRequest, StandardResponse,
)

app = FastAPI(title="安全培训智能 Agent API", version="2.0.0")

# 全局唯一 Agent（启动时初始化 RAG / LLM / 向量库）
agent = SafetyTrainingAgent()

STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _docx_response(data: bytes, filename: str) -> StreamingResponse:
    """构造 .docx 下载响应（中文文件名使用 RFC 5987 编码）。"""
    return StreamingResponse(
        io.BytesIO(data),
        media_type=_DOCX_MIME,
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            "Content-Length": str(len(data)),
        },
    )


@app.get("/health")
def health():
    stats = agent.kb.stats()
    return {
        "status": "ok",
        "backend": agent.backend,
        "llm_model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
        "vector_db": stats.get("vector_db"),
        "index_type": stats.get("index_type"),
    }


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ============================ 对话 ============================
@app.post("/api/chat")
def chat(req: AgentChatRequest):
    try:
        answer = agent.run(req.query, thread_id=req.thread_id)
        return StandardResponse(success=True, data={"answer": answer})
    except Exception as exc:
        return StandardResponse(success=False, message=str(exc))


# ============================ 知识库 ============================
@app.post("/api/kb/build")
def build_kb():
    try:
        result = agent.build_knowledge_base()
        return StandardResponse(success=True, data=result)
    except Exception as exc:
        return StandardResponse(success=False, message=str(exc))


@app.get("/api/kb/stats")
def kb_stats():
    try:
        return StandardResponse(success=True, data=agent.kb.stats())
    except Exception as exc:
        return StandardResponse(success=False, message=str(exc))


# ============================ 出题 ============================
@app.post("/api/exam/generate")
def generate_exam(req: ExamRequest):
    try:
        exam = agent.get_exam(req.craft_type, req.edu_level, req.question_counts)
        return StandardResponse(success=True, data=exam.model_dump())
    except Exception as exc:
        return StandardResponse(success=False, message=str(exc))


@app.post("/api/exam/download")
def download_exam(req: ExamRequest):
    try:
        exam = agent.get_exam(req.craft_type, req.edu_level, req.question_counts)
        data = exam_to_docx(exam)
        filename = f"安全教育试卷_{exam.craft_type}_{exam.edu_level}.docx"
        return _docx_response(data, filename)
    except Exception as exc:
        return StandardResponse(success=False, message=str(exc))


# ============================ 培训资料 ============================
@app.post("/api/material/generate")
def generate_material(req: ExamRequest):
    """按工种 + 层级生成标准化培训资料（复用 ExamRequest 字段）。"""
    try:
        material = agent.get_material(req.craft_type, req.edu_level)
        return StandardResponse(success=True, data=material.model_dump())
    except Exception as exc:
        return StandardResponse(success=False, message=str(exc))


@app.post("/api/material/download")
def download_material(req: ExamRequest):
    try:
        material = agent.get_material(req.craft_type, req.edu_level)
        data = material_to_docx(material)
        filename = f"安全培训资料_{material.craft_type}_{material.edu_level}.docx"
        return _docx_response(data, filename)
    except Exception as exc:
        return StandardResponse(success=False, message=str(exc))


# ============================ 判卷 ============================
@app.post("/api/grade")
def grade(req: GradeRequest):
    try:
        result = agent.grade(req.exam, req.paper)
        return StandardResponse(success=True, data=result.model_dump())
    except Exception as exc:
        return StandardResponse(success=False, message=str(exc))


# ============================ 台账 ============================
@app.post("/api/ledger")
def ledger(req: LedgerRequest):
    try:
        led = agent.make_ledger(
            req.craft_type, req.edu_level, req.trainee_names,
            req.grade_results, trainer=req.trainer, location=req.location,
            training_date=req.training_date, training_topic=req.training_topic,
        )
        return StandardResponse(success=True, data=led.model_dump())
    except Exception as exc:
        return StandardResponse(success=False, message=str(exc))


@app.post("/api/ledger/download")
def download_ledger(req: LedgerRequest):
    try:
        led = agent.make_ledger(
            req.craft_type, req.edu_level, req.trainee_names,
            req.grade_results, trainer=req.trainer, location=req.location,
            training_date=req.training_date, training_topic=req.training_topic,
        )
        data = ledger_to_docx(led)
        filename = f"安全培训台账_{led.craft_type}_{led.edu_level}.docx"
        return _docx_response(data, filename)
    except Exception as exc:
        return StandardResponse(success=False, message=str(exc))


# ============================ OCR ============================
@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        text = agent.ocr.extract_text(img)
        return StandardResponse(success=True, data={"text": text})
    except Exception as exc:
        return StandardResponse(success=False, message=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port,
                log_level=settings.log_level.lower())
