# -*- coding: utf-8 -*-
"""
RAG Studio — FastAPI 后端服务
将 demo1.py 的 RAGSystem 包装为 HTTP API，供前端调用
启动方式: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import re
import sys
import time
import io
import json
import threading
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from persistence_manager import PersistenceManager
from document_manager import DocumentManager, ChunkStrategy
from command_parser import CommandParser, CommandValidator
from kb_manager import KBManager

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import MNN.llm as llm
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError as e:
    FAISS_AVAILABLE = False
    try:
        import numpy as np
    except ImportError:
        np = None
    logging.warning(f"部分依赖不可用: {e}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── 全局单例状态 ──────────────────────────────────────────────────────────────
class RAGState:
    def __init__(self):
        self.llm_model = None
        self.embedder = None
        self.kb_manager: Optional[KBManager] = None
        self.document_manager: Optional[DocumentManager] = None
        self.persistence: Optional[PersistenceManager] = None
        self.command_parser = CommandParser()
        self.command_validator = CommandValidator(self.command_parser)

        self.base_knowledge_fragments: List[str] = []
        self.knowledge_fragments: List[str] = []
        self.fragment_embeddings = None
        self.faiss_index = None
        self.has_loaded_documents = False

        self.config = {
            "llm_config":     r"D:\MNN_RAG_Project\src\model\llm_config.json",
            "bge_model":      r"D:\MNN_RAG_Project\models\bge-m3",
            "knowledge_base": r"D:\MNN_RAG_Project\src\ipl6.md",
            "documents_dir":  r"D:\MNN_RAG_Project\documents",
            "cache_dir":      r"D:\MNN_RAG_Project\cache",
            "top_k": 3,
            "embed_dim": 512,
            "max_tokens": 2048,
        }

        self.model_loaded = False
        self.kb_loaded = False
        self.infer_times: List[float] = []
        self.query_count = 0
        self.start_time = time.time()
        self.last_infer_ms = 0
        self._lock = threading.Lock()


rag = RAGState()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Studio API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 修复1: 前端文件名去掉末尾下划线，与实际文件名一致
FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_frontend.html")

@app.get("/")
def serve_frontend():
    if os.path.exists(FRONTEND):
        return FileResponse(FRONTEND, media_type="text/html")
    return JSONResponse({"msg": "请将 rag_frontend.html 与 server.py 放在同一目录"})


# ── Pydantic 模型 ─────────────────────────────────────────────────────────────
class ConfigUpdate(BaseModel):
    llm_config:     Optional[str] = None
    bge_model:      Optional[str] = None
    knowledge_base: Optional[str] = None
    documents_dir:  Optional[str] = None
    cache_dir:      Optional[str] = None
    top_k:          Optional[int] = None
    embed_dim:      Optional[int] = None
    max_tokens:     Optional[int] = None

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

class KBAddRequest(BaseModel):
    text: str

class KBUpdateRequest(BaseModel):
    index: int
    new_text: str

class KBDeleteRequest(BaseModel):
    index:   Optional[int] = None
    keyword: Optional[str] = None

class KBImportRequest(BaseModel):
    content:   Optional[str] = None
    file_path: Optional[str] = None


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def _record_infer(ms: float):
    rag.last_infer_ms = ms
    rag.infer_times.append(ms)
    if len(rag.infer_times) > 50:
        rag.infer_times.pop(0)
    rag.query_count += 1

def _get_memory_mb() -> float:
    if not HAS_PSUTIL:
        return 0.0
    try:
        return round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 1)
    except Exception:
        return 0.0

def _require_kb():
    if not rag.kb_loaded or rag.kb_manager is None:
        raise HTTPException(400, "知识库未加载，请先在系统配置中加载知识库")

def _require_model():
    if not rag.model_loaded or rag.llm_model is None:
        raise HTTPException(400, "模型未加载，请先在系统配置中加载模型")

def _rebuild_index():
    if rag.embedder is None or not rag.knowledge_fragments:
        return
    logger.info(f"重建向量索引，片段数: {len(rag.knowledge_fragments)}")
    t0 = time.time()
    rag.fragment_embeddings = rag.embedder.encode(
        rag.knowledge_fragments, batch_size=32, show_progress_bar=False
    )
    logger.info(f"向量计算完成 ({time.time()-t0:.2f}s)")

    if FAISS_AVAILABLE:
        emb = rag.fragment_embeddings.astype("float32")
        dim = emb.shape[1]
        rag.faiss_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(emb)
        rag.faiss_index.add(emb)
        logger.info(f"Faiss 索引重建完成，向量数: {rag.faiss_index.ntotal}")

    if rag.persistence:
        rag.persistence.clear_cache()
        rag.persistence.save_embeddings(rag.fragment_embeddings)
        rag.persistence.save_fragments(rag.knowledge_fragments)
        if FAISS_AVAILABLE and rag.faiss_index:
            rag.persistence.save_index(rag.faiss_index)
        rag.persistence.save_metadata({
            "num_fragments": len(rag.knowledge_fragments),
            "embedding_shape": list(rag.fragment_embeddings.shape),
        })

def _retrieve(query: str, top_k: int) -> List[str]:
    if FAISS_AVAILABLE and rag.faiss_index is not None and np is not None:
        qe = rag.embedder.encode([query]).astype("float32")
        faiss.normalize_L2(qe)
        _, indices = rag.faiss_index.search(qe, top_k)
        return [rag.knowledge_fragments[i] for i in indices[0] if i < len(rag.knowledge_fragments)]
    elif rag.fragment_embeddings is not None and np is not None:
        qe = rag.embedder.encode([query])[0]
        sims = [
            float(np.dot(qe, fe) / (np.linalg.norm(qe) * np.linalg.norm(fe) + 1e-8))
            for fe in rag.fragment_embeddings
        ]
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [rag.knowledge_fragments[i] for i in top_idx]
    return []

def _init_support_components():
    cache_dir = rag.config.get("cache_dir", r"D:\MNN_RAG_Project\cache")
    if rag.persistence is None:
        rag.persistence = PersistenceManager(cache_dir)
    if rag.document_manager is None:
        rag.document_manager = DocumentManager(
            chunk_size=200, overlap=50, strategy=ChunkStrategy.RECURSIVE
        )

# 修复2: 统一的文档打标签函数，所有重建路径都经过这里
def _build_tagged_doc_chunks() -> List[str]:
    """
    将所有已加载文档的片段打上来源文件名标签。
    同时在每个文档前插入一条索引片段，使"上传了什么文件"类问题也能被检索到。
    只写入 knowledge_fragments，不触碰 base_knowledge_fragments，隔离完全安全。
    """
    if rag.document_manager is None:
        return []
    result = []
    for doc_name, paragraphs in rag.document_manager.documents.items():
        meta = rag.document_manager.file_metadata.get(doc_name, {})
        filename = os.path.basename(meta.get("path", doc_name))
        chunks = rag.document_manager.chunk_text("\n".join(paragraphs))
        total_chars = sum(len(p) for p in paragraphs)
        # 索引片段：让"上传了什么/有哪些文档"类问题可以命中
        result.append(
            f"[来源文件:{filename}] 用户已上传文档：{filename}，"
            f"共 {len(paragraphs)} 段，{total_chars} 个字符。"
        )
        # 内容片段：每条都带文件名标签，让"告诉我xx文件的内容"可以精确匹配
        result.extend([f"[来源文件:{filename}] {chunk}" for chunk in chunks])
    return result


# ── 配置 & 状态 ───────────────────────────────────────────────────────────────
@app.get("/api/config")
def get_config():
    return rag.config

@app.post("/api/config")
def update_config(cfg: ConfigUpdate):
    updated = {k: v for k, v in cfg.dict().items() if v is not None}
    rag.config.update(updated)
    logger.info(f"配置已更新: {updated}")
    return {"ok": True, "config": rag.config}

@app.get("/api/status")
def get_status():
    avg_infer = round(sum(rag.infer_times) / len(rag.infer_times), 1) if rag.infer_times else 0
    cache_valid = rag.persistence.is_cache_valid() if rag.persistence else False
    cache_info = rag.persistence.get_cache_info() if (rag.persistence and cache_valid) else {}
    return {
        "model_loaded":    rag.model_loaded,
        "kb_loaded":       rag.kb_loaded,
        "kb_path":         rag.config["knowledge_base"],
        "fragment_count":  len(rag.knowledge_fragments),
        "doc_count":       len(rag.document_manager.documents) if rag.document_manager else 0,
        "query_count":     rag.query_count,
        "last_infer_ms":   rag.last_infer_ms,
        "avg_infer_ms":    avg_infer,
        "memory_mb":       _get_memory_mb(),
        "faiss_available": FAISS_AVAILABLE,
        "faiss_vectors":   rag.faiss_index.ntotal if (FAISS_AVAILABLE and rag.faiss_index) else 0,
        "cache_valid":     cache_valid,
        "cache_size_mb":   round(cache_info.get("total_size_mb", 0), 2),
        "embed_dim":       rag.config.get("embed_dim", 512),
        "uptime_s":        round(time.time() - rag.start_time),
    }


# ── 模型管理 ──────────────────────────────────────────────────────────────────
@app.post("/api/model/load")
def load_model():
    if rag.model_loaded:
        return {"ok": True, "msg": "模型已处于加载状态"}

    llm_path   = rag.config["llm_config"]
    embed_path = rag.config["bge_model"]

    if not os.path.exists(llm_path):
        raise HTTPException(400, f"LLM 配置文件不存在: {llm_path}")
    if not os.path.exists(embed_path):
        raise HTTPException(400, f"嵌入模型路径不存在: {embed_path}")

    t0 = time.time()
    try:
        logger.info("加载 LLM 模型...")
        rag.llm_model = llm.create(llm_path)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rag.llm_model.load()

        logger.info("加载嵌入模型...")
        rag.embedder = SentenceTransformer(embed_path)

        _init_support_components()
        rag.model_loaded = True

        # 修复3: 模型加载完成后，补充向量化此前已上传但未能向量化的文档
        if rag.has_loaded_documents and rag.document_manager:
            rag.knowledge_fragments = rag.base_knowledge_fragments.copy()
            rag.knowledge_fragments.extend(_build_tagged_doc_chunks())
        if rag.knowledge_fragments:
            _rebuild_index()

        elapsed = round(time.time() - t0, 1)
        logger.info(f"模型加载完成，耗时 {elapsed}s")
        return {"ok": True, "msg": f"模型加载成功，耗时 {elapsed}s", "memory_mb": _get_memory_mb()}
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise HTTPException(500, f"模型加载失败: {e}")

@app.post("/api/model/unload")
def unload_model():
    rag.llm_model = None
    rag.embedder = None
    rag.model_loaded = False
    logger.info("模型已卸载")
    return {"ok": True, "msg": "模型已卸载"}


# ── 知识库加载 ────────────────────────────────────────────────────────────────
@app.post("/api/kb/load")
def load_kb():
    kb_path = rag.config["knowledge_base"]
    if not os.path.exists(kb_path):
        raise HTTPException(400, f"知识库文件不存在: {kb_path}")

    _init_support_components()
    rag.kb_manager = KBManager(kb_path, chunk_fn=rag.document_manager.chunk_text)

    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            content = f.read()

        rag.base_knowledge_fragments = rag.document_manager.chunk_text(content)
        rag.knowledge_fragments = rag.base_knowledge_fragments.copy()

        # 修复4: 知识库加载时，若已有文档则一并合并（带标签）
        if rag.has_loaded_documents:
            rag.knowledge_fragments.extend(_build_tagged_doc_chunks())

        # 优先从缓存加载（仅在无文档时使用缓存，避免缓存与当前文档状态不一致）
        if rag.embedder and not rag.has_loaded_documents and rag.persistence.is_cache_valid():
            fi, emb, frags, _ = rag.persistence.load_all()
            if fi is not None and emb is not None and frags:
                rag.faiss_index = fi
                rag.fragment_embeddings = emb
                rag.knowledge_fragments = frags
                rag.kb_loaded = True
                return {"ok": True, "msg": f"从缓存加载成功，共 {len(frags)} 条片段", "count": len(frags)}

        if rag.embedder:
            _rebuild_index()

        rag.kb_loaded = True
        count = len(rag.knowledge_fragments)
        return {"ok": True, "msg": f"知识库加载成功，共 {count} 条片段", "count": count}
    except Exception as e:
        raise HTTPException(500, f"知识库加载失败: {e}")

@app.post("/api/kb/rebuild-index")
def rebuild_index():
    _require_kb()
    try:
        _rebuild_index()
        return {"ok": True, "msg": f"向量索引已重建，共 {len(rag.knowledge_fragments)} 条片段"}
    except Exception as e:
        raise HTTPException(500, f"重建失败: {e}")


# ── 知识库 CRUD ───────────────────────────────────────────────────────────────
@app.get("/api/kb/list")
def kb_list(page: int = 1, keyword: Optional[str] = None, page_size: int = 20):
    _require_kb()
    items, total = rag.kb_manager.list_fragments(keyword=keyword, page=page, page_size=page_size)
    return {
        "items": [{"idx": idx, "text": text} for idx, text in items],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }

@app.get("/api/kb/search")
def kb_search(keyword: str):
    _require_kb()
    results = rag.kb_manager.search_fragments(keyword)
    return {"results": [{"idx": i, "text": t} for i, t in results], "count": len(results)}

@app.get("/api/kb/stats")
def kb_stats():
    _require_kb()
    stats = rag.kb_manager.get_stats()
    cache_valid = rag.persistence.is_cache_valid() if rag.persistence else False
    cache_info  = rag.persistence.get_cache_info() if (rag.persistence and cache_valid) else {}
    return {**stats, "cache_valid": cache_valid, "cache_size_mb": round(cache_info.get("total_size_mb", 0), 2)}

@app.post("/api/kb/add")
def kb_add(req: KBAddRequest):
    _require_kb()
    success, msg = rag.kb_manager.add_fragment(req.text)
    if not success:
        raise HTTPException(400, msg)
    _sync_kb_and_rebuild()
    return {"ok": True, "msg": msg}

@app.post("/api/kb/update")
def kb_update(req: KBUpdateRequest):
    _require_kb()
    success, msg = rag.kb_manager.update_fragment(req.index, req.new_text)
    if not success:
        raise HTTPException(400, msg)
    _sync_kb_and_rebuild()
    return {"ok": True, "msg": msg}

@app.post("/api/kb/delete")
def kb_delete(req: KBDeleteRequest):
    _require_kb()
    if req.keyword:
        success, msg, _ = rag.kb_manager.delete_by_keyword(req.keyword)
    elif req.index is not None:
        success, msg = rag.kb_manager.delete_fragment(req.index)
    else:
        raise HTTPException(400, "需要提供 index 或 keyword")
    if not success:
        raise HTTPException(400, msg)
    _sync_kb_and_rebuild()
    return {"ok": True, "msg": msg}

@app.post("/api/kb/import")
def kb_import(req: KBImportRequest):
    _require_kb()
    if req.file_path and os.path.exists(req.file_path):
        success, msg, count = rag.kb_manager.import_from_file(req.file_path)
    elif req.content:
        lines = [l.strip() for l in req.content.splitlines() if l.strip()]
        s, f, _ = rag.kb_manager.add_fragments_batch(lines)
        success, msg, count = True, f"导入完成: 成功 {s} 条，失败 {f} 条", s
    else:
        raise HTTPException(400, "请提供文件路径或内容")
    if not success:
        raise HTTPException(400, msg)
    if count > 0:
        _sync_kb_and_rebuild()
    return {"ok": True, "msg": msg, "count": count}

def _sync_kb_and_rebuild():
    """
    知识库文件（CRUD）变更后，重新读取并重建索引。
    base_knowledge_fragments 只存知识库原始内容，文档片段通过 _build_tagged_doc_chunks()
    追加，两者始终隔离，清除文档不会影响知识库向量。
    """
    if rag.document_manager is None:
        return
    try:
        with open(rag.config["knowledge_base"], "r", encoding="utf-8") as f:
            content = f.read()
        rag.base_knowledge_fragments = rag.document_manager.chunk_text(content)
        rag.knowledge_fragments = rag.base_knowledge_fragments.copy()
        # 修复5: 重建时也使用打标签的文档片段，保持标签一致性
        if rag.has_loaded_documents:
            rag.knowledge_fragments.extend(_build_tagged_doc_chunks())
        if rag.embedder:
            _rebuild_index()
    except Exception as e:
        logger.error(f"同步知识库失败: {e}")


# ── 文档管理 ──────────────────────────────────────────────────────────────────
@app.post("/api/docs/upload")
async def upload_doc(file: UploadFile = File(...)):
    _init_support_components()
    tmp_dir = rag.config.get("documents_dir", r"D:\MNN_RAG_Project\documents")
    os.makedirs(tmp_dir, exist_ok=True)
    save_path = os.path.join(tmp_dir, file.filename)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    success, msg, paragraphs = rag.document_manager.load_document(save_path)
    if not success:
        raise HTTPException(400, msg)

    rag.has_loaded_documents = True
    if rag.embedder:
        # 修复6: 使用统一的打标签函数，base_knowledge_fragments 不受影响
        rag.knowledge_fragments = rag.base_knowledge_fragments.copy()
        rag.knowledge_fragments.extend(_build_tagged_doc_chunks())
        _rebuild_index()

    return {"ok": True, "msg": msg, "chunks": len(paragraphs), "filename": file.filename}

@app.post("/api/docs/load-dir")
def load_dir():
    _init_support_components()
    doc_dir = rag.config.get("documents_dir", "")
    if not os.path.isdir(doc_dir):
        raise HTTPException(400, f"目录不存在: {doc_dir}")

    results = rag.document_manager.load_documents_from_directory(doc_dir)
    if not results:
        return {"ok": True, "msg": "目录中未找到支持的文件", "loaded": 0}

    rag.has_loaded_documents = True
    if rag.embedder:
        # 修复7: 使用统一的打标签函数
        rag.knowledge_fragments = rag.base_knowledge_fragments.copy()
        rag.knowledge_fragments.extend(_build_tagged_doc_chunks())
        _rebuild_index()

    loaded = sum(1 for v in results.values() if v[0])
    return {
        "ok": True, "msg": f"已加载 {loaded} 个文档", "loaded": loaded,
        "details": {k: {"success": v[0], "msg": v[1]} for k, v in results.items()}
    }

@app.get("/api/docs/stats")
def docs_stats():
    if rag.document_manager is None:
        return {"total_documents": 0, "total_paragraphs": 0, "total_characters": 0, "documents": {}}
    return rag.document_manager.get_document_stats()

@app.delete("/api/docs/clear")
def docs_clear():
    """
    清除文档：document_manager 清空，knowledge_fragments 还原为纯知识库内容。
    base_knowledge_fragments 从未被污染，直接 copy 还原，知识库向量完全不受影响。
    """
    if rag.document_manager:
        rag.document_manager.clear_documents()
    rag.has_loaded_documents = False
    rag.knowledge_fragments = rag.base_knowledge_fragments.copy()
    if rag.embedder and rag.knowledge_fragments:
        _rebuild_index()
    return {"ok": True, "msg": "文档已清除"}


# ── 对话查询 ──────────────────────────────────────────────────────────────────
@app.post("/api/query")
def query(req: QueryRequest):
    _require_model()
    _require_kb()

    top_k = req.top_k or rag.config.get("top_k", 3)
    t0 = time.time()

    # 修复8: 检测问题中是否包含文件名，优先精确匹配该文件的所有片段
    filename_hit = re.search(
        r'[\w\u4e00-\u9fa5\-]+\.(txt|md|pdf|docx)', req.question, re.IGNORECASE
    )
    if filename_hit:
        fname = filename_hit.group(0)
        file_frags = [f for f in rag.knowledge_fragments if f"[来源文件:{fname}]" in f]
        relevant = file_frags[: top_k * 3] if file_frags else _retrieve(req.question, top_k)
    else:
        relevant = _retrieve(req.question, top_k)

    # 修复9: 检索为空时直接返回提示，不把空 context 送给模型造成胡乱回答
    if not relevant:
        return {
            "answer": "未能在知识库中检索到相关内容。请确认知识库和文档已正确加载，或换一种提问方式。",
            "sources": [],
            "infer_ms": 0,
            "memory_mb": _get_memory_mb(),
        }

    context = "\n".join([f"- {f}" for f in relevant])
    prompt  = f"""根据以下知识库信息回答问题：

【知识库信息】
{context}

【问题】
{req.question}

【回答】
"""
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            response = rag.llm_model.response(prompt, stream=False)
        answer = response.text if hasattr(response, "text") else str(response)
        if "【回答】" in answer:
            answer = answer[answer.find("【回答】") + 4:].strip()
    except Exception as e:
        raise HTTPException(500, f"推理失败: {e}")

    elapsed_ms = round((time.time() - t0) * 1000)
    _record_infer(elapsed_ms)

    return {
        "answer":    answer,
        "sources":   relevant,
        "infer_ms":  elapsed_ms,
        "memory_mb": _get_memory_mb(),
    }


# ── 缓存管理 ──────────────────────────────────────────────────────────────────
@app.post("/api/cache/clear")
def clear_cache():
    if rag.persistence is None:
        _init_support_components()
    success = rag.persistence.clear_cache()
    return {"ok": success, "msg": "缓存已清除" if success else "清除失败"}

@app.get("/api/cache/info")
def cache_info():
    if rag.persistence is None:
        return {"is_valid": False, "total_size_mb": 0, "file_info": {}}
    return rag.persistence.get_cache_info()


# ── 启动提示 ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    print("\n" + "="*55)
    print("  RAG Studio 服务已启动")
    print("  前端界面: http://localhost:8000")
    print("  API 文档: http://localhost:8000/docs")
    print("="*55 + "\n")