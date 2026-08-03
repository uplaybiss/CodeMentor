"""Document upload and retrieval endpoints."""

from __future__ import annotations

import contextlib
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from api.dependencies import get_container_dep
from models.enums import DocumentStatus
from services.container import Container
from utils.errors import IngestionError
from utils.helpers import generate_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("/{student_id}")
async def upload_document(
    student_id: str,
    file: UploadFile = File(...),
    container: Container = Depends(get_container_dep),
) -> dict:
    """Upload a document (PDF/TXT/MD), chunk it and index it in ChromaDB."""
    from services.rag.ingestion import ingest_document

    filename = file.filename or "uploaded_document"
    content = await file.read()
    doc_id = generate_id("doc")

    try:
        # Ensure the student row exists (FK on documents.student_id) so a fresh
        # student id can upload documents directly.
        container.student_repo.get_or_create(student_id)
        # Persist metadata row first so a failed index is still trackable.
        record = container.document_repo.create(
            document_id=doc_id,
            student_id=student_id,
            filename=filename,
            status=DocumentStatus.UPLOADED.value,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not create document record: %s", exc)
        record = None

    # Write bytes to a temp file so ingestion can read PDFs by path.
    suffix = Path(filename).suffix.lower()
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        chunks = ingest_document(tmp_path, source=filename)
    except IngestionError as exc:
        if record is not None:
            container.document_repo.update_status(
                doc_id, DocumentStatus.FAILED.value
            )
        return {"error": str(exc), "status": "failed"}
    finally:
        with contextlib.suppress(NameError, OSError):
            Path(tmp_path).unlink(missing_ok=True)

    chunk_count = container.rag.ingest_chunks(doc_id, chunks)
    if record is not None:
        container.document_repo.update_status(
            doc_id, DocumentStatus.READY.value, chunk_count=chunk_count
        )
    logger.info("Document %s -> %d chunks for student %s", doc_id, chunk_count, student_id)
    return {"doc_id": doc_id, "filename": filename, "status": "ingested", "chunk_count": chunk_count}


@router.get("/{student_id}")
async def list_documents(
    student_id: str, container: Container = Depends(get_container_dep)
) -> list[dict]:
    """List all documents for a student."""
    records = container.document_repo.list_for_student(student_id)
    return [
        {
            "doc_id": r.id,
            "filename": r.filename,
            "status": r.status,
            "chunk_count": r.chunk_count,
            "created_at": r.created_at,
        }
        for r in records
    ]


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str, container: Container = Depends(get_container_dep)
) -> dict:
    """Delete a document and its vector embeddings."""
    container.rag.delete_document(doc_id)
    container.document_repo.delete(doc_id)
    return {"doc_id": doc_id, "status": "deleted"}
