"""Document upload / list / delete endpoint tests."""

from __future__ import annotations

DOC_CONTENT = (
    "Photosynthesis is the process by which green plants convert sunlight into "
    "chemical energy. The light-dependent reactions occur in the thylakoid "
    "membranes, while the Calvin cycle takes place in the stroma. "
    "Chlorophyll absorbs light primarily in the red and blue wavelengths, "
    "reflecting green light which gives plants their characteristic colour."
) * 3


class TestDocumentUpload:
    def test_upload_txt(self, client):
        resp = client.post(
            "/api/v1/documents/doc_s1",
            files={"file": ("notes.txt", DOC_CONTENT.encode(), "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ingested"
        assert data["filename"] == "notes.txt"
        assert data["chunk_count"] >= 1

    def test_list_documents(self, client):
        # First upload.
        client.post(
            "/api/v1/documents/doc_s2",
            files={"file": ("lecture.md", b"# Heading\n\nSome content.", "text/markdown")},
        )
        resp = client.get("/api/v1/documents/doc_s2")
        assert resp.status_code == 200
        records = resp.json()
        assert len(records) == 1
        assert records[0]["filename"] == "lecture.md"
        assert records[0]["status"] == "ready"

    def test_delete_document(self, client):
        upload = client.post(
            "/api/v1/documents/doc_s3",
            files={"file": ("temp.txt", b"temporary data", "text/plain")},
        )
        doc_id = upload.json()["doc_id"]
        resp = client.delete(f"/api/v1/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        # List should now be empty.
        listed = client.get("/api/v1/documents/doc_s3").json()
        assert listed == []

    def test_unsupported_file_type(self, client):
        resp = client.post(
            "/api/v1/documents/doc_s4",
            files={"file": ("image.png", b"\x89PNG", "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
