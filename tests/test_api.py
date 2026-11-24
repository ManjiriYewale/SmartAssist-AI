import pytest
from fastapi.testclient import TestClient

from backend.llm.model import set_llm_override
from backend.main import app
from tests.fakes import make_llm


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def up(client, samples, name):
    return client.post("/documents/upload", files={"file": (name, (samples / name).read_bytes())})


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["llm_configured"] is False


def test_upload_list_get_delete(client, samples):
    r = up(client, samples, "project_notes.txt")
    assert r.status_code == 200 and r.json()["status"] == "processed" and r.json()["duplicate"] is False
    doc_id = r.json()["id"]
    assert up(client, samples, "project_notes.txt").json()["duplicate"] is True
    assert len(client.get("/documents").json()) == 1
    assert client.get(f"/documents/{doc_id}").json()["filename"] == "project_notes.txt"
    assert client.get("/documents?search=zzz").json() == []
    assert client.delete(f"/documents/{doc_id}").status_code == 200
    assert client.get(f"/documents/{doc_id}").status_code == 404


def test_upload_validation_errors(client):
    r = client.post("/documents/upload", files={"file": ("evil.exe", b"MZ")})
    assert r.status_code == 400 and "Unsupported" in r.json()["detail"]
    r = client.post("/documents/upload", files={"file": ("fake.pdf", b"nope")})
    assert r.status_code == 400


def test_search_endpoint(client, samples):
    up(client, samples, "employee_handbook.pdf")
    r = client.post("/search", json={"query": "paternity leave days", "top_k": 3}).json()
    assert r["chunks"] and r["chunks"][0]["filename"] == "employee_handbook.pdf"
    assert client.post("/search", json={"query": "", "top_k": 3}).status_code == 422


def test_chat_endpoint_and_history(client, samples):
    up(client, samples, "employee_handbook.pdf")
    set_llm_override(make_llm(intent="document_qa", answer="24 days [1]", plan=[[("document_search", {"query": "annual leave"})]]))
    r = client.post("/chat", json={"message": "How many leave days?"}).json()
    assert r["answer"] == "24 days [1]" and r["sources"][0]["page"] == 1 and r["tools"][0]["name"] == "document_search"
    msgs = client.get(f"/conversations/{r['conversation_id']}/messages").json()
    assert len(msgs) == 2
    assert client.get("/conversations").json()[0]["message_count"] == 2
    assert client.post("/chat", json={"message": "x", "conversation_id": 999}).status_code == 404


def test_chat_without_api_key_is_a_clean_message(client):
    r = client.post("/chat", json={"message": "hello"})
    assert r.status_code == 200 and "GROQ_API_KEY" in r.json()["answer"]


def test_tools_endpoints(client, samples):
    assert len(client.get("/tools").json()) == 6
    r = client.post("/tools/calculator/run", json={"arguments": {"operation": "add_gst", "amount": 100}}).json()
    assert r["status"] == "success" and r["output"]["total_amount"] == 118.0
    bad = client.post("/tools/calculator/run", json={"arguments": {"operation": "nope"}}).json()
    assert bad["status"] == "error"
    assert client.post("/tools/ghost/run", json={"arguments": {}}).status_code == 422
    assert client.get("/stats").json()["tools"][0]["tool"] == "calculator"


def test_api_key_protection(client, monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "s3cret")
    from backend.config.settings import reload_settings
    reload_settings()
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/health").status_code == 200            # health stays public
