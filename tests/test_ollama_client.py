from app.ollama_client import normalize_ollama_base_url


def test_defaults_to_localhost_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    assert normalize_ollama_base_url(None) == "http://127.0.0.1:11434"


def test_rewrites_docker_ollama_hostname_to_localhost():
    assert normalize_ollama_base_url("http://ollama:11434") == "http://127.0.0.1:11434"
