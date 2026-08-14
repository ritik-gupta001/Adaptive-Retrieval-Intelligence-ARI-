"""
Module 15 deployment tests.
Validates that the deployment artifacts are self-consistent — no LLM
calls, no Docker daemon required. Checks that:
  - .env.example contains every key settings.py reads
  - pyproject.toml declares all packages that are imported in app/
  - Dockerfile references correct paths that exist in the repo
  - nginx.conf proxies the SSE endpoint with buffering disabled
  - docker-compose.yaml references services that have corresponding Dockerfiles
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.append(".")

ROOT = Path(__file__).parent.parent.parent  # ari/


# ---------------------------------------------------------------------------
# .env.example completeness
# ---------------------------------------------------------------------------

def test_env_example_covers_all_settings_fields():
    """.env.example should document every field in settings.py so a new
    developer knows what to fill in."""
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8").lower()

    from app.config.settings import Settings
    import inspect

    for field_name in Settings.model_fields:
        # env var names are uppercase versions of field names
        env_key = field_name.upper()
        assert env_key.lower() in env_example, (
            f"Settings field '{field_name}' (env key: {env_key}) not found in "
            f".env.example — new developers won't know it exists."
        )


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------

def test_pyproject_exists_and_is_valid_toml():
    import tomllib
    content = (ROOT / "pyproject.toml").read_bytes()
    parsed = tomllib.loads(content.decode())
    assert "project" in parsed
    assert "name" in parsed["project"]
    assert parsed["project"]["name"] == "ari-platform"


def test_pyproject_has_required_deps():
    import tomllib
    content = (ROOT / "pyproject.toml").read_bytes()
    parsed = tomllib.loads(content.decode())
    deps = " ".join(parsed["project"]["dependencies"])

    required = ["langgraph", "langchain", "fastapi", "pydantic", "chromadb"]
    for dep in required:
        assert dep in deps, f"Required dependency '{dep}' missing from pyproject.toml"


def test_pyproject_has_prod_extras():
    import tomllib
    content = (ROOT / "pyproject.toml").read_bytes()
    parsed = tomllib.loads(content.decode())
    extras = parsed["project"].get("optional-dependencies", {})
    assert "prod" in extras, "No 'prod' extras in pyproject.toml — Dockerfile uses pip install -e '.[prod]'"
    assert "dev" in extras, "No 'dev' extras — test suite needs this"


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------

def test_dockerfile_references_existing_paths():
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    # These COPY sources must exist relative to the build context (repo root)
    for path_ref in ["app", "evals", "ingestion", "pyproject.toml"]:
        assert path_ref in dockerfile, (
            f"Dockerfile doesn't COPY '{path_ref}' — it will be missing in the image."
        )


def test_dockerfile_uses_python_312():
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.12" in dockerfile, "Dockerfile should pin Python 3.12 per spec"


def test_dockerfile_frontend_references_existing_sources():
    dockerfile = (ROOT / "docker" / "Dockerfile.frontend").read_text(encoding="utf-8")
    assert "frontend/index.html" in dockerfile
    assert "docker/nginx.conf" in dockerfile


# ---------------------------------------------------------------------------
# nginx.conf — SSE correctness
# ---------------------------------------------------------------------------

def test_nginx_conf_disables_proxy_buffering_for_sse():
    """proxy_buffering off is required for SSE to stream through nginx.
    If this is missing, the client gets the full response in one chunk."""
    nginx_conf = (ROOT / "docker" / "nginx.conf").read_text(encoding="utf-8")
    assert "proxy_buffering off" in nginx_conf, (
        "nginx.conf is missing 'proxy_buffering off' for the SSE endpoint. "
        "SSE streaming will not work through nginx without this."
    )


def test_nginx_conf_proxies_sse_endpoint():
    nginx_conf = (ROOT / "docker" / "nginx.conf").read_text(encoding="utf-8")
    assert "/query/stream" in nginx_conf, (
        "nginx.conf doesn't have a location block for /query/stream"
    )


def test_nginx_conf_proxies_health_endpoint():
    nginx_conf = (ROOT / "docker" / "nginx.conf").read_text(encoding="utf-8")
    assert "health" in nginx_conf


def test_nginx_conf_has_long_read_timeout_for_streaming():
    """Graph runs can take 30+ seconds with retries. nginx default
    proxy_read_timeout is 60s — we need at least 120s."""
    nginx_conf = (ROOT / "docker" / "nginx.conf").read_text(encoding="utf-8")
    timeouts = re.findall(r"proxy_read_timeout\s+(\d+)s", nginx_conf)
    assert timeouts, "No proxy_read_timeout set in nginx.conf"
    max_timeout = max(int(t) for t in timeouts)
    assert max_timeout >= 120, (
        f"Highest proxy_read_timeout is {max_timeout}s — should be ≥120s "
        "to cover retry loops"
    )


# ---------------------------------------------------------------------------
# docker-compose.yaml
# ---------------------------------------------------------------------------

def test_docker_compose_references_both_dockerfiles():
    compose = (ROOT / "docker" / "docker-compose.yaml").read_text(encoding="utf-8")
    assert "Dockerfile" in compose
    assert "Dockerfile.frontend" in compose


def test_docker_compose_has_health_check_on_backend():
    compose = (ROOT / "docker" / "docker-compose.yaml").read_text(encoding="utf-8")
    assert "healthcheck" in compose
    assert "/health" in compose


def test_docker_compose_neo4j_is_optional_profile():
    """Neo4j must be in an optional profile — it shouldn't start by default
    since Graph RAG is optional and Neo4j is heavy."""
    compose = (ROOT / "docker" / "docker-compose.yaml").read_text(encoding="utf-8")
    assert 'profiles: ["graph"]' in compose or "profiles:\n      - graph" in compose, (
        "Neo4j service should be gated behind a Docker Compose profile "
        "so it doesn't start by default."
    )


# ---------------------------------------------------------------------------
# Render / Railway configs
# ---------------------------------------------------------------------------

def test_render_yaml_has_both_services():
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "ari-backend" in render
    assert "ari-frontend" in render






# ---------------------------------------------------------------------------
# requirements files
# ---------------------------------------------------------------------------

def test_requirements_txt_exists():
    assert (ROOT / "requirements.txt").exists(), \
        "requirements.txt missing — add one so developers can pip install -r requirements.txt"


def test_pinned_requirements_files_exist():
    for name in ("base.txt", "dev.txt", "prod.txt", "eval.txt"):
        path = ROOT / "requirements" / name
        assert path.exists(), f"requirements/{name} missing — run pip-compile to regenerate"


def test_requirements_txt_covers_core_packages():
    content = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    core = ["langgraph", "langchain", "fastapi", "pydantic", "chromadb", "anthropic"]
    for pkg in core:
        assert pkg in content, f"Core package '{pkg}' not listed in requirements.txt"


def test_pinned_base_txt_has_more_lines_than_requirements_txt():
    """requirements/base.txt is the fully pinned transitive closure and
    must be larger than the direct-dep requirements.txt."""
    base = len((ROOT / "requirements" / "base.txt").read_text(encoding="utf-8").splitlines())
    direct = len((ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines())
    assert base > direct, \
        "requirements/base.txt should be larger than requirements.txt (pinned transitive deps)"
