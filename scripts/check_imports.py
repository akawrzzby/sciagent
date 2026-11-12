"""Import / syntax check for the whole backend package."""

import os
import sys
import traceback

sys.path.insert(0, os.path.expanduser("~/sciagent/backend"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODULES = [
    "app.core.config",
    "app.core.logging",
    "app.core.events",
    "app.storage.models",
    "app.storage.db",
    "app.llm.client",
    "app.llm.prompts",
    "app.llm.guardrails",
    "app.parsing.layout",
    "app.parsing.headings",
    "app.parsing.metadata",
    "app.parsing.references",
    "app.parsing.extractors",
    "app.parsing.ocr",
    "app.parsing.dedup",
    "app.parsing.pipeline",
    "app.rag.chunker",
    "app.rag.sparse",
    "app.rag.embeddings",
    "app.rag.retriever",
    "app.graph.builder",
    "app.graph.future_work",
    "app.tools.sources",
    "app.tools.registry",
    "app.tools.viz",
    "app.agents.orchestrator",
    "app.services.papers",
    "app.services.agent",
    "app.services.translation",
    "app.api.routers.health",
    "app.api.routers.papers",
    "app.api.routers.rag",
    "app.api.routers.reader",
    "app.api.routers.search",
    "app.api.routers.graph",
    "app.api.routers.writing",
    "app.api.routers.viz",
    "app.api.routers.admin",
    "app.main",
]

failed = 0
for name in MODULES:
    try:
        __import__(name)
        print("  OK      ", name)
    except Exception as exc:
        failed += 1
        print("  FAIL    ", name, "->", type(exc).__name__, str(exc)[:200])
        traceback.print_exc(limit=3)

print()
print("=" * 60)
print("modules:", len(MODULES), " failed:", failed)
sys.exit(1 if failed else 0)
