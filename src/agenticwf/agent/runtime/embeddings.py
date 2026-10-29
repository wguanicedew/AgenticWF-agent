"""
Embeddings for matching by meaning: a request for a "changelog" finds the `release-notes` skill,
though they share no word (docs/agent-runtime.md, Selecting tools for a task). One index serves
tool and skill selection and long-term memory recall.

    [embeddings]
    provider = voyage            voyage | openai | ollama | vllm (any OpenAI-compatible /embeddings);
                                 empty: off, matching is BM25 alone
    model = voyage-3.5
    base_url =                   the provider's default; required for vllm
    api_key_env = VOYAGE_API_KEY the environment variable holding the key; the provider's default
    weight = 0.6                 share of the similarity in the combined score (BM25 the rest)
    min_similarity = 0.3         a candidate sharing no word with the query needs at least this
    timeout = 5

The EmbeddingIndex keeps the vector of every text it has embedded (keyed by the text's hash, so
a changed description is embedded again) and of recent queries. A provider that fails or is slow
is skipped for a minute; matching then falls back to BM25, never fails.

Only operators configure this (agent.cfg), so its host is trusted like a configured model's.

The event loop never waits for the provider: on it, similarities() uses cached vectors only and
fetches what is missing in the background (that one search goes by words); the harness fetches a
request's vectors in a thread before it selects tools or recalls memories (prepare()), so the
usual case is matched by meaning. Off the loop (a thread), similarities() waits for the provider.

TODO: vectors in PostgreSQL with pgvector, so a restart does not embed the catalogue again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import threading
import time
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

from agenticwf.agent.core.common.logging import get_logger

LOG = get_logger(__name__)

# provider -> (base URL, environment variable with the key)
PRESETS = {
    "voyage": ("https://api.voyageai.com/v1", "VOYAGE_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "vllm": (None, None),
}


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str = ""
    model: str = ""
    base_url: str | None = None
    api_key_env: str | None = None
    weight: float = 0.6
    min_similarity: float = 0.3
    timeout: float = 5.0
    batch: int = 64

    @property
    def enabled(self) -> bool:
        return bool(self.provider)

    def check(self) -> "EmbeddingSettings":
        if not self.enabled:
            return self
        if self.provider not in PRESETS:
            raise ValueError(f"[embeddings] provider must be one of {', '.join(PRESETS)}, not {self.provider!r}")
        if not self.model:
            raise ValueError("[embeddings] model is required")
        if not (self.base_url or PRESETS[self.provider][0]):
            raise ValueError(f"[embeddings] base_url is required for {self.provider}")
        if not 0 <= self.weight <= 1:
            raise ValueError("[embeddings] weight must be between 0 and 1")
        return self


class Embedder(Protocol):
    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        """One vector per text. `kind`: query (what is searched for) or document (what is searched)."""


class HttpEmbedder:
    """The `/embeddings` endpoint that Voyage, OpenAI, Ollama and vLLM share."""

    def __init__(self, settings: EmbeddingSettings):
        self.settings = settings.check()
        base, key_env = PRESETS[settings.provider]
        self.url = (settings.base_url or base).rstrip("/") + "/embeddings"
        env = settings.api_key_env or key_env
        self.api_key = os.environ.get(env) if env else None
        if env and not self.api_key:
            LOG.warning("embeddings: %s is not set; requests to %s are sent without a key", env, self.url)

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        body = {"model": self.settings.model, "input": texts}
        if self.settings.provider == "voyage":
            body["input_type"] = kind
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.settings.timeout) as response:   # noqa: S310 (operator's URL)
            data = json.loads(response.read())
        rows = sorted(data["data"], key=lambda r: r.get("index", 0))
        if len(rows) != len(texts):
            raise ValueError(f"asked for {len(texts)} embeddings, got {len(rows)}")
        return [row["embedding"] for row in rows]


class EmbeddingIndex:
    def __init__(self, embedder: Embedder, settings: EmbeddingSettings, max_documents: int = 50_000,
                 max_queries: int = 1_000, backoff: float = 60.0):
        self.embedder = embedder
        self.settings = settings
        self.max_documents = max_documents
        self.max_queries = max_queries
        self.backoff = backoff
        self._documents: OrderedDict[str, list[float]] = OrderedDict()
        self._queries: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()
        self._down_until = 0.0
        self._fetching: set[str] = set()   # queries being fetched in the background

    def similarities(self, query: str, texts: list[str], wait: bool | None = None) -> list[float] | None:
        """Cosine similarity of `query` to each text, or None when the provider cannot answer now.
        `wait` (default: not on an event loop): call the provider for what is not cached; without,
        only cached vectors are used, and what is missing is fetched in the background."""
        if time.monotonic() < self._down_until:
            return None
        if wait is None:
            wait = not on_event_loop()
        if not wait and not self._cached(query, texts):
            self._fetch_later(query, texts)
            return None
        try:
            query_vector = self._query(query)
            vectors = self._vectors(texts)
        except Exception as exc:   # network, HTTP or format errors: BM25 alone for a while
            LOG.warning("embeddings unavailable for %.0f s, matching by words only: %s", self.backoff, exc)
            self._down_until = time.monotonic() + self.backoff
            return None
        return [_cosine(query_vector, v) for v in vectors]

    def prepare(self, query: str, texts: list[str]) -> None:
        """Fetch what `query` and `texts` need (in a thread: it waits for the provider)."""
        if time.monotonic() < self._down_until or self._cached(query, texts):
            return
        try:
            self._query(query)
            self._vectors(texts)
        except Exception as exc:
            LOG.warning("embeddings unavailable for %.0f s, matching by words only: %s", self.backoff, exc)
            self._down_until = time.monotonic() + self.backoff

    def _cached(self, query: str, texts: list[str]) -> bool:
        with self._lock:
            return _key(query) in self._queries and all(_key(t) in self._documents for t in texts)

    def _fetch_later(self, query: str, texts: list[str]) -> None:
        key = _key(query)
        with self._lock:
            if key in self._fetching:
                return
            self._fetching.add(key)

        def run() -> None:
            try:
                self.prepare(query, texts)
            finally:
                with self._lock:
                    self._fetching.discard(key)
        threading.Thread(target=run, name="embeddings-fetch", daemon=True).start()

    def warm(self, texts: list[str]) -> None:
        """Embed these in the background (the shared catalogue, after a load)."""
        def run() -> None:
            try:
                self._vectors(texts)
            except Exception as exc:
                LOG.warning("embeddings: the catalogue could not be embedded yet: %s", exc)
        threading.Thread(target=run, name="embeddings-warm", daemon=True).start()

    # --- caches --------------------------------------------------------------------------------
    def _query(self, query: str) -> list[float]:
        key = _key(query)
        with self._lock:
            if key in self._queries:
                self._queries.move_to_end(key)
                return self._queries[key]
        vector = self.embedder.embed([query], "query")[0]
        with self._lock:
            self._queries[key] = vector
            while len(self._queries) > self.max_queries:
                self._queries.popitem(last=False)
        return vector

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        keys = [_key(t) for t in texts]
        with self._lock:
            missing = list(dict.fromkeys(k for k in keys if k not in self._documents))
        by_key = dict(zip(keys, texts))
        for start in range(0, len(missing), self.settings.batch):
            chunk = missing[start:start + self.settings.batch]
            vectors = self.embedder.embed([by_key[k] for k in chunk], "document")
            with self._lock:
                self._documents.update(zip(chunk, vectors))
        with self._lock:
            for key in keys:
                self._documents.move_to_end(key)
            found = [self._documents[k] for k in keys]
            while len(self._documents) > self.max_documents:
                self._documents.popitem(last=False)
        return found


def on_event_loop() -> bool:
    """Whether this code runs on an event loop's thread (where waiting would stop every task)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
