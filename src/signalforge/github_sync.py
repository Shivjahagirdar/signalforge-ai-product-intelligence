"""GitHub Issues connector: public repos, no credentials.

Pagination via the ``Link`` header, bounded retries with exponential backoff for 5xx/network errors, explicit
rate-limit handling (``X-RateLimit-Remaining`` / ``Reset`` / ``Retry-After``), incremental sync with ``since`` (cursor =
max ``updated_at`` seen), idempotent upsert into the workspace store, and pull-request exclusion.

Source limitations (shown in the UI): unauthenticated API = 60 requests/hour per IP (shared on the hosted app);
issue authors are developers/users, not paying accounts; reactions are a demand proxy, not revenue.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

from .datasets import adapt_github_issues
from .ingestion import IngestionError

API = "https://api.github.com"
REPO_RX = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")
MAX_PAGES = 5
PER_PAGE = 100


@dataclass
class HttpResponse:
    status: int
    headers: dict
    body: bytes


Http = Callable[[str], HttpResponse]


def urllib_http(url: str, timeout: int = 20) -> HttpResponse:
    if not url.startswith(API + "/"):
        raise IngestionError("Only api.github.com is allowed for this connector.")
    req = urllib.request.Request(url, headers={"User-Agent": "SignalForge-AI/3.0", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - host pinned above
            return HttpResponse(r.status, {k.lower(): v for k, v in r.headers.items()}, r.read(8 * 1024 * 1024))
    except urllib.error.HTTPError as e:
        return HttpResponse(e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, e.read() if hasattr(e, "read") else b"")


@dataclass
class SyncReport:
    repo: str
    pages: int = 0
    requests: int = 0
    retries: int = 0
    fetched: int = 0
    pull_requests_skipped: int = 0
    cursor: str | None = None
    incremental: bool = False
    rate_remaining: int | None = None
    status: str = "ok"
    warnings: list = field(default_factory=list)


def next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        m = re.search(r'<([^>]+)>\s*;\s*rel="next"', part)
        if m:
            return m.group(1)
    return None


def _get(http: Http, url: str, rep: SyncReport, sleep: Callable[[float], None], max_retries: int = 3) -> HttpResponse:
    delay = 1.0
    for attempt in range(max_retries + 1):
        rep.requests += 1
        try:
            r = http(url)
        except IngestionError:
            raise
        except Exception as exc:  # noqa: BLE001 - network errors are retried, then surfaced safely
            r = None
            err = type(exc).__name__
        else:
            err = None
            rem = r.headers.get("x-ratelimit-remaining")
            if rem is not None and str(rem).isdigit():
                rep.rate_remaining = int(rem)
            if r.status in (403, 429) and (rep.rate_remaining == 0 or "retry-after" in r.headers):
                reset = r.headers.get("x-ratelimit-reset")
                wait = f" Resets at epoch {reset}." if reset else ""
                raise IngestionError(f"GitHub rate limit reached (60 requests/hour unauthenticated).{wait} "
                                     "Already-synced issues stay in your workspace; try again later.")
            if r.status < 500:
                return r
        if attempt < max_retries:
            rep.retries += 1
            sleep(delay)
            delay *= 2
    raise IngestionError(f"GitHub didn't respond after {max_retries + 1} attempts ({err or 'server error'}). Try again shortly.")


def fetch_issues(repo: str, since: str | None = None, http: Http | None = None, sleep: Callable[[float], None] = time.sleep,
                 max_pages: int = MAX_PAGES) -> tuple[list[dict], SyncReport]:
    if not REPO_RX.fullmatch(repo or ""):
        raise IngestionError("Enter a public repository as owner/name, e.g. streamlit/streamlit.")
    http = http or urllib_http
    rep = SyncReport(repo=repo, incremental=bool(since))
    url = f"{API}/repos/{repo}/issues?state=all&per_page={PER_PAGE}&sort=updated&direction=asc"
    if since:
        url += f"&since={since}"
    items: list[dict] = []
    while url and rep.pages < max_pages:
        r = _get(http, url, rep, sleep)
        if r.status == 404:
            raise IngestionError("Repository not found or private. Only public repositories work without credentials.")
        if r.status >= 400:
            raise IngestionError(f"GitHub API error {r.status}.")
        batch = json.loads(r.body or b"[]")
        if not isinstance(batch, list):
            raise IngestionError(f"GitHub API: {batch.get('message', 'unexpected response')}")
        rep.pages += 1
        for it in batch:
            if "pull_request" in it:
                rep.pull_requests_skipped += 1
                continue
            items.append(it)
        url = next_link(r.headers.get("link"))
    if url:
        rep.warnings.append(f"Stopped after {max_pages} pages to protect the shared rate limit; sync again to continue.")
    rep.fetched = len(items)
    stamps = [it.get("updated_at") for it in items if it.get("updated_at")]
    rep.cursor = max(stamps) if stamps else since
    return items, rep


def to_records(items: list[dict], repo: str) -> list[dict]:
    if not items:
        return []
    df = adapt_github_issues(items, repo)
    upd = {f"{repo}#{it['number']}": it.get("updated_at", "") for it in items if "pull_request" not in it}
    urls = {f"{repo}#{it['number']}": it.get("html_url", "") for it in items if "pull_request" not in it}
    recs = df.to_dict("records")
    for r in recs:
        r["updated_at"] = upd.get(r["ticket_id"], "")
        r["source_url"] = urls.get(r["ticket_id"], "")
    return recs


def sync_repo(store, ws: str, repo: str, http: Http | None = None, sleep: Callable[[float], None] = time.sleep,
              full: bool = False) -> SyncReport:
    """Incremental, idempotent sync of a public repo into a workspace."""
    key = f"github:{repo}"
    prev = None if full else (store.get_sync(ws, key) or {}).get("cursor")
    try:
        items, rep = fetch_issues(repo, prev, http, sleep)
    except IngestionError as e:
        store.set_sync(ws, key, prev, f"error: {e}"[:200], 0)
        raise
    counts = store.upsert_feedback(ws, to_records(items, repo), key)
    rep.status = f"ok: {counts['inserted']} new, {counts['updated']} updated, {counts['unchanged']} unchanged"
    store.set_sync(ws, key, rep.cursor, rep.status, rep.fetched)
    store.audit(ws, "github_sync", key, {"cursor": prev}, {"cursor": rep.cursor, **counts, "pages": rep.pages}, "connector")
    return rep
