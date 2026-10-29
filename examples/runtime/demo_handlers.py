"""
Demo tool handlers over a tiny in-memory repository. Real handlers would use a workspace,
ripgrep, a sandboxed test runner, and so on.

Each handler returns `facts` that the harness applies to working memory (files examined, tests
run), so the model does not have to remember to record them.
"""

from __future__ import annotations

import hashlib

from agenticwf.agent.runtime.tools.registry import ToolRegistry
from agenticwf.agent.runtime.tools.spec import CallContext, ToolResult

REPO = {
    "src/payment/service.py": "def calculate_total(items):\n    return str(sum(items))\n",
    "tests/test_payment.py": "def test_total():\n    assert calculate_total([10, 20]) == 30\n",
}


def _version(path: str) -> str:
    return hashlib.sha256(REPO[path].encode()).hexdigest()[:8]


def search_code(call: CallContext, args: dict) -> ToolResult:
    hits = [p for p, text in REPO.items() if args["query"] in text or args["query"] in p]
    return ToolResult(content={"matches": hits}, summary=f"{len(hits)} matches")


def read_file(call: CallContext, args: dict) -> ToolResult:
    path = args["path"]
    if path not in REPO:
        return ToolResult.error("not_found", f"No such file: {path}")
    return ToolResult(content=REPO[path], summary=path,
                      facts={"files": [{"path": path, "version": _version(path)}]})


def edit_file(call: CallContext, args: dict) -> ToolResult:
    path, old, new = args["path"], args["old_text"], args["new_text"]
    if path not in REPO:
        return ToolResult.error("not_found", f"No such file: {path}")
    if REPO[path].count(old) != 1:
        return ToolResult.error("no_unique_match", "old_text must occur exactly once")
    REPO[path] = REPO[path].replace(old, new)
    return ToolResult(content={"success": True, "path": path}, summary=f"edited {path}",
                      facts={"files": [{"path": path, "version": _version(path), "edited": True}]})


def run_tests(call: CallContext, args: dict) -> ToolResult:
    ok = "return sum(items)" in REPO["src/payment/service.py"]
    passed, failed = (182, 0) if ok else (181, 1)
    failing = [] if ok else ["tests/test_payment.py::test_total"]
    return ToolResult(content={"passed": passed, "failed": failed, "failing": failing},
                      summary=f"{passed} passed, {failed} failed",
                      facts={"tests": {"command": "pytest", "passed": passed, "failed": failed, "failing": failing}})


def web_search(call: CallContext, args: dict) -> ToolResult:
    return ToolResult(content={"results": [{"title": "Example result", "url": "https://example.com"}]})


def delete_file(call: CallContext, args: dict) -> ToolResult:
    existed = REPO.pop(args["path"], None) is not None
    return ToolResult(content={"deleted": existed},
                      facts={"files": [{"path": args["path"], "edited": True, "note": "deleted"}]})


def register(registry: ToolRegistry) -> None:
    for name, fn in [("search_code", search_code), ("read_file", read_file), ("edit_file", edit_file),
                     ("run_tests", run_tests), ("web_search", web_search), ("delete_file", delete_file)]:
        registry.register_handler(f"demo.{name}", fn)
