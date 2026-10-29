"""DatabaseMcpStore: McpCatalog's database source, on agenticwf.agent.core.orm (agent_mcp_endpoints, agent_mcp_tools).

Header values (bearer tokens, API keys) are encrypted with a SecretBox (agenticwf.agent.runtime.secrets) and
decrypted only in this process; the row keeps a hint per header for the admin view. Without a
SecretBox, endpoints without headers still work and saving headers is refused.
"""

from __future__ import annotations

import calendar
import datetime
import json

from sqlalchemy import delete, select

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import McpEndpointRow, McpToolRow
from agenticwf.agent.runtime.mcp.catalog import EndpointRecord
from agenticwf.agent.runtime.mcp.endpoint import McpEndpoint
from agenticwf.agent.runtime.mcp.expand import McpTool
from agenticwf.agent.runtime.secrets import SecretBox, SecretsUnavailable, hint
from agenticwf.agent.runtime.tools.spec import Access

LOG = get_logger(__name__)


class DatabaseMcpStore:
    def __init__(self, box: SecretBox | None):
        self.box = box

    def list_endpoints(self) -> list[EndpointRecord]:
        records = []
        for row in _endpoint_rows():
            headers, unreadable = {}, False
            if row["headers_encrypted"]:
                if self.box is None:
                    unreadable = True
                    LOG.error("MCP endpoint %s has stored headers but no secrets key is configured", row["id"])
                else:
                    try:
                        headers = json.loads(self.box.decrypt(row["headers_encrypted"]))
                    except (SecretsUnavailable, ValueError):
                        unreadable = True
                        LOG.error("MCP endpoint %s: its headers cannot be decrypted (secrets key changed?)", row["id"])
            settings = row["settings"] or {}
            endpoint = McpEndpoint(
                name=row["name"], transport=row["transport"], url=row["url"], headers=headers,
                tools=tuple(settings.get("tools") or ()), effects=settings.get("effects"),
                approval=settings.get("approval"), timeout=float(settings.get("timeout", 60)),
                source="database", owner=(row["owner_subject"], row["owner_org"]), visibility=row["visibility"],
                access=Access.from_mapping(settings.get("access")), label=row["label"] or "",
                description=row["description"] or "", enabled=bool(row["enabled"]))
            records.append(EndpointRecord(
                endpoint, status=row["status"], error=row["error"] or "", problems=tuple(row["problems"] or ()),
                server=row["server"], server_version=row["server_version"],
                checked_at=float(calendar.timegm(row["checked_at"].utctimetuple())) if row["checked_at"] else None,
                header_hints=dict(row["header_hints"] or {}), headers_unreadable=unreadable))
        return records

    def list_tools(self) -> list[tuple[str, McpTool]]:
        """(endpoint id, tool) for every stored tool."""
        return [(row["endpoint_id"], McpTool(
            name=row["name"], mcp_name=row["mcp_name"], endpoint=f"database:{row['endpoint_id']}",
            summary=row["summary"], description=row["description"], input_schema=row["input_schema"],
            effects=row["effects"], approval=row["approval"], idempotent=bool(row["idempotent"]),
            read_only=bool(row["read_only"]), enabled=bool(row["enabled"]),
            owner=(row["owner_subject"], row["owner_org"]), visibility=row["visibility"] or ""))
            for row in _tool_rows()]

    def save_endpoint(self, record: EndpointRecord, tools: list[McpTool] | None, keep_headers: bool = False) -> None:
        """`keep_headers`: leave the stored (encrypted) headers as they are; they are not rewritten
        from what this process could decrypt."""
        if keep_headers:
            _save_endpoint(record, None, None, tools, keep_headers=True)
            return
        headers = record.endpoint.headers
        if headers and self.box is None:
            raise PermissionError("no secrets key is configured (AGENT_SECRETS_KEY_FILE): header values "
                                  "cannot be stored in the database")
        encrypted = self.box.encrypt(json.dumps(headers)) if headers else None
        _save_endpoint(record, encrypted, {k: hint(v) for k, v in headers.items()}, tools)

    def save_tool(self, endpoint_id: str, tool: McpTool) -> None:
        _save_tools(endpoint_id, [tool], replace_all=False)

    def delete_endpoint(self, endpoint_id: str) -> bool:
        return _delete_endpoint(endpoint_id)


def _as_dicts(rows, table) -> list[dict]:
    return [{c.name: getattr(r, c.name) for c in table.columns} for r in rows]


@read_session
def _endpoint_rows(*, session=None) -> list[dict]:
    return _as_dicts(session.execute(select(McpEndpointRow).order_by(McpEndpointRow.id)).scalars(),
                     McpEndpointRow.__table__)


@read_session
def _tool_rows(*, session=None) -> list[dict]:
    return _as_dicts(session.execute(select(McpToolRow).order_by(McpToolRow.id)).scalars(), McpToolRow.__table__)


@transactional_session
def _save_endpoint(record: EndpointRecord, encrypted: str | None, hints: dict[str, str] | None,
                   tools: list[McpTool] | None, keep_headers: bool = False, *, session=None) -> None:
    e = record.endpoint
    row = session.get(McpEndpointRow, e.id)
    if row is not None and (row.owner_subject, row.owner_org) != tuple(e.owner):
        raise PermissionError(f"MCP endpoint {e.id!r} belongs to someone else")   # checked again here: no race
    row = row or McpEndpointRow(id=e.id)
    row.name, (row.owner_subject, row.owner_org) = e.name, e.owner
    row.label, row.description, row.transport, row.url = e.label, e.description, e.transport, e.url
    if not keep_headers or row.headers_encrypted is None and row.header_hints is None:
        row.headers_encrypted, row.header_hints = encrypted, hints or {}
    row.settings = {"tools": list(e.tools), "effects": e.effects, "approval": e.approval, "timeout": e.timeout,
                    "access": e.access.to_dict()}
    row.visibility, row.enabled = e.effective_visibility, int(e.enabled)
    row.status, row.error, row.problems = record.status, record.error, list(record.problems)
    row.server, row.server_version = record.server, record.server_version
    row.checked_at = datetime.datetime.fromtimestamp(record.checked_at, datetime.timezone.utc).replace(tzinfo=None) if record.checked_at else None
    session.merge(row)
    session.flush()
    if tools is not None:
        _save_tools(e.id, tools, replace_all=True, session=session)


@transactional_session
def _save_tools(endpoint_id: str, tools: list[McpTool], replace_all: bool, *, session=None) -> None:
    """Tool ids are `<scope>:<name>`, the endpoint's scope with the tool's name (<endpoint>__<tool>)."""
    if replace_all:
        session.execute(delete(McpToolRow).where(McpToolRow.endpoint_id == endpoint_id))
    scope = endpoint_id.rpartition(":")[0]
    for tool in tools:
        tool_id = f"{scope}:{tool.name}"
        row = session.get(McpToolRow, tool_id)
        if row is not None and row.endpoint_id != endpoint_id:   # cannot happen (no "__" in names): never take one over
            LOG.error("MCP tool %s of %s not saved: it belongs to %s", tool_id, endpoint_id, row.endpoint_id)
            continue
        row = row or McpToolRow(id=tool_id)
        row.endpoint_id, row.name, row.mcp_name = endpoint_id, tool.name, tool.mcp_name
        row.owner_subject, row.owner_org = tool.owner
        row.summary, row.description, row.input_schema = tool.summary, tool.description, tool.input_schema
        row.effects, row.approval = tool.effects, tool.approval
        row.idempotent, row.read_only, row.enabled = int(tool.idempotent), int(tool.read_only), int(tool.enabled)
        row.visibility = tool.visibility
        session.merge(row)


@transactional_session
def _delete_endpoint(endpoint_id: str, *, session=None) -> bool:
    session.execute(delete(McpToolRow).where(McpToolRow.endpoint_id == endpoint_id))
    row = session.get(McpEndpointRow, endpoint_id)
    if row is None:
        return False
    session.delete(row)
    return True
