"""MCP: endpoints the agent can reach and the tools they expand into (docs/mcp.md).

    endpoint    McpEndpoint: one MCP server, where it was defined (plugin, a user's plugin, the
                database), who owns it and who may use it
    client      McpClient: a small JSON-RPC client over stdio and streamable HTTP
    expand      an endpoint's tools/list → McpTool records → ToolSpecs
    manager     McpManager: connections, expansion, the handler behind every MCP tool
    catalog     McpCatalog: endpoints users add while the system runs, with their stored tools
    db_store    DatabaseMcpStore: McpCatalog's tables (agent_mcp_endpoints, agent_mcp_tools)

Import from the submodules; this package imports nothing itself.
"""
