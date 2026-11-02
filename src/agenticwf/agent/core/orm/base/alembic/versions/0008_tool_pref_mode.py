"""agent_tool_prefs.mode: a tool, plugin or MCP endpoint a user set to ask before each call, besides
off (docs/admin-page.md, Tools).

Revision ID: 0008_tool_pref_mode
Revises: 0007_usage_trace
Create Date: 2026-10-31

Rows from before have no mode: they stay off. Idempotent, like every revision here.
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0008_tool_pref_mode"
down_revision = "0007_usage_trace"
branch_labels = None
depends_on = None


def _existing() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if "agent_tool_prefs" not in inspector.get_table_names():
        return set()
    return {c["name"] for c in inspector.get_columns("agent_tool_prefs")}


def upgrade() -> None:
    existing = _existing()
    if existing and "mode" not in existing:
        op.add_column("agent_tool_prefs", sa.Column("mode", sa.String(8), nullable=True))


def downgrade() -> None:
    if "mode" in _existing():
        op.execute("DELETE FROM agent_tool_prefs WHERE mode = 'ask'")   # without modes, a row means off
        with op.batch_alter_table("agent_tool_prefs") as batch:
            batch.drop_column("mode")
