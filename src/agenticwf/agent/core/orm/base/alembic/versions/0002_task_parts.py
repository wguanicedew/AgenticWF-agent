"""agent_tasks.base: the instructions snapshot, written once; history and tool calls in rows of their own.

Revision ID: 0002_task_parts
Revises: 0001_model_owner
Create Date: 2026-10-12

The new tables (agent_step_messages, agent_tool_calls) come from create_all. Idempotent, like every
revision here. Rows written before keep working: a NULL `base` means the whole record is in `record`,
and the next save writes the new layout.
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_task_parts"
down_revision = "0001_model_owner"
branch_labels = None
depends_on = None


def _existing() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if "agent_tasks" not in inspector.get_table_names():
        return set()
    return {c["name"] for c in inspector.get_columns("agent_tasks")}


def upgrade() -> None:
    existing = _existing()
    if existing and "base" not in existing:
        op.add_column("agent_tasks", sa.Column("base", sa.LargeBinary(), nullable=True))


def downgrade() -> None:
    if "base" in _existing():
        with op.batch_alter_table("agent_tasks") as batch:
            batch.drop_column("base")
