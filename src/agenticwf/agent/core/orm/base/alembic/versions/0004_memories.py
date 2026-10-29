"""agent_tasks.memory: whether a finished task's consolidation into long-term memory is still to run;
agent_chat_sessions.summary, summary_keys: a long chat's summary, kept with it.

Revision ID: 0004_memories
Revises: 0003_scoped_task_types
Create Date: 2026-10-12

The new table (agent_memories) comes from create_all. Idempotent, like every revision here.
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_memories"
down_revision = "0003_scoped_task_types"
branch_labels = None
depends_on = None


def _existing(table: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if table not in inspector.get_table_names():
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


_COLUMNS = (("agent_tasks", "memory", lambda: sa.String(16)),
            ("agent_chat_sessions", "summary", sa.Text),
            ("agent_chat_sessions", "summary_keys", sa.JSON))


def upgrade() -> None:
    for table, column, kind in _COLUMNS:
        existing = _existing(table)
        if existing and column not in existing:
            op.add_column(table, sa.Column(column, kind(), nullable=True))


def downgrade() -> None:
    for table, column, _ in _COLUMNS:
        if column in _existing(table):
            with op.batch_alter_table(table) as batch:
                batch.drop_column(column)
