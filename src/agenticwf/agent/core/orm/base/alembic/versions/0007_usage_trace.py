"""agent_model_usage.seconds, trace_id, and an index on intent_id: the trace view of an intent
(docs/api.md, Trace).

Revision ID: 0007_usage_trace
Revises: 0006_job_trace
Create Date: 2026-10-28

The new table (agent_tool_prefs) comes from create_all. Idempotent, like every revision here.
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0007_usage_trace"
down_revision = "0006_job_trace"
branch_labels = None
depends_on = None

TABLE, INDEX = "agent_model_usage", "ix_agent_model_usage_intent_id"
_COLUMNS = (("seconds", sa.Float), ("trace_id", lambda: sa.String(32)))


def _inspect() -> tuple[set[str], set[str]]:
    inspector = sa.inspect(op.get_bind())
    if TABLE not in inspector.get_table_names():
        return set(), set()
    return {c["name"] for c in inspector.get_columns(TABLE)}, {i["name"] for i in inspector.get_indexes(TABLE)}


def upgrade() -> None:
    columns, indexes = _inspect()
    if not columns:
        return
    for column, kind in _COLUMNS:
        if column not in columns:
            op.add_column(TABLE, sa.Column(column, kind(), nullable=True))
    if INDEX not in indexes:
        op.create_index(INDEX, TABLE, ["intent_id"])


def downgrade() -> None:
    columns, indexes = _inspect()
    if INDEX in indexes:
        op.drop_index(INDEX, table_name=TABLE)
    with op.batch_alter_table(TABLE) as batch:
        for column, _ in _COLUMNS:
            if column in columns:
                batch.drop_column(column)
