"""agent_tasks.cancel: a cancel requested in one process, seen by a worker in another.

Revision ID: 0005_shared_workers
Revises: 0004_memories
Create Date: 2026-10-13

The new tables (agent_jobs, agent_events) come from create_all. Idempotent, like every revision here.
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0005_shared_workers"
down_revision = "0004_memories"
branch_labels = None
depends_on = None


def _existing() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if "agent_tasks" not in inspector.get_table_names():
        return set()
    return {c["name"] for c in inspector.get_columns("agent_tasks")}


def upgrade() -> None:
    existing = _existing()
    if existing and "cancel" not in existing:
        op.add_column("agent_tasks", sa.Column("cancel", sa.Integer(), nullable=True))


def downgrade() -> None:
    if "cancel" in _existing():
        with op.batch_alter_table("agent_tasks") as batch:
            batch.drop_column("cancel")
