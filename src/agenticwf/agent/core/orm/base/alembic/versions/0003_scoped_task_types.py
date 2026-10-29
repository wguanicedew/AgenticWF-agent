"""agent_tasks.task_type: room for scoped references (`<scope>:<name>`, up to 320 characters).

Revision ID: 0003_scoped_task_types
Revises: 0002_task_parts
Create Date: 2026-10-12

Idempotent, like every revision here. SQLite does not enforce VARCHAR lengths: nothing to do there.
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0003_scoped_task_types"
down_revision = "0002_task_parts"
branch_labels = None
depends_on = None


def _length() -> int | None:
    inspector = sa.inspect(op.get_bind())
    if "agent_tasks" not in inspector.get_table_names():
        return None
    column = next((c for c in inspector.get_columns("agent_tasks") if c["name"] == "task_type"), None)
    return getattr(column["type"], "length", None) if column else None


def upgrade() -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    length = _length()
    if length is not None and length < 320:
        op.alter_column("agent_tasks", "task_type", type_=sa.String(320), existing_nullable=False)


def downgrade() -> None:
    pass   # narrowing could cut stored references
