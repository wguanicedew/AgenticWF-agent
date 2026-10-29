"""agent_jobs.trace: the span that queued a job, for its span to link to (tracing).

Revision ID: 0006_job_trace
Revises: 0005_shared_workers
Create Date: 2026-10-24

Idempotent, like every revision here.
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0006_job_trace"
down_revision = "0005_shared_workers"
branch_labels = None
depends_on = None


def _existing() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if "agent_jobs" not in inspector.get_table_names():
        return set()
    return {c["name"] for c in inspector.get_columns("agent_jobs")}


def upgrade() -> None:
    existing = _existing()
    if existing and "trace" not in existing:
        op.add_column("agent_jobs", sa.Column("trace", sa.String(64), nullable=True))


def downgrade() -> None:
    if "trace" in _existing():
        with op.batch_alter_table("agent_jobs") as batch:
            batch.drop_column("trace")
