"""agent_models: who owns a model and who may use it.

Revision ID: 0001_model_owner
Revises:
Create Date: 2026-10-09

Idempotent, like every revision here: `agenticwf.agent.api.db init` creates missing tables from the models first
(so a new database already has these columns) and then runs every revision.
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_model_owner"
down_revision = None
branch_labels = None
depends_on = None

def _columns() -> list[sa.Column]:
    return [sa.Column("owner_subject", sa.String(255), nullable=True),
            sa.Column("owner_org", sa.String(255), nullable=True),
            sa.Column("visibility", sa.String(16), nullable=False, server_default="all")]


def _existing() -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if "agent_models" not in inspector.get_table_names():
        return set()
    return {c["name"] for c in inspector.get_columns("agent_models")}


def upgrade() -> None:
    existing = _existing()
    if not existing:
        return   # no table yet: create_all makes it with every column
    for column in _columns():
        if column.name not in existing:
            op.add_column("agent_models", column)


def downgrade() -> None:
    existing = _existing()
    with op.batch_alter_table("agent_models") as batch:
        for column in _columns():
            if column.name in existing:
                batch.drop_column(column.name)
