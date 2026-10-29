"""Programmatic Alembic access (used by `python -m agenticwf.agent.api.db init` through build_database()).

The database URL comes from the ``[database] default`` option of the core configuration, so there is
no separate alembic.ini to keep in step.
"""

from alembic import command
from alembic.config import Config

from agenticwf.agent.core.common.config import config_get
from agenticwf.agent.core.orm.base.session import DATABASE_SECTION, DEFAULT_SCHEMA_NAME

SCRIPT_LOCATION = "agenticwf.agent.core.orm.base:alembic"


def alembic_config(url: str | None = None, **kwargs) -> Config:
    cfg = Config(**kwargs)
    cfg.set_main_option("script_location", SCRIPT_LOCATION)
    # ConfigParser interpolation: a literal % (URL-encoded password) must be doubled.
    url = url or config_get(DATABASE_SECTION, "default")
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    if DEFAULT_SCHEMA_NAME:
        cfg.set_main_option("version_table_schema", DEFAULT_SCHEMA_NAME)
    return cfg


def upgrade(revision: str = "head", url: str | None = None) -> None:
    command.upgrade(alembic_config(url), revision)


def stamp(revision: str = "head", url: str | None = None) -> None:
    command.stamp(alembic_config(url), revision)


def at_head(url: str | None = None) -> bool:
    """The database is at this code's newest revision (a worker process waits for this: the API
    process migrates, and a table that already exists may still lack new columns)."""
    import sqlalchemy as sa
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    from agenticwf.agent.core.orm.base.session import get_engine

    head = ScriptDirectory.from_config(alembic_config(url)).get_current_head()
    engine = sa.create_engine(url) if url else get_engine()
    with engine.connect() as conn:
        opts = {"version_table_schema": DEFAULT_SCHEMA_NAME} if DEFAULT_SCHEMA_NAME else {}
        return MigrationContext.configure(conn, opts=opts).get_current_revision() == head
