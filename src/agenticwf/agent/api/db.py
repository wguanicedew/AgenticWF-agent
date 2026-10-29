"""Create or upgrade the agent's database schema.

    python -m agenticwf.agent.api.db init      create missing tables, then stamp/upgrade Alembic to head

The database URL comes from the ``[database] default`` option of the core configuration
(AGENTICWF_CONFIG). Alembic uses agenticwf.agent.core.orm.base.alembic with that same URL, so there is no
separate alembic.ini to keep in step.
"""

from __future__ import annotations

import argparse

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.core.orm.base import utils as db_utils

import agenticwf.agent.core.orm.models.agent  # noqa: F401  (registers the tables on BASE.metadata)

LOG = get_logger(__name__)


def init_db() -> None:
    engine = db_utils.build_database()   # schema, missing tables, then Alembic to head
    LOG.info("database schema ready (%s)", engine.url.render_as_string(hide_password=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["init"])
    parser.parse_args()
    init_db()


if __name__ == "__main__":
    main()
