"""
Utils to create the database or destroy the database
"""

import logging
import traceback

import sqlalchemy
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql.base import PGInspector
from sqlalchemy.schema import CreateSchema, MetaData, Table, DropTable, ForeignKeyConstraint, DropConstraint
from sqlalchemy.sql.ddl import DropSchema

from agenticwf.agent.core.orm.base import migrate, models, session

LOG = logging.getLogger(__name__)


def build_database(echo=True, tests=False):
    """Build the database: the configured schema, missing tables, then Alembic to head. """
    engine = session.get_engine(echo=echo)

    schema = session.DEFAULT_SCHEMA_NAME
    if schema:
        try:
            with engine.begin() as conn:
                conn.execute(CreateSchema(schema, if_not_exists=True))
        except Exception as e:
            LOG.warning('Cannot create schema %s, please validate manually if schema creation is needed, '
                        'continuing: %s', schema, e)

    models.register_models(engine)   # CREATE TABLE IF NOT EXISTS for every model

    # Revisions are idempotent (they check before they alter), so a database without a recorded
    # revision runs all of them: tables that existed before create_all get their new columns.
    migrate.upgrade("head")
    return engine


def destroy_database(echo=True):
    """Destroy the database"""
    engine = session.get_engine(echo=echo)

    try:
        models.unregister_models(engine)
    except Exception as e:
        print('Cannot destroy schema -- assuming already gone, continuing:', e)
        print(traceback.format_exc())


def destroy_everything(echo=True):
    """ Using metadata.reflect() to get all constraints and tables.
        metadata.drop_all() as it handles cyclical constraints between tables.
        Ref. http://www.sqlalchemy.org/trac/wiki/UsageRecipes/DropEverything
    """
    engine = session.get_engine(echo=echo)

    try:
        # the transaction only applies if the DB supports
        # transactional DDL, i.e. Postgresql, MS SQL Server
        with engine.begin() as conn:

            inspector = inspect(conn)

            for tname, fkcs in reversed(
                    inspector.get_sorted_table_and_fkc_names(schema='*')):
                if tname:
                    drop_table_stmt = DropTable(Table(tname, MetaData(), schema='*'))
                    conn.execute(drop_table_stmt)
                elif fkcs:
                    if not engine.dialect.supports_alter:
                        continue
                    for tname, fkc in fkcs:
                        fk_constraint = ForeignKeyConstraint((), (), name=fkc)
                        Table(tname, MetaData(), fk_constraint)
                        drop_constraint_stmt = DropConstraint(fk_constraint)
                        conn.execute(drop_constraint_stmt)

            if session.DEFAULT_SCHEMA_NAME:
                conn.execute(DropSchema(session.DEFAULT_SCHEMA_NAME, cascade=True))

            if engine.dialect.name == 'postgresql':
                assert isinstance(inspector, PGInspector), 'expected a PGInspector'
                for enum in inspector.get_enums(schema='*'):
                    sqlalchemy.Enum(**enum).drop(bind=conn)

    except Exception as e:
        print('Cannot destroy db:', e)
        print(traceback.format_exc())


def dump_schema():
    """ Creates a schema dump to a specific database. """
    engine = session.get_dump_engine()
    models.unregister_models(engine)
    print("\n")
    models.register_models(engine)


def row2dict(row):
    """ Convert a result row to dict. """
    return {str(col): value for col, value in row._mapping.items()}


def rows2dict(rows):
    """ Convert result rows to dicts. """
    return [row2dict(row) for row in rows]
