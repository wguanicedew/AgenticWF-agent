"""
Engine, session factory and session decorators.

The engine and the sessionmaker are created once, lazily, on first use (thread safe). Code takes a
session one of three ways:

    with session_scope() as session: ...      commit on success, rollback on error
    @transactional_session                     the same, for a function taking ``session=``
    @read_session / @stream_session            SELECTs only: never commits

A function given ``session=`` by its caller runs inside the caller's transaction.

"""

import logging
import sys

from contextlib import contextmanager
from functools import wraps
from inspect import isgeneratorfunction
from os.path import basename
from threading import Lock
from typing import Any, Iterator

from retrying import retry
from sqlalchemy import MetaData, create_engine, create_mock_engine, event, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DatabaseError, DBAPIError, OperationalError, TimeoutError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from agenticwf.agent.core.common.config import config_get, config_has_option
from agenticwf.agent.core.common.exceptions import AgenticWFException, DatabaseException
from agenticwf.agent.core.orm.base.types import JSON


LOG = logging.getLogger(__name__)

DATABASE_SECTION = 'database'

DEFAULT_SCHEMA_NAME = None
if config_has_option(DATABASE_SECTION, 'schema'):
    DEFAULT_SCHEMA_NAME = config_get(DATABASE_SECTION, 'schema') or None


class Base(DeclarativeBase):
    """Declarative base of every table. Typed (``Mapped[...]``) and classic ``Column`` declarations both work."""

    metadata = MetaData(schema=DEFAULT_SCHEMA_NAME)
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


BASE = Base

_ENGINE: Engine | None = None
_MAKER: sessionmaker[Session] | None = None
_LOCK = Lock()


def _fk_pragma_on_connect(dbapi_con, con_record):
    # Hack for previous versions of sqlite3
    try:
        dbapi_con.execute('pragma foreign_keys=ON')
    except AttributeError:
        pass


def mysql_convert_decimal_to_float(dbapi_conn, connection_rec):
    """
    The default datatype returned by mysql-python for numerics is decimal.Decimal.
    This type cannot be serialised to JSON, therefore we need to autoconvert to floats.
    Even worse, there's two types of decimals created by the MySQLdb driver, so we must
    override both.
    :param dbapi_conn: DBAPI connection
    :param connection_rec: connection record
    """

    try:
        import MySQLdb.converters  # pylint: disable=import-error
        from MySQLdb.constants import FIELD_TYPE  # pylint: disable=import-error
    except ImportError:
        raise AgenticWFException('Trying to use MySQL without mysql-python installed!')
    conv = MySQLdb.converters.conversions.copy()
    conv[FIELD_TYPE.DECIMAL] = float
    conv[FIELD_TYPE.NEWDECIMAL] = float
    dbapi_conn.converter = conv


def psql_convert_decimal_to_float():
    """
    The default datatype returned by psycopg2 for numerics is decimal.Decimal.
    This type cannot be serialised to JSON, therefore we need to autoconvert to floats.
    The type is registered globally, so once per process is enough.
    """

    try:
        import psycopg2.extensions  # pylint: disable=import-error
    except ImportError:
        raise AgenticWFException('Trying to use PostgreSQL without psycopg2 installed!')

    DEC2FLOAT = psycopg2.extensions.new_type(psycopg2.extensions.DECIMAL.values,
                                             'DEC2FLOAT',
                                             lambda value, curs: float(value) if value is not None else None)
    psycopg2.extensions.register_type(DEC2FLOAT)


def my_on_connect(dbapi_con, connection_record):
    """ Adds information to track performance and ressource by module.
        Info are recorded in the V$SESSION and V$SQLAREA views.
    """
    caller = basename(sys.argv[0])
    dbapi_con.clientinfo = caller
    dbapi_con.client_identifier = caller
    dbapi_con.module = caller
    dbapi_con.action = caller


def _engine_params(sql_connection):
    config_params = [('pool_size', int), ('max_overflow', int), ('pool_timeout', int),
                     ('pool_recycle', int), ('echo', int), ('echo_pool', str),
                     ('pool_reset_on_return', str), ('pool_pre_ping', int)]
    # A pooled connection is tested before use, and replaced when the server dropped it.
    params = {'pool_pre_ping': True}
    for param, param_type in config_params:
        if config_has_option(DATABASE_SECTION, param):
            try:
                params[param] = param_type(config_get(DATABASE_SECTION, param))
            except (TypeError, ValueError):
                LOG.warning('ignoring invalid [%s] %s', DATABASE_SECTION, param)
    params['pool_pre_ping'] = bool(params['pool_pre_ping'])
    params['execution_options'] = {'schema_translate_map': {None: DEFAULT_SCHEMA_NAME}}
    # Errors (logged, and turned into DatabaseException messages) never carry the statement's
    # values: they can hold secrets, e.g. a URL with a token.
    params['hide_parameters'] = True
    if 'oracledb' in sql_connection:
        try:
            import oracledb  # pylint: disable=import-error
            oracledb.init_oracle_client()
            params['thick_mode'] = True
        except Exception as err:
            LOG.warning('Could not start Oracle thick mode; falling back to thin: %s', err)
    return params


def _create_engine():
    sql_connection = config_get(DATABASE_SECTION, 'default')
    engine = create_engine(sql_connection, **_engine_params(sql_connection))

    dialect = engine.dialect.name
    if dialect == 'mysql':
        event.listen(engine, 'connect', mysql_convert_decimal_to_float)
    elif dialect == 'postgresql':
        if engine.dialect.driver == 'psycopg2':
            psql_convert_decimal_to_float()
    elif dialect == 'sqlite':
        event.listen(engine, 'connect', _fk_pragma_on_connect)
    elif dialect == 'oracle':
        event.listen(engine, 'connect', my_on_connect)
    return engine


def get_engine(echo=True) -> Engine:
    """ The engine of the configured database, created on first use.
        :returns: engine
    """
    global _ENGINE, _MAKER
    if _ENGINE is None:
        with _LOCK:
            if _ENGINE is None:
                engine = _create_engine()
                _MAKER = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
                _ENGINE = engine
    return _ENGINE


def get_maker() -> sessionmaker[Session]:
    """ The sessionmaker bound to the engine, created with it. """
    get_engine()
    assert _MAKER is not None
    return _MAKER


def get_session() -> Session:
    """ A new session on the configured database, assumes that schema already in place.
        The caller closes it (or uses session_scope()).
        :returns: session
    """
    return get_maker()()


def dispose_engine():
    """ Closes the pool's connections and forgets the engine: the next use creates a new one
        (e.g. after the configuration changed, or in a forked process). """
    global _ENGINE, _MAKER
    with _LOCK:
        if _ENGINE is not None:
            _ENGINE.dispose()
        _ENGINE, _MAKER = None, None


def get_dump_engine():
    """ Creates a dump engine to a specific database.
        :returns: engine """

    statements = list()

    def dump(sql, *multiparams, **params):
        statement = str(sql.compile(dialect=engine.dialect))
        if statement in statements:
            return
        statements.append(statement)
        if statement.endswith(')\n\n'):
            if engine.dialect.name == 'oracle':
                print(statement.replace(')\n\n', ') PCTFREE 0;\n'))
            else:
                print(statement.replace(')\n\n', ');\n'))
        elif statement.endswith(')'):
            print(statement.replace(')', ');\n'))
        else:
            print(statement)
    sql_connection = config_get(DATABASE_SECTION, 'default')

    engine = create_mock_engine(sql_connection, dump)
    return engine


_CONN_ERR_CODES = ('2002', '2003', '2006', '2013',  # MySQL
                   'ORA-00028',  # Oracle session has been killed
                   'ORA-01012',  # not logged on
                   'ORA-03113',  # end-of-file on communication channel
                   'ORA-03114',  # not connected to ORACLE
                   'ORA-03135',  # connection lost contact
                   'ORA-25408',  # can not safely replay call
                   'server closed the connection unexpectedly',  # PostgreSQL
                   'closed the connection',)


def retry_if_db_connection_error(exception):
    """Return True if error in connecting to db."""
    if isinstance(exception, DatabaseException):
        # raised by the decorators below from the SQLAlchemy error
        exception = exception.__cause__ or exception
    if isinstance(exception, DBAPIError) and exception.connection_invalidated:
        return True
    if isinstance(exception, (OperationalError, DatabaseException)):
        message = str(exception.args[0]) if exception.args else ''
        return any(code in message for code in _CONN_ERR_CODES)
    return False


# Only a call that opened its own session is retried: a caller's transaction can't be replayed.
_retry_on_connection_error = retry(retry_on_exception=retry_if_db_connection_error,
                                   wait_fixed=500,   # milliseconds
                                   stop_max_attempt_number=2,
                                   wrap_exception=False)


@contextmanager
def _scope(commit) -> Iterator[Session]:
    session = get_session()
    try:
        yield session
        if commit:
            session.commit()
    except (TimeoutError, DatabaseError) as error:
        LOG.error('database error: %s', error)
        session.rollback()
        raise DatabaseException(str(error)) from error
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """ Transactional scope: commit on success, rollback on error. SQLAlchemy errors are raised
        as DatabaseException. """
    with _scope(commit=True) as session:
        yield session


def _session_decorator(function, commit):
    @_retry_on_connection_error
    def with_own_session(*args, **kwargs):
        with _scope(commit) as session:
            return function(*args, session=session, **kwargs)

    @wraps(function)
    def new_funct(*args, **kwargs):
        if kwargs.get('session') is not None:
            return function(*args, **kwargs)
        kwargs.pop('session', None)
        return with_own_session(*args, **kwargs)
    return new_funct


def read_session(function):
    '''
    decorator that set the session variable to use inside a function.
    With that decorator it's possible to use the session variable like if a global variable session is declared.
    session is a sqlalchemy session, and you can get one calling get_session().
    This is useful if only SELECTs and the like are being done; anything involving
    INSERTs, UPDATEs etc should use transactional_session.
    '''
    if isgeneratorfunction(function):
        raise AgenticWFException('read_session decorator should not be used with generator. Use stream_session instead.')
    return _session_decorator(function, commit=False)


def transactional_session(function):
    '''
    decorator that set the session variable to use inside a function.
    With that decorator it's possible to use the session variable like if a global variable session is declared.
    session is a sqlalchemy session, and you can get one calling get_session().
    '''
    return _session_decorator(function, commit=True)


def stream_session(function):
    '''
    decorator that set the session variable to use inside a generator function.
    The session stays open while the caller iterates, and is closed when the generator ends or is closed.
    This is useful if only SELECTs and the like are being done; anything involving
    INSERTs, UPDATEs etc should use transactional_session.
    '''
    if not isgeneratorfunction(function):
        raise AgenticWFException('stream_session decorator should be used only with generator. Use read_session instead.')

    @wraps(function)
    def new_funct(*args, **kwargs):
        if kwargs.get('session') is not None:
            yield from function(*args, **kwargs)
            return
        kwargs.pop('session', None)
        with _scope(commit=False) as session:
            yield from function(*args, session=session, **kwargs)
    return new_funct


def safe_bulk_update_mappings(session, model, mappings):
    """ bulk_update_mappings() on the rows of `mappings` that no other transaction holds locked
        (skipped, not waited for). """
    if not mappings:
        return

    # detect the primary key column name(s)
    mapper = inspect(model)
    pk_cols = [col.name for col in mapper.primary_key]

    if len(pk_cols) != 1:
        return session.bulk_update_mappings(model, mappings)

    pk = pk_cols[0]

    # extract ids from mappings
    ids = [m[pk] for m in mappings if pk in m]

    if not ids:
        return

    # claim only rows that are not locked
    stmt = (
        select(getattr(model, pk))
        .where(getattr(model, pk).in_(ids))
        .with_for_update(skip_locked=True)
    )
    claimed = set(session.execute(stmt).scalars().all())

    if not claimed:
        return

    # filter mappings to only claimed rows
    filtered = [m for m in mappings if m.get(pk) in claimed]

    if filtered:
        return session.bulk_update_mappings(model, filtered)
