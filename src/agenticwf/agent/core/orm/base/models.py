"""
Base SQLAlchemy model for AgenticWF relational data.

Domain-specific tables should subclass (BASE, ModelBase) in their own module
and be imported before register_models()/unregister_models() are called, so
that they are registered on BASE.metadata.
"""

import datetime
from enum import Enum

from sqlalchemy import Boolean
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import object_mapper

from agenticwf.agent.core.common.utils import date_to_str
from agenticwf.agent.core.orm.base.enum import EnumSymbol
from agenticwf.agent.core.orm.base.session import BASE


@compiles(Boolean, "oracle")
def compile_boolean_oracle(type_, compiler, **kw):
    return "NUMBER(1)"


def utcnow():
    """The current time in UTC, naive: the DateTime columns store UTC without a zone."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


class ModelBase(object):
    """Base class for AgenticWF Models"""

    def save(self, flush=True, session=None):
        """Save this object"""
        session.add(self)
        if flush:
            session.flush()

    def delete(self, flush=True, session=None):
        """Delete this object"""
        session.delete(self)
        if flush:
            session.flush()

    def update(self, values, flush=True, session=None):
        """dict.update() behaviour."""
        for k, v in values.items():
            self[k] = v
        self["updated_at"] = utcnow()
        if session and flush:
            session.flush()

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self._column_items())

    def _column_items(self):
        return [(attr.key, getattr(self, attr.key)) for attr in object_mapper(self).column_attrs]

    def keys(self):
        return [key for key, _ in self.items()]

    def values(self):
        return [value for _, value in self.items()]

    def items(self):
        attr_items = self._column_items()
        items_extend = self._items_extend()
        return attr_items + items_extend

    def _items_extend(self):
        return []

    def to_dict(self):
        return {key: value for key, value
                in self.items() if not key.startswith('_')}

    def to_dict_json(self):
        return {key: self._expand_item(value) for key, value
                in self.items() if not key.startswith('_')}

    @classmethod
    def _expand_item(cls, obj):
        """
        Return a valid representation of `obj` depending on its type.
        """
        if isinstance(obj, datetime.datetime):
            return date_to_str(obj)
        elif isinstance(obj, (datetime.time, datetime.date)):
            return obj.isoformat()
        elif isinstance(obj, datetime.timedelta):
            return obj.days * 24 * 60 * 60 + obj.seconds
        elif isinstance(obj, EnumSymbol):
            return obj.description
        elif isinstance(obj, Enum):
            return obj.value

        return obj


def register_models(engine):
    """
    Creates database tables for all models registered on BASE.metadata.

    Domain model modules must be imported (so their classes are declared on
    BASE) before this is called.
    """
    BASE.metadata.create_all(engine)   # pylint: disable=maybe-no-member


def unregister_models(engine):
    """
    Drops database tables for all models registered on BASE.metadata.
    """
    BASE.metadata.drop_all(engine)   # pylint: disable=maybe-no-member
