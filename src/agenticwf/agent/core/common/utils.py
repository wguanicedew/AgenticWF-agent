"""
Common utility functions.
"""

import datetime
import json
import re

DATE_FORMAT = "%a, %d %b %Y %H:%M:%S UTC"

_SLUG_INVALID_CHARS_RE = re.compile(r'[^a-z0-9]+')


def str_to_date(string):
    """
    Converts a string to the corresponding datetime value.

    :param string: the string to convert to datetime value.
    """
    return datetime.datetime.strptime(string, DATE_FORMAT) if string else None


def date_to_str(date):
    """
    Converts a datetime value to a string.

    :param date: the datetime value to convert.
    """
    return datetime.datetime.strftime(date, DATE_FORMAT) if date else None


def json_dumps(obj, indent=None, sort_keys=False):
    return json.dumps(obj, indent=indent, sort_keys=sort_keys, default=str)


def json_loads(obj):
    return json.loads(obj)


def slugify(text, max_length=64):
    """
    Lowercase, hyphen-separate, and truncate text into a name matching
    `runtime.execution.skill.verifier.NAME_RE` (`^[a-z0-9]+(-[a-z0-9]+)*$`).
    Falls back to "untitled" if nothing alphanumeric survives.
    """
    slug = _SLUG_INVALID_CHARS_RE.sub('-', text.lower()).strip('-')
    slug = slug[:max_length].rstrip('-')
    return slug or 'untitled'
