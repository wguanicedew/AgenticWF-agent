"""
AgenticWF Exceptions.
"""

import traceback


class AgenticWFException(Exception):
    """
    To correctly use this class, inherit from it and define
    a 'message' property. That message will get printf'd
    with the keyword arguments provided to the constructor.
    """

    def __init__(self, *args, **kwargs):
        super(AgenticWFException, self).__init__()
        self._message = "An unknown AgenticWF exception occurred."
        self.args = args
        self.kwargs = kwargs
        self.error_code = 100
        self._error_string = None

    def construct_error_string(self):
        try:
            self._error_string = "%s: %s" % (self._message, self.kwargs)
        except Exception:
            # at least get the core message out if something happened
            self._error_string = self._message
        if len(self.args) > 0:
            # Convert all arguments into their string representations...
            args = ["%s" % arg for arg in self.args if arg]
            self._error_string = (self._error_string + "\nDetails: %s" % '\n'.join(args))
        return self._error_string.strip()

    def __str__(self):
        self.construct_error_string()
        return self._error_string.strip()

    def get_detail(self):
        self.construct_error_string()
        return self._error_string.strip() + "\nStacktrace: %s" % traceback.format_exc()


class DatabaseException(AgenticWFException):
    """
    Base exception for database related errors.
    """
    def __init__(self, *args, **kwargs):
        super(DatabaseException, self).__init__(*args, **kwargs)
        self._message = "Database exception."
        self.error_code = 200


class InvalidDatabaseType(DatabaseException):
    """
    Raised when a value cannot be adapted to its expected database type.
    """
    def __init__(self, *args, **kwargs):
        super(InvalidDatabaseType, self).__init__(*args, **kwargs)
        self._message = "Invalid Database type exception."
        self.error_code = 201


class NoObject(DatabaseException):
    """
    Raised when an expected database row does not exist.
    """
    def __init__(self, *args, **kwargs):
        super(NoObject, self).__init__(*args, **kwargs)
        self._message = "No object exception."
        self.error_code = 202


class DuplicatedObject(DatabaseException):
    """
    Raised when a database row already exists.
    """
    def __init__(self, *args, **kwargs):
        super(DuplicatedObject, self).__init__(*args, **kwargs)
        self._message = "Duplicated object exception."
        self.error_code = 203
