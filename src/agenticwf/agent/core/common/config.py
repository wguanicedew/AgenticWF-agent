"""
Configurations.

configuration file looking for path:
    1. $AGENTICWF_CONFIG
    2. $AGENTICWF_HOME/etc/agenticwf/agent.cfg
    3. /etc/agenticwf/agent.cfg
    4. $VIRTUAL_ENV/etc/agenticwf/agent.cfg

A '#' after a value, preceded by whitespace, starts a comment: `provider = vllm  # local GPUs`.
"""

import os

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


def config_has_section(section):
    """
    Return whether there is a section

    :param section: the named section.
    :returns: True/False.
    """
    __CONFIG = get_config()
    return __CONFIG.has_section(section)


def config_has_option(section, option):
    """
    Return whether there is an option for a given option in a section

    :param section: the named section.
    :param option: the named option.
    :returns: True/False.
    """
    __CONFIG = get_config()
    return __CONFIG.has_option(section, option)


def config_list_options(section):
    """
    Return list of (name, value) for a given section

    :param section: the named section.
    :returns: list of (name, value).
    """
    __CONFIG = get_config()
    return __CONFIG.items(section)


def config_get(section, option):
    """
    Return the string value for a given option in a section

    :param section: the named section.
    :param option: the named option.
    :returns: the configuration value.
    """
    __CONFIG = get_config()
    return __CONFIG.get(section, option)


def config_get_int(section, option):
    """
    Return the integer value for a given option in a section

    :param section: the named section.
    :param option: the named option.
    :returns: the integer configuration value.
    """
    __CONFIG = get_config()
    return __CONFIG.getint(section, option)


def config_get_float(section, option):
    """
    Return the float value for a given option in a section

    :param section: the named section.
    :param option: the named option.
    :returns: the float configuration value.
    """
    __CONFIG = get_config()
    return __CONFIG.getfloat(section, option)


def config_get_bool(section, option):
    """
    Return the boolean value for a given option in a section

    :param section: the named section.
    :param option: the named option.
    :returns: the boolean configuration value.
    """
    __CONFIG = get_config()
    return __CONFIG.getboolean(section, option)


def new_parser():
    """
    Return an empty parser that reads configuration files the way the agent does

    :returns: a ConfigParser that strips comments after values.
    """
    return ConfigParser.ConfigParser(inline_comment_prefixes=('#',))


def get_main_config_file():
    __CONFIG = new_parser()
    if os.environ.get('AGENTICWF_CONFIG', None):
        configfile = os.environ['AGENTICWF_CONFIG']
        if __CONFIG.read(configfile) == [configfile]:
            return configfile
    else:
        configfiles = ['%s/etc/agenticwf/agent.cfg' % os.environ.get('AGENTICWF_HOME', ''),
                       '/etc/agenticwf/agent.cfg',
                       '%s/etc/agenticwf/agent.cfg' % os.environ.get('VIRTUAL_ENV', '')]
        for configfile in configfiles:
            if __CONFIG.read(configfile) == [configfile]:
                return configfile
    return None


def get_main_config():
    __CONFIG = new_parser()

    __HAS_CONFIG = False
    if os.environ.get('AGENTICWF_CONFIG', None):
        configfile = os.environ['AGENTICWF_CONFIG']
        if not __CONFIG.read(configfile) == [configfile]:
            raise Exception('AGENTICWF_CONFIG is defined as %s, ' % configfile,
                            'but could not load configurations from it.')
        __HAS_CONFIG = True
    else:
        configfiles = ['%s/etc/agenticwf/agent.cfg' % os.environ.get('AGENTICWF_HOME', ''),
                       '/etc/agenticwf/agent.cfg',
                       '%s/etc/agenticwf/agent.cfg' % os.environ.get('VIRTUAL_ENV', '')]

        for configfile in configfiles:
            if __CONFIG.read(configfile) == [configfile]:
                __HAS_CONFIG = True
                break
    return __CONFIG, __HAS_CONFIG


def get_config():
    __CONFIG, __HAS_CONFIG = get_main_config()

    if not __HAS_CONFIG:
        raise Exception("Could not load configuration file."
                        "AgenticWF looks for a configuration file, in order:"
                        "\n\t${AGENTICWF_CONFIG}"
                        "\n\t${AGENTICWF_HOME}/etc/agenticwf/agent.cfg"
                        "\n\t/etc/agenticwf/agent.cfg"
                        "\n\t${VIRTUAL_ENV}/etc/agenticwf/agent.cfg")
    return __CONFIG
