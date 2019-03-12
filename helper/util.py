import os
# import sys
from importlib import import_module
from functools import wraps
import logging
import logging.config
# import inspect
import json
import yaml

def get_attribute(kls):
    """ Python version of Class.forName() in Java """
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    # m = __import__( module )
    m = import_module(module )
    return getattr(m, parts[len(parts)-1] )

def get_logger(module_name):
    """ get the default logger """
    # logging.config.fileConfig('stock/logging.conf')
    # create logger
    setup_logging()
    logger = logging.getLogger(module_name)

    return logger


def setup_logging(default_path='helper/logging.yaml',
                  default_level=logging.INFO, 
                  env_key='LOG_CFG'):
    """Setup YAML logging configuration """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def setup_logging_json(default_path='helper/logging.json',
                  default_level=logging.INFO,
                  env_key='LOG_CFG'):
    """Setup JSON logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def catch_exception(logger):
    """ a decorator to log the messages of exceptions """
    def decorated(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if logger:
                    logger.error(e)
                raise
        return wrapped
    return decorated


def get_timestamp():
    import calendar;
    import time;
    return calendar.timegm(time.gmtime())


LOGGER = get_logger(__name__)

@catch_exception(LOGGER)
def div_zero():
    return 5/0

if __name__ == '__main__':
    # LOGGER.error("this is a test")
    # setup_logging()
    # print(div_zero())
#     print('tes')
#     print(__name__.split('.')[0])
    print(get_timestamp())

