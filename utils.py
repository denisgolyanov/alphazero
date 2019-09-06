import logging
import sys

EXTREME_DEBUG_LEVEL = 5

def setup_logger(name, log_file, level):
    handler = logging.FileHandler(log_file)
    internal_logger = logging.getLogger(name)
    internal_logger.setLevel(level)
    internal_logger.addHandler(handler)
    return internal_logger

#logging.basicConfig(format='%(asctime)-15s:%(levelname)-4s:\t%(message)s')
#LOG_FILE_PATH = "/home/toky/shalgi/alphazero/logs/log.info"
#DEBUG_LOG_FILE_PATH = "/home/toky/shalgi/alphazero/logs/log.debug"

LOG_FILE_PATH = r"log.info"
logger = setup_logger("info", LOG_FILE_PATH, logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)
#debug_logger = setup_logger("debug", DEBUG_LOG_FILE_PATH, logging.DEBUG)

import types
def verbose_debug(logger, *args, **kwargs):
    logger.log(EXTREME_DEBUG_LEVEL, *args, **kwargs)

logger.verbose_debug = types.MethodType(verbose_debug, logger)
