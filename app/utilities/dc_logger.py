from logging import getLogger, INFO, Formatter, LoggerAdapter, StreamHandler, FileHandler
from colorlog import ColoredFormatter
from app.utilities.constants import Constants
import os
import sys

def get_logger(name, level=INFO, file_name = Constants.fetch_constant("log_config")["filepath"]):

    if not os.path.exists(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    # Colored console handler
    console_handler = StreamHandler(sys.stdout)
    color_format = "%(log_color)s%(levelname)-8s%(reset)s %(cyan)s%(asctime)s%(reset)s %(blue)s%(filename)s:%(lineno)d%(reset)s %(purple)s%(funcName)s%(reset)s %(log_color)s-->%(reset)s %(message)s"
    colored_formatter = ColoredFormatter(
        color_format,
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    console_handler.setFormatter(colored_formatter)
    
    # File handler without colors (for log files)
    file_handler = FileHandler(file_name)
    file_format = " %(levelname)s : %(asctime)s %(filename)s:%(lineno)d %(funcName)s --> %(message)s"
    file_formatter = Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    
    logger = getLogger(name)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level)
    return logger

class LoggerAdap(LoggerAdapter):
    def process(self,msg,kwargs):
        return '%s' % (msg), kwargs