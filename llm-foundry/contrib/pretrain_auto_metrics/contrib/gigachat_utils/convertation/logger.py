import logging
import time
import os
import decorator
import psutil


@decorator.decorator    
def time_memory_stat(
    func, 
    *args, **kwargs
):
    process = psutil.Process(os.getpid())
    start_time = time.time()
    logger.info(f'{func.__name__} started')
    result = func(*args, **kwargs)
    logger.info(f'{func.__name__} done, time = {time.time() - start_time:.3f}, current memory {process.memory_info().rss / 1024 ** 3:.2f} Gb')
    return result

def setup_logger():
    """
    Set up and return a configured logger.
    """
    # Create a logger
    config = {
        "format": '%(asctime)s - %(levelname)s - %(message)s',
        "datefmt": '%d-%b-%y %H:%M:%S' 
    }
    logging.basicConfig(**config)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

def print_args(args):
    msg = ''
    for key, value in args.__dict__.items():
        msg += f'{key}: {value} \n'
    return msg

logger = setup_logger()
