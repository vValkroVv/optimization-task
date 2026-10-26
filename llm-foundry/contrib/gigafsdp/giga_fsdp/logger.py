import logging
import sys
import os
import torch.distributed as dist

class RankFilter(logging.Filter):
    """
    A logging filter that only allows logs from rank 0.
    """
    def filter(self, record):
        # Check if distributed training is initialized
        if dist.is_initialized():
            # Only allow logs from rank 0
            return dist.get_rank() == 0
        # If distributed training is not initialized, allow all logs
        return True
    
def setup_logger():
    """
    Set up and return a configured logger.
    """
    # Create a logger
    logger = logging.getLogger(__name__)

    # Check if DEBUG_GIGAFSDP is set to "1"
    if os.environ.get('DEBUG_GIGAFSDP', None) == "1":
        logger.setLevel(logging.DEBUG)
    else:
        # If DEBUG_GIGAFSDP is not "1", set the logger level to INFO or higher
        logger.setLevel(logging.INFO)

    # Prevent propagation to the root logger
    logger.propagate = False

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s.%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create a stream handler (output to console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)  # Set handler-specific log level
    stream_handler.setFormatter(formatter)
    # Add the RankFilter to the handler
    stream_handler.addFilter(RankFilter())

    # Add the handler to the logger
    logger.addHandler(stream_handler)

    return logger

logger = setup_logger()