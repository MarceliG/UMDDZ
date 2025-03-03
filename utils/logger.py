import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s() | %(message)s",
    datefmt="%m-%d-%Y %H:%M:%S",
)

logger = logging.getLogger()
