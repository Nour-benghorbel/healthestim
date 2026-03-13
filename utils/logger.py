import logging
from config import LOG_PATH

def setup_logger():
    """
    Configure le système de logs de l'application.
    """
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("healthestim")