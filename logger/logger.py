import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(
    save_dir: Path,
    log_config: str = "logger/logger_config.json",
    default_level: int = logging.INFO,
):
    """Set up logging configuration.

    Args:
        save_dir (Path): Where to save logs.
        log_config (str, optional): Log config file. Defaults to "logger/logger_config.json".
        default_level (int, optional): Default log level. Defaults to logging.INFO.
    """
    log_config_path = Path(log_config)
    if not log_config_path.is_file():
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)

    config = read_json(log_config)
    # modify logging paths based on run config
    for _, handler in config["handlers"].items():
        if "filename" in handler:
            handler["filename"] = str(save_dir / handler["filename"])

    logging.config.dictConfig(config)
