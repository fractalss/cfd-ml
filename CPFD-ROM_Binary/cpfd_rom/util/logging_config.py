from __future__ import annotations

import logging
import sys


DETAIL_LEVEL = 15
logging.addLevelName(DETAIL_LEVEL, "DETAIL")


def configure_logging(verbosity: int = 0, quiet: bool = False) -> None:
    """
    Configure terminal logging for the ROM command-line interface.

    Levels
    ------
    default:
        INFO and above
    -v:
        DETAIL and above
    -vv:
        DEBUG and above
    --quiet:
        ERROR and above
    """
    if quiet:
        level = logging.ERROR
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = DETAIL_LEVEL
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def detail(logger: logging.Logger, message: str, *args, **kwargs) -> None:
    """Write a message visible with -v or -vv."""
    logger.log(DETAIL_LEVEL, message, *args, **kwargs)