import typing
import logging as std_logging

if typing.TYPE_CHECKING:
    from loguru import Logger
else:
    Logger = None

__all__ = ["logger"]


def __get_logger():
    """获取 logger，优先使用 loguru，否则使用标准库 logging"""
    try:
        from loguru import logger
        return logger
    except ImportError:
        # 如果 loguru 不存在，使用标准库 logging
        return std_logging.getLogger("llava")


logger = __get_logger()
