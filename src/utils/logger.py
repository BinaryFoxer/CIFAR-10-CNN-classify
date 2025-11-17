import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config import config

def setup_logger(
        name: str = __name__,
        log_dir: str = config.LOG_DIR,
        log_file: str = "training.log",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        mode: str = 'a'
) -> logging.Logger:
    """
    创建并配置模块专属日志记录器

    参数:
        name (str): 通常是 __name__，用于区分不同模块
        log_dir (str): 日志目录路径
        log_file (str): 日志文件名
        console_level: 控制台日志级别
        file_level: 文件日志级别
        mode: 文件写入模式 ('w' 覆盖 / 'a' 追加)
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 创建记录器（以模块名区分）
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置全局最低级别

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 日志格式（包含模块名）
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件处理器（输出到文件）
    file_handler = logging.FileHandler(
        filename=Path(log_dir) / log_file,
        mode=mode,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    # 控制台处理器（输出到终端）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
