import os

# 项目根目录（假设config目录在项目根目录下）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 数据库文件路径（开发环境）
DB_PATH = os.path.join(PROJECT_ROOT, "data", "dev", "football_dev.db")

# 模型保存路径（开发环境）
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "dev")

# 训练日志保存路径（开发环境）
LOG_SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "dev", "training_logs")