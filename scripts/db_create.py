import sqlite3
import os


def create_football_database():
    """
    创建优化后的足球比赛智能推荐系统数据库（football_dev.db）
    适配二级预测系统业务逻辑，仅保留必要表结构
    数据库文件存储于项目根目录的data/dev文件夹下
    """
    # 确保脚本无论从哪个目录运行，路径都正确
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data", "dev")

    # 创建data/dev文件夹（若不存在）
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("✅ 已创建data/dev文件夹（用于存储开发环境数据库文件）")

    # 数据库文件路径
    db_path = os.path.join(data_dir, "football_dev.db")

    # 连接数据库（不存在则自动创建）
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 启用外键约束（SQLite默认关闭，确保关联完整性）
    cursor.execute("PRAGMA foreign_keys = ON;")
    print("✅ 已启用外键约束")

    # 1. 球队表（team）- 仅存储球队名称，作为关联基础
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS team (
        team_id INTEGER PRIMARY KEY AUTOINCREMENT,
        team_name TEXT NOT NULL UNIQUE,  -- 球队名称唯一（如"曼彻斯特城"）
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 数据录入时间（自动记录）
    )
    ''')
    print("✅ 已创建 team 表（球队基础信息）")

    # 2. 比赛信息表（match）- 核心枢纽，关联球队，包含盘口信息
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS match (
        match_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_no TEXT NOT NULL,  -- 3位业务编号（如"001"）
        home_team_id INTEGER NOT NULL,
        away_team_id INTEGER NOT NULL,
        betting_cycle_date DATE NOT NULL,  -- 竞彩周期日期（仅日期，如"2025-11-11"）
        handicap_value INTEGER,  -- 让分盘口值（整数，如-2、-1、0、+1、+2，可为NULL表示无让球盘）
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        -- 外键约束
        FOREIGN KEY (home_team_id) REFERENCES team(team_id) ON DELETE CASCADE,
        FOREIGN KEY (away_team_id) REFERENCES team(team_id) ON DELETE CASCADE
    )
    ''')
    print("✅ 已创建 match 表（比赛核心信息，包含盘口字段）")

    # 3. 预测者表（predictor）- 记录预测者及预测统计
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictor (
        predictor_id INTEGER PRIMARY KEY AUTOINCREMENT,
        predictor_name TEXT NOT NULL UNIQUE,  -- 预测者名称唯一（如"用户A"）
        total_predictions INTEGER DEFAULT 0,  -- 总预测次数（自动累加）
        total_hits INTEGER DEFAULT 0,  -- 总命中次数（自动累加）
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 数据录入时间
    )
    ''')
    print("✅ 已创建 predictor 表（预测者信息）")

    # 4. 预测信息表（prediction）- 存储具体预测内容
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prediction (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER NOT NULL,  -- 关联比赛ID
        predictor_id INTEGER NOT NULL,  -- 关联预测者ID
        original_term TEXT NOT NULL,  -- 原始预测术语（如"让胜"、"平/负"）
        predict_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        -- 外键约束
        FOREIGN KEY (match_id) REFERENCES match(match_id) ON DELETE CASCADE,
        FOREIGN KEY (predictor_id) REFERENCES predictor(predictor_id) ON DELETE CASCADE
    )
    ''')
    print("✅ 已创建 prediction 表（预测详情，已删除translated_result字段）")

    # 5. 赛果信息表（result）- 含全场及上半场数据，支持半全场判定
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS result (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER NOT NULL UNIQUE,  -- 一场比赛只能有一个赛果（唯一约束）
        home_goals INTEGER NOT NULL,  -- 主队进球数（核心比分）
        away_goals INTEGER NOT NULL,  -- 客队进球数（核心比分）
        half_time_home_goals INTEGER,  -- 上半场主队进球数（可为NULL）
        half_time_away_goals INTEGER,  -- 上半场客队进球数（可为NULL）
        full_time_result TEXT NOT NULL,  -- 自动计算：主胜/平/主负
        half_time_result TEXT,  -- 自动计算：上半场主胜/平/主负（可为NULL）
        half_full_result TEXT,  -- 自动计算：半全场赛果（如"胜胜"、"平负"，可为NULL）
        goal_diff INTEGER NOT NULL,  -- 自动计算：主队-客队（净胜球）
        total_goals INTEGER NOT NULL,  -- 自动计算：总进球数（home_goals + away_goals）
        result_detail TEXT NOT NULL,  -- 自动生成：如"2:1 主胜"
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (match_id) REFERENCES match(match_id) ON DELETE CASCADE
    )
    ''')
    print("✅ 已创建 result 表（赛果信息，支持半全场判定）")

    # 6. 日志表（log）- 记录系统操作，支持调试追溯和环境隔离
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        env TEXT NOT NULL,  -- 环境标识（dev/prod），用于环境隔离
        log_type TEXT NOT NULL,  -- 日志类型（DATA_INPUT/MODEL_TRAINING/PREDICTION/ERROR/WARNING/INFO）
        content TEXT NOT NULL,  -- 日志内容
        related_match_id INTEGER,  -- 关联比赛ID（可为空）
        related_predictor_id INTEGER,  -- 关联预测者ID（可为空）
        related_prediction_id INTEGER,  -- 关联预测ID（可为空）
        details TEXT,  -- 详细信息（JSON格式，可为空）
        log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 日志生成时间
        -- 外键约束
        FOREIGN KEY (related_match_id) REFERENCES match(match_id) ON DELETE SET NULL,
        FOREIGN KEY (related_predictor_id) REFERENCES predictor(predictor_id) ON DELETE SET NULL,
        FOREIGN KEY (related_prediction_id) REFERENCES prediction(prediction_id) ON DELETE SET NULL
    )
    ''')
    print("✅ 已创建 log 表（系统日志，支持环境隔离）")

    # 提交事务并关闭连接
    conn.commit()
    conn.close()
    print(f"\n🎉 开发环境数据库创建成功！文件路径：{os.path.abspath(db_path)}")
    print("提示：可通过PyCharm的SQLite插件打开该文件查看表结构")


if __name__ == "__main__":
    create_football_database()