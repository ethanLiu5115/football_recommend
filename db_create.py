import sqlite3
import os

def create_football_database():
    """
    创建优化后的足球比赛智能推荐系统数据库（football.db）
    包含9张核心表，适配球队/赛事精简需求、上半场数据验证、盘口关联一致性等业务场景
    数据库文件存储于项目根目录的data文件夹下
    """
    # 创建data文件夹（若不存在）
    if not os.path.exists("data"):
        os.makedirs("data")
        print("✅ 已创建data文件夹（用于存储数据库文件）")

    # 数据库文件路径
    db_path = os.path.join("data", "football.db")

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

    # 2. 赛事表（league）- 仅存储赛事名称，作为模型特征
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS league (
        league_id INTEGER PRIMARY KEY AUTOINCREMENT,
        league_name TEXT NOT NULL UNIQUE,  -- 赛事名称唯一（如"英格兰足球超级联赛"）
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 数据录入时间（自动记录）
    )
    ''')
    print("✅ 已创建 league 表（赛事基础信息）")

    # 3. 比赛信息表（match）- 核心枢纽，关联球队、赛事
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS match (
        match_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_no TEXT NOT NULL,  -- 3位业务编号（如"001"）
        home_team_id INTEGER NOT NULL,
        away_team_id INTEGER NOT NULL,
        league_id INTEGER NOT NULL,
        betting_cycle_date DATE NOT NULL,  -- 竞彩周期日期（仅日期，如"2025-11-11"）
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        -- 外键约束
        FOREIGN KEY (home_team_id) REFERENCES team(team_id) ON DELETE CASCADE,
        FOREIGN KEY (away_team_id) REFERENCES team(team_id) ON DELETE CASCADE,
        FOREIGN KEY (league_id) REFERENCES league(league_id) ON DELETE CASCADE
    )
    ''')
    print("✅ 已创建 match 表（比赛核心信息）")

    # 4. 盘口信息表（handicap）- 存储整数盘口及验证状态
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS handicap (
        handicap_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER NOT NULL,  -- 关联比赛ID
        handicap_value INTEGER NOT NULL,  -- 盘口值（仅允许-2、-1、0、+1、+2）
        verify_status TEXT DEFAULT 'unverified',  -- 验证状态（未验证/已验证）
        verify_time TIMESTAMP,  -- 验证时间（验证后自动记录）
        -- 外键约束：比赛删除时同步删除盘口
        FOREIGN KEY (match_id) REFERENCES match(match_id) ON DELETE CASCADE
    )
    ''')
    print("✅ 已创建 handicap 表（盘口信息）")

    # 5. 预测者表（predictor）- 记录预测者及预测统计
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

    # 6. 预测信息表（prediction）- 存储具体预测内容，关联盘口确保一致性
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prediction (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER NOT NULL,  -- 关联比赛ID
        predictor_id INTEGER NOT NULL,  -- 关联预测者ID
        original_term TEXT NOT NULL,  -- 原始预测术语
        translated_result TEXT,  -- 翻译后的结果（暂留空）
        predict_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        -- 外键约束
        FOREIGN KEY (match_id) REFERENCES match(match_id) ON DELETE CASCADE,
        FOREIGN KEY (predictor_id) REFERENCES predictor(predictor_id) ON DELETE CASCADE
    )
    ''')
    print("✅ 已创建 prediction 表（预测详情）")

    # 7. 赛果信息表（result）- 含全场及上半场数据，支持"胜胜"等术语验证
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS result (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER NOT NULL UNIQUE,  -- 一场比赛只能有一个赛果（唯一约束）
        home_goals INTEGER NOT NULL,  -- 主队进球数（核心比分）
        away_goals INTEGER NOT NULL,  -- 客队进球数（核心比分）
        full_time_result TEXT NOT NULL,  -- 自动计算：主胜/平/主负
        goal_diff INTEGER NOT NULL,  -- 自动计算：主队-客队（净胜球）
        result_detail TEXT NOT NULL,  -- 自动生成：如"2:1 主胜"
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (match_id) REFERENCES match(match_id) ON DELETE CASCADE
    )
    ''')
    print("✅ 已创建 result 表（赛果信息，基于比分记录）")

    # 8. 球队历史战绩表（team_history）- 关联历史比赛，支持模型特征提取
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS team_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
        team_id INTEGER NOT NULL,  -- 球队ID
        opponent_team_id INTEGER NOT NULL,  -- 对手球队ID
        is_home INTEGER NOT NULL,  -- 是否主场（1=主场，0=客场）
        match_date DATE NOT NULL,  -- 比赛日期（YYYY-MM-DD）
        league_id INTEGER NOT NULL,  -- 赛事ID
        result TEXT NOT NULL,  -- 结果（胜/平/负）
        goal_scored INTEGER NOT NULL,  -- 该球队进球数
        goal_conceded INTEGER NOT NULL,  -- 该球队失球数
        history_match_id INTEGER,  -- 关联系统内比赛ID（可为空，兼容外部数据）
        -- 外键约束：关联数据删除时同步删除历史记录
        FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE,
        FOREIGN KEY (opponent_team_id) REFERENCES team(team_id) ON DELETE CASCADE,
        FOREIGN KEY (league_id) REFERENCES league(league_id) ON DELETE CASCADE,
        FOREIGN KEY (history_match_id) REFERENCES match(match_id) ON DELETE SET NULL
    )
    ''')
    print("✅ 已创建 team_history 表（球队历史战绩）")

    # 9. 日志表（log）- 记录系统操作，支持调试追溯
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        log_type TEXT NOT NULL,  -- 日志类型（info/error/warn）
        content TEXT NOT NULL,  -- 日志内容
        related_match_id INTEGER,  -- 关联比赛ID（可为空）
        log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- 日志生成时间
        -- 外键约束：比赛删除时保留日志，关联ID设为NULL
        FOREIGN KEY (related_match_id) REFERENCES match(match_id) ON DELETE SET NULL
    )
    ''')
    print("✅ 已创建 log 表（系统日志）")

    # 提交事务并关闭连接
    conn.commit()
    conn.close()
    print(f"\n🎉 数据库创建成功！文件路径：{os.path.abspath(db_path)}")
    print("提示：可通过PyCharm的SQLite插件打开该文件查看表结构")

if __name__ == "__main__":
    create_football_database()