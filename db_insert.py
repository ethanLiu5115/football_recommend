import sqlite3
import os
from datetime import date


# -------------------------- 通用数据库工具函数 --------------------------
def get_db_connection():
    """创建并返回数据库连接（确保data文件夹存在）"""
    os.makedirs("data", exist_ok=True)  # 自动创建data文件夹（如果不存在）
    db_path = os.path.join("data", "football.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # 支持按列名访问
    return conn


def get_or_create_team_id(team_name, conn):
    """获取球队ID，不存在则创建"""
    cursor = conn.cursor()
    cursor.execute("SELECT team_id FROM team WHERE team_name = ?", (team_name,))
    team = cursor.fetchone()
    if team:
        return team["team_id"]
    # 不存在则插入新球队
    cursor.execute("INSERT INTO team (team_name) VALUES (?)", (team_name,))
    conn.commit()
    return cursor.lastrowid


def get_or_create_league_id(league_name, conn):
    """获取赛事ID，不存在则创建"""
    cursor = conn.cursor()
    cursor.execute("SELECT league_id FROM league WHERE league_name = ?", (league_name,))
    league = cursor.fetchone()
    if league:
        return league["league_id"]
    # 不存在则插入新赛事
    cursor.execute("INSERT INTO league (league_name) VALUES (?)", (league_name,))
    conn.commit()
    return cursor.lastrowid


def get_match_id(betting_cycle_date, match_no):
    """通过比赛编号和周期日期获取match_id"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT match_id FROM match 
        WHERE match_no = ? AND betting_cycle_date = ?
    ''', (match_no, betting_cycle_date))
    match = cursor.fetchone()
    conn.close()
    return match["match_id"] if match else None


def get_or_create_predictor_id(predictor_name, conn=None):
    """获取预测者ID，不存在则创建（支持外部传入连接以减少连接次数）"""
    close_conn = False  # 标记是否需要关闭连接
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    cursor = conn.cursor()
    cursor.execute("SELECT predictor_id FROM predictor WHERE predictor_name = ?", (predictor_name,))
    predictor = cursor.fetchone()
    if predictor:
        if close_conn:
            conn.close()
        return predictor["predictor_id"]

    # 不存在则插入新预测者
    cursor.execute("INSERT INTO predictor (predictor_name) VALUES (?)", (predictor_name,))
    conn.commit()
    new_id = cursor.lastrowid

    if close_conn:
        conn.close()
    return new_id


# -------------------------- 新比赛数据插入函数 --------------------------
def insert_new_match_basic_info():
    """插入新的比赛基础信息（含关联的球队和赛事）"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 新比赛基础数据：[赛事类型, 比赛编号, 主队, 客队]
    new_matches = [
        ("日联杯", "001", "广岛三箭", "横滨FC"),
        ("日联杯", "002", "柏太阳神", "川崎前锋"),
        ("世预赛", "003", "圣马力诺", "塞浦路斯"),
        ("世预赛", "004", "法罗群岛", "捷克"),
        ("世预赛", "005", "荷兰", "芬兰"),
        ("世预赛", "006", "罗马尼亚", "奥地利"),
        ("世预赛", "007", "丹麦", "希腊"),
        ("世预赛", "008", "克罗地亚", "直布罗陀"),
        ("美职足", "009", "奥斯汀FC", "洛杉矶FC"),
        ("世青赛", "010", "美国U20", "摩洛哥U20"),
        ("世青赛", "011", "挪威U20", "法国U20"),
    ]

    # 竞彩周期日期（统一使用2025-10-12）
    betting_cycle_date = date(2025, 10, 12).isoformat()

    for league_name, match_no, home_team, away_team in new_matches:
        # 获取或创建球队、赛事ID
        home_team_id = get_or_create_team_id(home_team, conn)
        away_team_id = get_or_create_team_id(away_team, conn)
        league_id = get_or_create_league_id(league_name, conn)

        # 插入比赛信息（忽略重复插入，避免报错）
        try:
            cursor.execute("""
                INSERT INTO match (match_no, home_team_id, away_team_id, league_id, betting_cycle_date)
                VALUES (?, ?, ?, ?, ?)
            """, (match_no, home_team_id, away_team_id, league_id, betting_cycle_date))
        except sqlite3.IntegrityError:
            print(f"比赛 {match_no} 已存在，跳过插入")

    conn.commit()
    conn.close()
    print("✅ 新比赛基础信息插入完成")


def insert_new_handicap_and_predictions():
    """插入新盘口数据和各预测者的预测（依赖比赛基础信息）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    betting_cycle_date = "2025-10-12"  # 与比赛基础信息日期一致

    # 新盘口和预测数据：[赛事, 比赛编号, 盘口, 首推预测, 次推预测]
    new_handicap_pred_data = [
        ("日联杯", "001", -1, "胜", ""),
        ("日联杯", "002", -1, "让负", ""),
        ("世预赛", "003", 2, "让胜", "让平"),
        ("世预赛", "004", 1, "让胜", "让平"),
        ("世预赛", "005", -3, "胜胜", ""),
        ("世预赛", "006", 1, "负", ""),
        ("世预赛", "007", -1, "胜", ""),
        ("世预赛", "008", -5, "让胜", "让平"),
        ("世预赛", "009", 1, "让负", ""),
        ("世青赛", "010", -1, "让负", ""),
        ("世青赛", "011", 1, "负", ""),
    ]

    # 复用已有预测者ID（如“植树”“大数据”等）
    zhishu_id = get_or_create_predictor_id("植树", conn)
    bigdata_id = get_or_create_predictor_id("大数据", conn)
    hongyun_id = get_or_create_predictor_id("鸿运", conn)

    for item in new_handicap_pred_data:
        league, match_no, handicap_value, first_pred, second_pred = item
        match_id = get_match_id(betting_cycle_date, match_no)
        if not match_id:
            print(f"⚠️ 未找到比赛 {match_no}，跳过盘口和预测插入")
            continue

        # 插入盘口
        try:
            cursor.execute('''
                INSERT INTO handicap (match_id, handicap_value, verify_status)
                VALUES (?, ?, '一致')
            ''', (match_id, handicap_value))
        except sqlite3.IntegrityError:
            print(f"盘口 {match_no} 已存在，跳过插入")

        # 插入“植树”的预测（合并首推和次推）
        original_term = first_pred if not second_pred else f"{first_pred}/{second_pred}"
        try:
            cursor.execute('''
                INSERT INTO prediction (match_id, predictor_id, original_term)
                VALUES (?, ?, ?)
            ''', (match_id, zhishu_id, original_term))
        except sqlite3.IntegrityError:
            print(f"“植树”对 {match_no} 的预测已存在，跳过插入")

    # 可按需添加其他预测者的预测逻辑（如“大数据”“鸿运”等）
    conn.commit()
    conn.close()
    print("✅ 新盘口和预测数据插入完成")


# -------------------------- 主函数（按顺序执行） --------------------------
def main():
    print("=== 开始插入新数据 ===")
    # 步骤1：插入新比赛基础信息（必须先执行，后续数据依赖）
    insert_new_match_basic_info()
    # 步骤2：插入新盘口和各预测者数据
    insert_new_handicap_and_predictions()
    print("=== 所有新数据插入操作完成 ===")


if __name__ == "__main__":
    main()