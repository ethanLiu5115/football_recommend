import streamlit as st
import sqlite3
import os
import sys
import pandas as pd
from datetime import date, datetime, timedelta

# -------------------------- 全局初始化：环境适配+Session State --------------------------
# 项目根目录路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 环境加载与标识
CURRENT_ENV = os.getenv("FOOTBALL_ENV", "dev")
if CURRENT_ENV == "prod":
    from config.prod_config import DB_PATH
else:
    from config.dev_config import DB_PATH

# Session State初始化（存储临时数据）
if "default_betting_date" not in st.session_state:
    st.session_state.default_betting_date = date.today()
if "min_goals" not in st.session_state:
    st.session_state.min_goals = 0
if "undo_data" not in st.session_state:
    st.session_state.undo_data = None  # 存储撤销操作数据（无时间限制）


# -------------------------- 核心工具函数 --------------------------
def get_db_connection():
    """建立数据库连接（统一复用）"""
    if not os.path.exists(DB_PATH):
        st.error(f"数据库文件不存在：{DB_PATH}，请先运行数据库创建脚本")
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def log_operation(log_type, content, related_match_id=None, related_predictor_id=None, related_prediction_id=None,
                  details=None):
    """记录操作日志到log表"""
    conn = get_db_connection()
    if not conn:
        return
    try:
        conn.execute('''
            INSERT INTO log (env, log_type, content, related_match_id, related_predictor_id, related_prediction_id, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (CURRENT_ENV, log_type, content, related_match_id, related_predictor_id, related_prediction_id, details))
        conn.commit()
    except Exception as e:
        st.warning(f"日志记录失败：{str(e)}")
    finally:
        conn.close()


# -------------------------- 表结构初始化（首次运行自动更新） --------------------------
def init_prediction_table():
    """更新prediction表，新增prediction_type字段（支持多类型预测）"""
    conn = get_db_connection()
    if not conn:
        return
    try:
        # 检查字段是否已存在，不存在则新增
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(prediction)")
        columns = [col[1] for col in cursor.fetchall()]
        if "prediction_type" not in columns:
            cursor.execute('''
                ALTER TABLE prediction ADD COLUMN prediction_type TEXT NOT NULL DEFAULT '胜平负/让球胜平负'
            ''')
            conn.commit()
            print("✅ 已更新prediction表，新增prediction_type字段")
    except Exception as e:
        print(f"更新prediction表失败：{str(e)}")
    finally:
        conn.close()


# -------------------------- Tab1：比赛+盘口录入 函数 --------------------------
def get_existing_teams():
    """获取所有已存在的球队名称"""
    conn = get_db_connection()
    if not conn:
        return []
    teams = conn.execute("SELECT team_name FROM team").fetchall()
    conn.close()
    return [row["team_name"] for row in teams]


def get_or_create_team_id(team_name):
    """查找球队ID，不存在则创建"""
    conn = get_db_connection()
    if not conn:
        return None
    team = conn.execute("SELECT team_id FROM team WHERE team_name = ?", (team_name,)).fetchone()
    if team:
        conn.close()
        return team["team_id"]
    # 创建新球队
    cursor = conn.cursor()
    cursor.execute("INSERT INTO team (team_name) VALUES (?)", (team_name,))
    conn.commit()
    new_team_id = cursor.lastrowid
    conn.close()
    log_operation("DATA_INPUT", f"新增球队：{team_name}")
    st.success(f"已新增球队：{team_name}")
    return new_team_id


def is_match_exists(match_no, betting_cycle_date):
    """检查（日期+编号）是否已存在"""
    conn = get_db_connection()
    if not conn:
        return False
    count = conn.execute('''
        SELECT COUNT(*) as c FROM match 
        WHERE match_no = ? AND betting_cycle_date = ?
    ''', (match_no, betting_cycle_date)).fetchone()["c"]
    conn.close()
    return count > 0


def save_match_and_handicap(match_no, betting_cycle_date, home_team, away_team, handicap_value):
    """保存比赛信息+盘口（合并为一个操作），添加盘口非0和编号规则验证"""
    # 比赛编号验证（001-099）
    if not (match_no.isdigit() and len(match_no) == 3):
        st.error("❌ 比赛编号必须是3位数字")
        return False
    if match_no[0] != '0':
        st.error("❌ 比赛编号第一位必须为0（格式：001-099）")
        return False
    if not (1 <= int(match_no) <= 99):
        st.error("❌ 比赛编号范围：001-099")
        return False

    # 盘口非0验证（让球盘口不能为0）
    if handicap_value == 0:
        st.error("❌ 让球盘口不能为0，请输入非0整数（主队让球为负，受让为正）")
        return False

    # 球队ID获取
    home_team_id = get_or_create_team_id(home_team)
    away_team_id = get_or_create_team_id(away_team)
    if not all([home_team_id, away_team_id]):
        st.error("球队信息获取失败，无法保存")
        return False

    # 数据库保存
    conn = get_db_connection()
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO match (match_no, home_team_id, away_team_id, betting_cycle_date, handicap_value)
            VALUES (?, ?, ?, ?, ?)
        ''', (match_no, home_team_id, away_team_id, betting_cycle_date, handicap_value))
        match_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # 存储比赛撤销数据（无时间限制）
        st.session_state.undo_data = {
            "type": "match",
            "match_id": match_id,
            "match_no": match_no,
            "betting_cycle_date": betting_cycle_date
        }

        # 日志与反馈
        log_operation("DATA_INPUT", f"新增比赛：{match_no}（{home_team}vs{away_team}，盘口{handicap_value}）",
                      related_match_id=match_id)
        st.success(f"比赛保存成功！编号：{match_no}（竞彩周期：{betting_cycle_date}）")
        return True
    except Exception as e:
        conn.rollback()
        conn.close()
        st.error(f"保存失败：{str(e)}")
        log_operation("ERROR", f"比赛保存失败：{str(e)}", related_match_id=None)
        return False


def undo_last_match():
    """撤销最近一次比赛+盘口录入（无时间限制）"""
    undo_data = st.session_state.undo_data
    if not undo_data or undo_data["type"] != "match":
        st.error("无可用比赛撤销操作")
        return False

    match_id = undo_data["match_id"]
    conn = get_db_connection()
    try:
        # 检查该比赛是否已关联预测/赛果
        has_pred = conn.execute("SELECT COUNT(*) as c FROM prediction WHERE match_id = ?", (match_id,)).fetchone()[
                       "c"] > 0
        has_result = conn.execute("SELECT COUNT(*) as c FROM result WHERE match_id = ?", (match_id,)).fetchone()[
                         "c"] > 0

        if has_pred or has_result:
            st.error("❌ 该比赛已关联预测/赛果，无法撤销")
            conn.close()
            return False

        # 删除比赛记录
        conn.execute("DELETE FROM match WHERE match_id = ?", (match_id,))
        conn.commit()
        conn.close()

        # 清空撤销数据
        st.session_state.undo_data = None

        log_operation("UNDO", f"撤销比赛：编号{undo_data['match_no']}（{undo_data['betting_cycle_date']}）",
                      related_match_id=match_id)
        st.success(f"撤销成功！已删除比赛：编号{undo_data['match_no']}")
        return True
    except Exception as e:
        conn.rollback()
        conn.close()
        st.error(f"比赛撤销失败：{str(e)}")
        log_operation("ERROR", f"比赛撤销失败：{str(e)}", related_match_id=match_id)
        return False


# -------------------------- Tab2：预测信息录入 函数 --------------------------
def get_existing_predictors():
    """获取所有已存在的预测者名称"""
    conn = get_db_connection()
    if not conn:
        return []
    predictors = conn.execute("SELECT predictor_name FROM predictor").fetchall()
    conn.close()
    return [row["predictor_name"] for row in predictors]


def get_or_create_predictor_id(predictor_name):
    """查找预测者ID，不存在则创建"""
    if not predictor_name.strip():
        st.error("预测者名称不能为空")
        return None
    conn = get_db_connection()
    if not conn:
        return None
    predictor = conn.execute("SELECT predictor_id FROM predictor WHERE predictor_name = ?",
                             (predictor_name,)).fetchone()
    if predictor:
        conn.close()
        return predictor["predictor_id"]
    # 创建新预测者
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictor (predictor_name, total_predictions, total_hits)
        VALUES (?, 0, 0)
    ''', (predictor_name,))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    log_operation("DATA_INPUT", f"新增预测者：{predictor_name}", related_predictor_id=new_id)
    st.success(f"已新增预测者：{predictor_name}")
    return new_id


def get_matches_by_date(betting_date):
    """按竞彩日期筛选比赛（用于下拉选择）"""
    conn = get_db_connection()
    if not conn:
        return {}
    matches = conn.execute('''
        SELECT m.match_id, m.match_no, h.team_name as home_name, a.team_name as away_name, m.handicap_value
        FROM match m
        JOIN team h ON m.home_team_id = h.team_id
        JOIN team a ON m.away_team_id = a.team_id
        WHERE m.betting_cycle_date = ?
        ORDER BY m.match_no
    ''', (betting_date,)).fetchall()
    conn.close()
    match_dict = {}
    for m in matches:
        display_text = f"{m['match_no']} {m['home_name']}vs{m['away_name']}（盘口：{m['handicap_value']}）"
        match_dict[display_text] = m['match_id']
    return match_dict


def is_duplicate_prediction(predictor_id, match_id, prediction_type, original_term):
    """检查同一预测者+比赛+预测类型+预测内容是否完全重复"""
    conn = get_db_connection()
    if not conn:
        return False
    count = conn.execute('''
        SELECT COUNT(*) as c FROM prediction 
        WHERE predictor_id = ? AND match_id = ? AND prediction_type = ? AND original_term = ?
    ''', (predictor_id, match_id, prediction_type, original_term)).fetchone()["c"]
    conn.close()
    return count > 0


def save_prediction(predictor_id, match_id, prediction_type, prediction_str):
    """
    保存预测信息到数据库。
    :param predictor_id: 预测者ID（必填）
    :param match_id: 比赛ID（必填）
    :param prediction_type: 预测类型（胜平负/让球胜平负/总进球数/半全场）
    :param prediction_str: 格式化后的预测字符串（如"胜/让平"、"0球/3球"）
    :return: 是否保存成功
    """
    try:
        conn = get_db_connection()
        if not conn:
            st.error("数据库连接失败")
            return False
        cursor = conn.cursor()

        # 检查完全重复预测（预测者+比赛+类型+内容）
        cursor.execute(
            """
            SELECT prediction_id FROM prediction 
            WHERE predictor_id = ? AND match_id = ? AND prediction_type = ? AND original_term = ?
            """,
            (predictor_id, match_id, prediction_type, prediction_str)
        )
        if cursor.fetchone():
            st.warning("⚠️ 已存在完全相同的预测，无需重复保存")
            conn.close()
            return False

        # 插入新预测
        cursor.execute(
            """
            INSERT INTO prediction (predictor_id, match_id, prediction_type, original_term)
            VALUES (?, ?, ?, ?)
            """,
            (predictor_id, match_id, prediction_type, prediction_str)
        )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        st.error(f"❌ 保存预测失败：{str(e)}")
        return False


def undo_last_prediction():
    """撤销最近一次预测录入（无时间限制）"""
    undo_data = st.session_state.undo_data
    if not undo_data or undo_data["type"] != "prediction":
        st.error("无可用预测撤销操作")
        return False

    pred_id = undo_data["pred_id"]
    predictor_id = undo_data["predictor_id"]
    has_result = undo_data["has_result"]

    conn = get_db_connection()
    try:
        # 查询预测是否命中（用于回滚命中数）
        pred = conn.execute('''
            SELECT original_term, match_id FROM prediction WHERE prediction_id = ?
        ''', (pred_id,)).fetchone()
        if not pred:
            st.error("预测记录不存在")
            conn.close()
            return False

        # 删除预测记录
        conn.execute("DELETE FROM prediction WHERE prediction_id = ?", (pred_id,))

        # 若已录赛果（说明已更新过预测次数和命中数），需回滚
        if has_result:
            # 获取赛果数据用于判断是否命中
            match_id = pred["match_id"]
            res = conn.execute('''
                SELECT home_goals, away_goals, half_full_result, goal_diff FROM result WHERE match_id = ?
            ''', (match_id,)).fetchone()
            handicap = conn.execute("SELECT handicap_value FROM match WHERE match_id = ?", (match_id,)).fetchone()

            if res and handicap:
                actual_goal_diff = res["goal_diff"]
                away_goals = res["away_goals"]
                half_full_result = res["half_full_result"]
                handicap_value = handicap["handicap_value"]

                # 判断是否命中
                hit = judge_prediction_hit(pred["original_term"], actual_goal_diff, handicap_value, away_goals,
                                           half_full_result)

                # 回滚统计
                if hit:
                    conn.execute('''
                        UPDATE predictor 
                        SET total_predictions = total_predictions - 1,
                            total_hits = total_hits - 1
                        WHERE predictor_id = ?
                    ''', (predictor_id,))
                else:
                    conn.execute('''
                        UPDATE predictor 
                        SET total_predictions = total_predictions - 1
                        WHERE predictor_id = ?
                    ''', (predictor_id,))

        conn.commit()
        conn.close()

        # 清空撤销数据
        st.session_state.undo_data = None

        log_operation("UNDO", f"撤销预测：预测ID{pred_id}，预测者{predictor_id}",
                      related_prediction_id=pred_id, related_predictor_id=predictor_id)
        st.success("撤销成功！已删除该预测记录")
        return True
    except Exception as e:
        conn.rollback()
        conn.close()
        st.error(f"预测撤销失败：{str(e)}")
        log_operation("ERROR", f"预测撤销失败：{str(e)}", related_prediction_id=pred_id)
        return False


# -------------------------- Tab3：赛果录入 函数 --------------------------
def is_result_exists(match_id):
    """检查比赛是否已录入赛果"""
    conn = get_db_connection()
    if not conn:
        return False
    count = conn.execute("SELECT COUNT(*) as c FROM result WHERE match_id = ?", (match_id,)).fetchone()["c"]
    conn.close()
    return count > 0


def calculate_result_derivatives(home_goals, away_goals, half_home_goals, half_away_goals):
    """计算赛果衍生字段"""
    # 全场结果
    if home_goals > away_goals:
        full_time_result = "胜"
    elif home_goals == away_goals:
        full_time_result = "平"
    else:
        full_time_result = "负"
    goal_diff = home_goals - away_goals
    total_goals = home_goals + away_goals
    result_detail = f"{home_goals}:{away_goals} {full_time_result}"

    # 上半场结果（必填）
    if half_home_goals > half_away_goals:
        half_time_result = "胜"
    elif half_home_goals == half_away_goals:
        half_time_result = "平"
    else:
        half_time_result = "负"
    half_full_result = f"{half_time_result}{full_time_result}"

    return {
        "full_time_result": full_time_result,
        "half_time_result": half_time_result,
        "half_full_result": half_full_result,
        "goal_diff": goal_diff,
        "total_goals": total_goals,
        "result_detail": result_detail
    }


def judge_prediction_hit(original_term, actual_goal_diff, handicap_value, away_goals, half_full_result):
    """判断预测是否命中（适配多类型）"""
    terms = [t.strip() for t in original_term.split("/")]
    hit = False
    let_goals = -handicap_value  # 转换为实际让球数（主队让球为正）
    home_goals = actual_goal_diff + away_goals  # 主队进球数 = 净胜球 + 客队进球数
    actual_total_goals = home_goals + away_goals  # 总进球数 = 主队 + 客队

    for term in terms:
        if term == "胜":
            hit = hit or (actual_goal_diff > 0)
        elif term == "平":
            hit = hit or (actual_goal_diff == 0)
        elif term == "负":
            hit = hit or (actual_goal_diff < 0)
        elif term == "让胜":
            hit = hit or (actual_goal_diff > let_goals)
        elif term == "让平":
            hit = hit or (actual_goal_diff == let_goals)
        elif term == "让负":
            hit = hit or (actual_goal_diff < let_goals)
        # 总进球数判断
        elif "球" in term or term == "7+":  # 匹配"0球"、"3球"、"7+球"
            # 清理术语：去掉"球"字，保留核心值（如"0球"→"0"，"7+球"→"7+"）
            clean_term = term.replace("球", "").strip()
            if clean_term == "7+":
                # "7+球" → 实际总进球数 ≥7 算命中
                hit = hit or (actual_total_goals >= 7)
            else:
                # 其他情况（如"0"、"3"）→ 实际总进球数 == 数值 算命中
                try:
                    target_goals = int(clean_term)
                    hit = hit or (actual_total_goals == target_goals)
                except ValueError:
                    # 异常格式（如乱输）跳过，不影响其他判断
                    continue
        # 半全场判断
        elif len(term) == 2:
            hit = hit or (term == half_full_result)
    return hit


# -------------------------- Tab3：赛果录入 函数 --------------------------
def save_result(match_id, home_goals, away_goals, half_home_goals, half_away_goals):
    """保存赛果+自动更新预测者统计 + 模型统计（含置信度分桶 & Top2 命中状态同步）"""
    # 验证上半场进球数合法性
    if half_home_goals > home_goals or half_away_goals > away_goals:
        st.error("上半场进球数不能超过全场进球数")
        return False

    # 计算衍生字段
    derivatives = calculate_result_derivatives(home_goals, away_goals, half_home_goals, half_away_goals)
    actual_goal_diff = derivatives["goal_diff"]
    half_full_result = derivatives["half_full_result"]

    # 获取比赛盘口
    conn = get_db_connection()
    if not conn:
        return False
    handicap = conn.execute("SELECT handicap_value FROM match WHERE match_id = ?", (match_id,)).fetchone()
    if not handicap:
        st.error("该比赛无盘口信息，无法判断让球预测")
        conn.close()
        return False
    handicap_value = handicap["handicap_value"]
    conn.close()

    # ---------------------- 内部工具函数：更新某一统计表（适配模型“置信度分桶”表结构） ----------------------
    def _update_bucket_table(inner_conn, table_name, prediction_type, bucket_name, hit_flag):
        """
        更新指定统计表（按“模型整体”维度聚合）：
        - 以 predictor_id = -1 作为“模型整体”的虚拟预测者ID
        - 按 (predictor_id, prediction_type, bucket_name) 聚合
        - total_predictions += 1
        - total_hits += (1 if hit_flag else 0)
        - accuracy = total_hits / total_predictions
        """
        cur = inner_conn.cursor()
        model_pid = -1  # 虚拟ID：代表“模型整体表现”

        # 如果统计表不存在，则直接跳过（不影响赛果保存主流程）
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table_name,)
        )
        if cur.fetchone() is None:
            return

        # 查询是否已有该 (模型ID, 玩法类型, 分桶名) 的统计行
        cur.execute(
            f"""
            SELECT id, total_predictions, total_hits
            FROM {table_name}
            WHERE predictor_id = ? AND prediction_type = ? AND bucket_name = ?
            LIMIT 1
            """,
            (model_pid, prediction_type, bucket_name)
        )
        row = cur.fetchone()

        if row:
            rec_id = row["id"]
            total_pred = row["total_predictions"] + 1
            total_hit = row["total_hits"] + (1 if hit_flag else 0)
            acc = total_hit / total_pred if total_pred > 0 else 0.0
            cur.execute(
                f"""
                UPDATE {table_name}
                SET total_predictions = ?, total_hits = ?, accuracy = ?, last_update_time = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (total_pred, total_hit, acc, rec_id)
            )
        else:
            # 若不存在记录，则插入首行
            total_pred = 1
            total_hit = 1 if hit_flag else 0
            acc = total_hit / total_pred if total_pred > 0 else 0.0
            cur.execute(
                f"""
                INSERT INTO {table_name}
                (predictor_id, prediction_type, bucket_name, total_predictions, total_hits, accuracy)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (model_pid, prediction_type, bucket_name, total_pred, total_hit, acc)
            )

    # ---------------------- 开始事务（确保预测者统计和模型统计原子性）----------------------
    conn = get_db_connection()
    try:
        # 检查是否已存在赛果
        if is_result_exists(match_id):
            st.error("该比赛已录入赛果，不可重复录入")
            conn.close()
            return False

        cursor = conn.cursor()

        # 插入赛果
        cursor.execute(
            '''
            INSERT INTO result (
                match_id, home_goals, away_goals, half_time_home_goals, half_time_away_goals,
                full_time_result, half_time_result, half_full_result, goal_diff, total_goals, result_detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                match_id, home_goals, away_goals, half_home_goals, half_away_goals,
                derivatives["full_time_result"], derivatives["half_time_result"],
                derivatives["half_full_result"], derivatives["goal_diff"],
                derivatives["total_goals"], derivatives["result_detail"]
            )
        )
        result_id = cursor.lastrowid

        # ---------------------- 原有逻辑：更新预测者统计 ----------------------
        predictions = conn.execute(
            '''
            SELECT prediction_id, predictor_id, original_term FROM prediction WHERE match_id = ?
            ''',
            (match_id,)
        ).fetchall()
        hit_pred_ids = []
        for pred in predictions:
            pred_id = pred["prediction_id"]
            predictor_id = pred["predictor_id"]
            original_term = pred["original_term"]

            # 判断是否命中（支持胜平负/让球/总进球数/半全场）
            hit = judge_prediction_hit(
                original_term,
                derivatives["goal_diff"],
                handicap_value,
                away_goals,
                half_full_result
            )

            if hit:
                cursor.execute(
                    '''
                    UPDATE predictor 
                    SET total_predictions = total_predictions + 1,
                        total_hits = total_hits + 1
                    WHERE predictor_id = ?
                    ''',
                    (predictor_id,)
                )
                hit_pred_ids.append(pred_id)
            else:
                cursor.execute(
                    '''
                    UPDATE predictor 
                    SET total_predictions = total_predictions + 1
                    WHERE predictor_id = ?
                    ''',
                    (predictor_id,)
                )

        # ---------------------- 新增逻辑：更新模型统计 ----------------------
        # 1. 取出该比赛所有模型预测记录（未结算的），包含 confidence
        model_preds = conn.execute(
            '''
            SELECT id, original_term, prediction_type, confidence 
            FROM model_prediction_records 
            WHERE match_id = ? AND is_hit IS NULL
            ''',
            (match_id,)
        ).fetchall()

        model_hit_count = 0  # 模型预测命中数
        model_total_count = len(model_preds)  # 模型预测总数

        if model_total_count > 0:
            for model_pred in model_preds:
                mp_id = model_pred["id"]
                original_term = model_pred["original_term"]
                prediction_type = model_pred["prediction_type"]
                confidence = model_pred["confidence"]

                # 判定是否命中
                hit = judge_prediction_hit(
                    original_term,
                    derivatives["goal_diff"],
                    handicap_value,
                    away_goals,
                    half_full_result
                )
                hit_int = 1 if hit else 0
                model_hit_count += hit_int

                # 1) 更新 model_prediction_records 的 is_hit
                cursor.execute(
                    '''
                    UPDATE model_prediction_records 
                    SET is_hit = ? 
                    WHERE id = ?
                    ''',
                    (hit_int, mp_id)
                )

                # 2) 根据置信度更新对应区间统计表（模型整体维度）
                if confidence is not None:
                    # 置信度区间划分：[0.5,0.6), [0.6,0.7), [0.7,0.8), [0.8,0.9), [0.9,1.0]
                    bucket_table = None
                    bucket_name = None
                    if 0.5 <= confidence < 0.6:
                        bucket_table = "model_pred_stats_conf_05_06"
                        bucket_name = "[0.5,0.6)"
                    elif 0.6 <= confidence < 0.7:
                        bucket_table = "model_pred_stats_conf_06_07"
                        bucket_name = "[0.6,0.7)"
                    elif 0.7 <= confidence < 0.8:
                        bucket_table = "model_pred_stats_conf_07_08"
                        bucket_name = "[0.7,0.8)"
                    elif 0.8 <= confidence < 0.9:
                        bucket_table = "model_pred_stats_conf_08_09"
                        bucket_name = "[0.8,0.9)"
                    elif 0.9 <= confidence <= 1.0:
                        bucket_table = "model_pred_stats_conf_09_10"
                        bucket_name = "[0.9,1.0]"

                    if bucket_table is not None and bucket_name is not None:
                        _update_bucket_table(conn, bucket_table, prediction_type, bucket_name, hit)

                # 3) 同步更新 Top2 记录的命中状态
                #    注意：Top2 记录由训练脚本在推理阶段写入，这里仅负责根据赛果回填 is_hit。
                cursor.execute(
                    '''
                    UPDATE model_pred_stats_top2
                    SET is_hit = ?
                    WHERE match_id = ?
                      AND prediction_type = ?
                      AND original_term = ?
                      AND is_hit IS NULL
                    ''',
                    (hit_int, match_id, prediction_type, original_term)
                )

            # 3. 更新模型总历史统计（total_predictions + total_hits + accuracy）
            cursor.execute(
                '''
                SELECT total_predictions, total_hits FROM model_historical_stats LIMIT 1
                '''
            )
            model_stats = cursor.fetchone()
            if model_stats:
                new_total_pred = model_stats["total_predictions"] + model_total_count
                new_total_hit = model_stats["total_hits"] + model_hit_count
            else:
                # 理论上不会发生，防御性处理
                new_total_pred = model_total_count
                new_total_hit = model_hit_count

            new_accuracy = new_total_hit / new_total_pred if new_total_pred > 0 else 0.0

            cursor.execute(
                '''
                UPDATE model_historical_stats 
                SET total_predictions = ?, total_hits = ?, accuracy = ?, last_update_time = CURRENT_TIMESTAMP
                WHERE id = 1
                ''',
                (new_total_pred, new_total_hit, new_accuracy)
            )

        # ---------------------- 提交事务 ----------------------
        conn.commit()
        conn.close()

        # 存储赛果撤销数据（新增模型预测相关信息，用于撤销回滚）
        st.session_state.undo_data = {
            "type": "result",
            "match_id": match_id,
            "result_id": result_id,
            "hit_pred_ids": hit_pred_ids,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "predictions": [(p["prediction_id"], p["predictor_id"]) for p in predictions],
            "model_preds": [p["id"] for p in model_preds],  # 模型预测记录ID列表
            "model_hit_count": model_hit_count,  # 本次模型命中数
            "model_total_count": model_total_count  # 本次模型预测数
        }

        # 日志与反馈
        log_operation("DATA_INPUT", f"新增赛果：比赛{match_id}，比分{home_goals}:{away_goals}", related_match_id=match_id)
        if model_total_count > 0:
            st.success(f"赛果保存成功！模型预测同步更新：新增{model_total_count}次预测，命中{model_hit_count}次")
        else:
            st.success(f"赛果保存成功！{derivatives['result_detail']}，半全场：{half_full_result}")
        return True
    except Exception as e:
        conn.rollback()
        conn.close()
        st.error(f"赛果保存失败：{str(e)}")
        log_operation("ERROR", f"赛果保存失败：{str(e)}", related_match_id=match_id)
        return False


def undo_last_operation():
    """撤销上一次赛果保存操作（同步回滚预测者统计 + 模型统计 & Top2 命中状态）"""
    undo_data = st.session_state.undo_data
    if not undo_data or undo_data["type"] != "result":
        st.error("无可用赛果撤销操作")
        return False

    match_id = undo_data["match_id"]
    result_id = undo_data["result_id"]
    hit_pred_ids = undo_data["hit_pred_ids"]
    all_predictions = undo_data["predictions"]
    model_pred_ids = undo_data.get("model_preds", [])  # 模型预测记录ID
    model_hit_count = undo_data.get("model_hit_count", 0)  # 本次模型命中数
    model_total_count = undo_data.get("model_total_count", 0)  # 本次模型预测数

    conn = get_db_connection()
    try:
        # 删除赛果记录
        conn.execute("DELETE FROM result WHERE result_id = ?", (result_id,))

        # ---------------------- 原有逻辑：回滚预测者统计 ----------------------
        for pred_id, predictor_id in all_predictions:
            if pred_id in hit_pred_ids:
                conn.execute('''
                    UPDATE predictor 
                    SET total_predictions = total_predictions - 1,
                        total_hits = total_hits - 1
                    WHERE predictor_id = ?
                ''', (predictor_id,))
            else:
                conn.execute('''
                    UPDATE predictor 
                    SET total_predictions = total_predictions - 1
                    WHERE predictor_id = ?
                ''', (predictor_id,))

        # ---------------------- 新增逻辑：回滚模型统计 ----------------------
        if model_total_count > 0 and len(model_pred_ids) > 0:
            # 1. 重置模型预测记录的命中状态（恢复为未结算）
            conn.executemany('''
                UPDATE model_prediction_records 
                SET is_hit = NULL 
                WHERE id = ?
            ''', [(pid,) for pid in model_pred_ids])

            # 2. 回滚模型历史统计
            cursor = conn.cursor()
            cursor.execute('''
                SELECT total_predictions, total_hits FROM model_historical_stats LIMIT 1
            ''')
            model_stats = cursor.fetchone()
            new_total_pred = max(0, model_stats["total_predictions"] - model_total_count)
            new_total_hit = max(0, model_stats["total_hits"] - model_hit_count)
            new_accuracy = new_total_hit / new_total_pred if new_total_pred > 0 else 0.0

            cursor.execute('''
                UPDATE model_historical_stats 
                SET total_predictions = ?, total_hits = ?, accuracy = ?, last_update_time = CURRENT_TIMESTAMP
                WHERE id = 1
            ''', (new_total_pred, new_total_hit, new_accuracy))

            # 3. 回滚 Top2 记录的命中状态（如果该场比赛曾入选 Top2）
            conn.execute(
                '''
                UPDATE model_pred_stats_top2
                SET is_hit = NULL
                WHERE match_id = ?
                ''',
                (match_id,)
            )

        # 提交事务
        conn.commit()
        conn.close()

        # 清空撤销数据
        st.session_state.undo_data = None

        # 日志与反馈
        log_operation("UNDO", f"撤销赛果：比赛{match_id}，比分{undo_data['home_goals']}:{undo_data['away_goals']}",
                      related_match_id=match_id)
        if model_total_count > 0:
            st.success(f"撤销成功！已回滚模型统计：减少{model_total_count}次预测，减少{model_hit_count}次命中，并重置相关 Top2 命中状态")
        else:
            st.success("撤销成功！已恢复到赛果录入前状态")
        return True
    except Exception as e:
        conn.rollback()
        conn.close()
        st.error(f"赛果撤销失败：{str(e)}")
        log_operation("ERROR", f"赛果撤销失败：{str(e)}", related_match_id=match_id)
        return False


# -------------------------- Tab4：数据查询验证 函数 --------------------------
def query_data_by_date(betting_date, show_no_result=False):
    """按日期查询比赛+预测+赛果数据（包含预测类型）"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    try:
        # 基础查询：包含预测类型
        query = '''
            SELECT 
                m.match_id, m.match_no, h.team_name as home_team, a.team_name as away_team,
                m.handicap_value, m.betting_cycle_date,
                r.home_goals, r.away_goals, r.half_full_result, r.result_detail,
                p.predictor_id, pred.predictor_name, p.original_term, p.prediction_type
            FROM match m
            JOIN team h ON m.home_team_id = h.team_id
            JOIN team a ON m.away_team_id = a.team_id
            LEFT JOIN result r ON m.match_id = r.match_id
            LEFT JOIN prediction p ON m.match_id = p.match_id
            LEFT JOIN predictor pred ON p.predictor_id = pred.predictor_id
            WHERE m.betting_cycle_date = ?
        '''
        params = (betting_date,)

        # 筛选未录入赛果的比赛
        if show_no_result:
            query += " AND r.match_id IS NULL"

        df = pd.read_sql(query, conn, params=params)
        conn.close()

        # 处理命中状态
        def get_hit_status(row):
            if pd.isna(row["result_detail"]) or pd.isna(row["original_term"]):
                return "待赛果"
            actual_goal_diff = (row["home_goals"] - row["away_goals"]) if not pd.isna(row["home_goals"]) else 0
            handicap_value = row["handicap_value"]
            return "命中" if judge_prediction_hit(row["original_term"], actual_goal_diff, handicap_value,
                                                  row["away_goals"], row["half_full_result"]) else "未命中"

        df["命中状态"] = df.apply(get_hit_status, axis=1)

        # 格式化输出列（新增预测类型）
        output_cols = [
            "betting_cycle_date", "match_no", "home_team", "away_team", "handicap_value",
            "predictor_name", "prediction_type", "original_term", "result_detail", "half_full_result", "命中状态"
        ]
        return df[output_cols].fillna("无")
    except Exception as e:
        conn.close()
        st.error(f"查询失败：{str(e)}")
        return pd.DataFrame()


# -------------------------- 主页面布局 --------------------------
def main():
    st.set_page_config(page_title="比赛数据录入系统（整合版）", layout="wide")

    # 初始化prediction表（首次运行自动更新）
    init_prediction_table()

    # 顶部环境标识
    env_color = "#28a745" if CURRENT_ENV == "prod" else "#ffc107"
    env_label = "🚀 生产环境" if CURRENT_ENV == "prod" else "🔧 开发环境"
    st.markdown(f"<h3 style='color:{env_color};text-align:center'>{env_label}</h3>", unsafe_allow_html=True)
    st.title("⚽ 足球比赛数据录入系统")

    # 多Tab设计
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. 比赛+盘口录入",
        "2. 预测信息录入",
        "3. 赛果录入",
        "4. 数据查询验证"
    ])

    # -------------------------- Tab1：比赛+盘口录入 --------------------------
    with tab1:
        st.subheader("比赛基础信息+盘口录入")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            # 字段输入
            match_no = st.text_input("比赛编号", placeholder="请输入3位数字（如001）")
            betting_cycle_date = st.date_input(
                "竞彩周期日期",
                value=st.session_state.default_betting_date,
                help="凌晨比赛（如11.12 02:00）属于11.11的竞彩周期，请选择11.11"
            )

            # 主队+模糊匹配
            existing_teams = get_existing_teams()
            home_team = st.text_input("主队名称", placeholder="如：罗马")
            if home_team and home_team in existing_teams:
                st.info(f"✅ 已匹配到球队：{home_team}")

            # 客队+自动排除主队
            away_team_options = [t for t in existing_teams if t != home_team]
            away_team = st.text_input("客队名称", placeholder="如：本菲卡")
            if away_team and away_team in away_team_options:
                st.info(f"✅ 已匹配到球队：{away_team}")
            elif away_team == home_team:
                st.warning("⚠️ 主队和客队不能相同")

            # 盘口输入（支持任意非0整数）
            handicap_value = st.number_input(
                "让球数（盘口）",
                step=1, help="主队让球为负数（如-1），主队受让为正数（如+1），不能为0"
            )

            # 保存+撤销按钮
            col_save, col_undo = st.columns(2)
            with col_save:
                if st.button("保存比赛+盘口", type="primary", use_container_width=True):
                    if home_team.strip() == "" or away_team.strip() == "":
                        st.error("❌ 主队和客队名称不能为空")
                    elif home_team == away_team:
                        st.error("❌ 主队和客队不能相同")
                    elif is_match_exists(match_no, betting_cycle_date):
                        st.error(f"❌ 该竞彩周期（{betting_cycle_date}）已存在编号{match_no}")
                    else:
                        save_match_and_handicap(match_no, betting_cycle_date, home_team, away_team, handicap_value)
            with col_undo:
                if st.button("撤销上一次比赛录入", type="secondary", use_container_width=True):
                    undo_last_match()

        with col2:
            st.info("""
            ### 录入说明
            1. **比赛编号**：3位数字（001-099），第一位为0，同一竞彩周期内不可重复
            2. **竞彩周期日期**：
               - 默认当天日期
               - 凌晨比赛（如11.12 00:30）请选择11.11
            3. **球队名称**：
               - 输入时自动匹配已有球队
               - 新球队会自动创建
               - 主队和客队不能相同
            4. **盘口规则**：
               - 支持任意非0整数（无最大限制）
               - 主队让球：负数（如-1=主队让1球）
               - 主队受让：正数（如+1=主队受让1球）
            5. **撤销功能**：
               - 仅可撤销最近一次比赛录入
               - 已关联预测/赛果的比赛无法撤销
            """)

            # 显示现有球队
            with st.expander("查看现有球队", expanded=False):
                if existing_teams:
                    st.write(f"共{len(existing_teams)}支球队：")
                    st.write(", ".join(existing_teams))
                else:
                    st.write("暂无已录入球队")

    # -------------------------- Tab2：预测信息录入 --------------------------
    with tab2:
        st.subheader("预测信息录入")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            # 预测者选择（模糊匹配）
            existing_predictors = get_existing_predictors()
            predictor_name = st.text_input("预测者名称", placeholder="输入名称（支持模糊匹配）")

            # 模糊匹配下拉
            matched_predictors = [p for p in existing_predictors if
                                  predictor_name.lower() in p.lower()] if predictor_name else []
            if matched_predictors:
                selected_predictor = st.selectbox("匹配到的预测者", matched_predictors, index=None)
                if selected_predictor:
                    predictor_name = selected_predictor

            # 比赛选择（按日期筛选）
            pred_betting_date = st.date_input("竞彩周期日期", value=st.session_state.default_betting_date)
            match_dict = get_matches_by_date(pred_betting_date)
            if not match_dict:
                st.warning(f"该日期无已录入比赛")
                selected_match_text = None
            else:
                selected_match_text = st.selectbox(
                    "选择比赛", list(match_dict.keys()), index=None, placeholder="请选择比赛"
                )
            match_id = match_dict[selected_match_text] if selected_match_text else None

            # 预测方式选择
            prediction_type = st.radio(
                "预测方式",
                ["胜平负/让球胜平负", "总进球数", "半全场"],
                index=0
            )

            # -------------------------- 核心：统一生成格式化预测字符串 --------------------------
            prediction_str = ""  # 最终传给数据库的格式化字符串
            selected_count = 0  # 统计选中项数量（用于限制最大选择数）

            if prediction_type == "胜平负/让球胜平负":
                st.markdown("#### 选择预测结果（最多2项）")
                col_pred1, col_pred2 = st.columns(2)
                options = []
                with col_pred1:
                    if st.checkbox("胜", key="no1"): options.append("胜")
                    if st.checkbox("平", key="no2"): options.append("平")
                    if st.checkbox("负", key="no3"): options.append("负")
                with col_pred2:
                    if st.checkbox("让胜", key="h1"): options.append("让胜")
                    if st.checkbox("让平", key="h2"): options.append("让平")
                    if st.checkbox("让负", key="h3"): options.append("让负")
                selected_count = len(options)
                if selected_count > 0:
                    prediction_str = "/".join(options)  # 格式："胜/让平"

            elif prediction_type == "总进球数":
                st.subheader("选择可能的总进球数 (可多选)")
                col_tg1, col_tg2, col_tg3 = st.columns(3)
                goal_options = ["0球", "1球", "2球", "3球", "4球", "5球", "6球", "7+球"]
                options = []
                with col_tg1:
                    if st.checkbox(goal_options[0], key="tg_0"): options.append(goal_options[0])
                    if st.checkbox(goal_options[3], key="tg_3"): options.append(goal_options[3])
                    if st.checkbox(goal_options[6], key="tg_6"): options.append(goal_options[6])
                with col_tg2:
                    if st.checkbox(goal_options[1], key="tg_1"): options.append(goal_options[1])
                    if st.checkbox(goal_options[4], key="tg_4"): options.append(goal_options[4])
                    if st.checkbox(goal_options[7], key="tg_7"): options.append(goal_options[7])
                with col_tg3:
                    if st.checkbox(goal_options[2], key="tg_2"): options.append(goal_options[2])
                    if st.checkbox(goal_options[5], key="tg_5"): options.append(goal_options[5])
                selected_count = len(options)
                if selected_count > 0:
                    prediction_str = "/".join(options)  # 格式："0球/3球/7+球"
                else:
                    st.warning("⚠️ 请至少选择一个总进球数选项")

            elif prediction_type == "半全场":
                st.markdown("#### 选择半全场结果（最多3项）")
                col_ht1, col_ht2, col_ht3 = st.columns(3)
                options = []
                with col_ht1:
                    if st.checkbox("胜胜", key="ht1"): options.append("胜胜")
                    if st.checkbox("平胜", key="ht4"): options.append("平胜")
                    if st.checkbox("负胜", key="ht7"): options.append("负胜")
                with col_ht2:
                    if st.checkbox("胜平", key="ht2"): options.append("胜平")
                    if st.checkbox("平平", key="ht5"): options.append("平平")
                    if st.checkbox("负平", key="ht8"): options.append("负平")
                with col_ht3:
                    if st.checkbox("胜负", key="ht3"): options.append("胜负")
                    if st.checkbox("平负", key="ht6"): options.append("平负")
                    if st.checkbox("负负", key="ht9"): options.append("负负")
                selected_count = len(options)
                if selected_count > 0:
                    prediction_str = "/".join(options)  # 格式："胜胜/平胜"

            # -------------------------- 保存+撤销按钮 --------------------------
            col_save, col_undo = st.columns(2)
            with col_save:
                if st.button("保存预测", type="primary", use_container_width=True):
                    # 验证逻辑
                    if not predictor_name.strip():
                        st.error("❌ 请输入预测者名称")
                    elif not match_id:
                        st.error("❌ 请选择有效比赛")
                    elif selected_count == 0:
                        st.error("❌ 请选择至少1项预测结果")
                    elif (prediction_type == "胜平负/让球胜平负" and selected_count > 2) or \
                            (prediction_type == "半全场" and selected_count > 3):
                        max_opt = 2 if prediction_type == "胜平负/让球胜平负" else 3
                        st.error(f"❌ 最多选择{max_opt}项")
                    else:
                        # 获取/创建预测者ID
                        predictor_id = get_or_create_predictor_id(predictor_name)
                        if predictor_id:
                            # 修正：传格式化后的字符串prediction_str，而非列表
                            success = save_prediction(
                                predictor_id=predictor_id,
                                match_id=match_id,
                                prediction_type=prediction_type,
                                prediction_str=prediction_str
                            )
                            if success:
                                st.success(f"✅ 预测保存成功：{prediction_type} - {prediction_str}")
                with col_undo:
                    if st.button("撤销上一次预测", type="secondary", use_container_width=True):
                        undo_last_prediction()

            # -------------------------- 右侧说明（更新总进球数描述） --------------------------
            with col2:
                st.info("""
                ### 录入说明
                1. **预测者名称**：
                   - 输入时自动模糊匹配已有预测者
                   - 新预测者会自动创建
                2. **比赛选择**：
                   - 先选择竞彩周期日期
                   - 仅显示该日期下已录入的比赛
                3. **预测方式规则**：
                   - 胜平负/让球胜平负：最多2项（支持平/让平双选）
                   - 总进球数：可多选独立进球数（如0球/3球/7+球）
                   - 半全场：最多3项（按3×3布局排列）
                4. **重复预测**：
                   - 同一预测者+同一场比赛+同一类型+同一内容：不可重复
                   - 同一预测者+同一场比赛+不同类型：可并存
                5. **统计规则**：
                   - 预测保存时不更新统计次数
                   - 赛果录入后，按命中情况更新（命中：次数+1、命中数+1；未命中：仅次数+1）
                6. **撤销功能**：
                   - 仅可撤销最近一次预测录入
                   - 已关联赛果的预测撤销会同步回滚统计次数
                """)

                # 显示现有预测者
                with st.expander("查看现有预测者", expanded=False):
                    if existing_predictors:
                        st.write(f"共{len(existing_predictors)}位预测者：")
                        st.write(", ".join(existing_predictors))
                    else:
                        st.write("暂无已录入预测者")

    # -------------------------- Tab3：赛果录入 --------------------------
    with tab3:
        st.subheader("赛果录入（上半场比分必填）")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            # 比赛选择（按日期筛选）
            res_betting_date = st.date_input("竞彩周期日期", value=st.session_state.default_betting_date,
                                             key="res_date")
            res_match_dict = get_matches_by_date(res_betting_date)

            if not res_match_dict:
                st.warning(f"该日期无已录入比赛")
                selected_res_match_text = None
            else:
                selected_res_match_text = st.selectbox(
                    "选择比赛", list(res_match_dict.keys()), index=None, placeholder="请选择已结束的比赛"
                )
            match_id = res_match_dict[selected_res_match_text] if selected_res_match_text else None

            # 比分录入
            if match_id:
                st.markdown("#### 全场比分")
                col_goals1, col_goals2 = st.columns(2)
                with col_goals1:
                    home_goals = st.number_input("主队进球数", min_value=0, step=1, key="home_goals")
                with col_goals2:
                    away_goals = st.number_input("客队进球数", min_value=0, step=1, key="away_goals")

                st.markdown("#### 上半场比分（必填）")
                col_half1, col_half2 = st.columns(2)
                with col_half1:
                    half_home_goals = st.number_input(
                        "上半场主队进球数", min_value=0, max_value=home_goals, step=1, key="half_home"
                    )
                with col_half2:
                    half_away_goals = st.number_input(
                        "上半场客队进球数", min_value=0, max_value=away_goals, step=1, key="half_away"
                    )

                # 保存+撤销按钮
                col_save, col_undo = st.columns(2)
                with col_save:
                    if st.button("保存赛果", type="primary", use_container_width=True):
                        save_result(match_id, home_goals, away_goals, half_home_goals, half_away_goals)
                with col_undo:
                    if st.button("撤销上一次赛果", type="secondary", use_container_width=True):
                        undo_last_operation()
            else:
                st.info("请先选择比赛")

        with col2:
            st.info("""
            ### 录入说明
            1. **比赛选择**：
               - 先选择竞彩周期日期
               - 仅显示该日期下已录入的比赛
            2. **比分规则**：
               - 进球数必须是非负整数
               - 上半场进球数不能超过全场进球数
               - 一场比赛只能录入一次赛果
            3. **自动计算**：
               - 全场赛果（主胜/平/主负）
               - 半全场赛果（如胜胜、平负）
               - 总进球数、净胜球
            4. **撤销功能**：
               - 仅可撤销最近一次赛果录入
               - 撤销后会同步回滚所有关联预测的统计次数
            """)

    # -------------------------- Tab4：数据查询验证 --------------------------
    with tab4:
        st.subheader("数据查询验证")
        st.markdown("---")

        # 筛选条件
        col_query1, col_query2 = st.columns(2)
        with col_query1:
            query_date = st.date_input("竞彩周期日期", value=st.session_state.default_betting_date, key="query_date")
        with col_query2:
            show_no_result = st.checkbox("仅显示未录入赛果的比赛")

        # 查询按钮
        if st.button("执行查询", use_container_width=True):
            result_df = query_data_by_date(query_date, show_no_result)
            if result_df.empty:
                st.info(f"无符合条件的数据")
            else:
                st.subheader(f"查询结果（共{len(result_df)}条记录）")

                # 命中状态颜色格式化
                def highlight_hit(val):
                    if val == "命中":
                        return 'background-color: #d4edda; color: #155724'
                    elif val == "未命中":
                        return 'background-color: #f8d7da; color: #721c24'
                    else:  # 待赛果
                        return 'background-color: #fff3cd; color: #856404'

                # 显示表格（包含预测类型）
                styled_df = result_df.style.applymap(
                    highlight_hit, subset=["命中状态"]
                ).hide(axis="index")
                st.dataframe(styled_df, use_container_width=True)

                # 下载功能
                csv = result_df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label="📥 下载查询结果",
                    data=csv,
                    file_name=f"数据查询结果_{query_date}_{pd.Timestamp.now().strftime('%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.info("""
        ### 查询说明
        1. **筛选条件**：
           - 按竞彩周期日期查询
           - 可选择仅显示未录入赛果的比赛
        2. **结果字段**：
           - 比赛信息：编号、对阵、盘口
           - 预测信息：预测者、预测类型、预测结果
           - 赛果信息：全场比分、半全场
           - 命中状态：绿色=命中，红色=未命中，黄色=待赛果
        3. **下载功能**：
           - 支持下载CSV格式文件
           - 文件名包含查询日期和时间
        """)


if __name__ == "__main__":
    main()