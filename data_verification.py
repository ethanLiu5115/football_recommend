import sqlite3
import os

# ------------------------------
# 基础数据库连接函数
# ------------------------------
def get_db_connection():
    """建立数据库连接"""
    db_path = os.path.join("data", "football.db")
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------
# 1. 数据一致性验证函数
# ------------------------------
def verify_data_consistency(match_id=None, prediction_id=None, result_id=None):
    """
    验证数据一致性：检查比赛、预测、赛果之间的关联有效性
    :param match_id: 比赛ID（可选）
    :param prediction_id: 预测ID（可选）
    :param result_id: 赛果ID（可选）
    :return: (是否有效, 错误信息)
    """
    conn = get_db_connection()
    if not conn:
        return False, "数据库连接失败"

    try:
        # 验证比赛是否存在
        if match_id:
            match = conn.execute("SELECT 1 FROM match WHERE match_id = ?", (match_id,)).fetchone()
            if not match:
                return False, f"比赛ID {match_id} 不存在"

        # 验证预测是否关联有效比赛
        if prediction_id:
            pred = conn.execute(
                "SELECT match_id FROM prediction WHERE prediction_id = ?",
                (prediction_id,)
            ).fetchone()
            if not pred:
                return False, f"预测ID {prediction_id} 不存在"
            # 检查关联的比赛是否存在
            match = conn.execute("SELECT 1 FROM match WHERE match_id = ?", (pred["match_id"],)).fetchone()
            if not match:
                return False, f"预测ID {prediction_id} 关联的比赛不存在"

        # 验证赛果是否关联有效比赛且未重复
        if result_id:
            res = conn.execute(
                "SELECT match_id FROM result WHERE result_id = ?",
                (result_id,)
            ).fetchone()
            if not res:
                return False, f"赛果ID {result_id} 不存在"
            # 检查关联的比赛是否存在
            match = conn.execute("SELECT 1 FROM match WHERE match_id = ?", (res["match_id"],)).fetchone()
            if not match:
                return False, f"赛果ID {result_id} 关联的比赛不存在"

        return True, "数据一致"
    except Exception as e:
        return False, f"验证失败：{str(e)}"
    finally:
        conn.close()


# ------------------------------
# 2. 预测者数据自动更新
# ------------------------------
def update_predictor_total(predictor_id):
    """
    新增预测后，更新预测者的总预测次数（+1）
    :param predictor_id: 预测者ID
    """
    conn = get_db_connection()
    if not conn:
        return
    try:
        conn.execute("""
            UPDATE predictor 
            SET total_predictions = total_predictions + 1 
            WHERE predictor_id = ?
        """, (predictor_id,))
        conn.commit()
    except Exception as e:
        print(f"更新预测者总次数失败：{str(e)}")
    finally:
        conn.close()


def judge_prediction_hit(prediction_id, result_id):
    """修正让球盘判断逻辑：先转换盘口为实际让球数，再比较"""
    conn = get_db_connection()
    if not conn:
        return False

    try:
        # 1. 获取预测和比赛关联信息
        pred = conn.execute("""
            SELECT p.original_term, p.match_id 
            FROM prediction p 
            WHERE p.prediction_id = ?
        """, (prediction_id,)).fetchone()
        if not pred:
            return False
        original_term = pred["original_term"]
        match_id = pred["match_id"]

        # 2. 获取赛果的实际净胜球
        res = conn.execute("""
            SELECT r.home_goals, r.away_goals, r.goal_diff 
            FROM result r 
            WHERE r.result_id = ? AND r.match_id = ?
        """, (result_id, match_id)).fetchone()
        if not res:
            return False
        actual_goal_diff = res["goal_diff"]  # 实际净胜球（主队-客队）

        # 3. 获取盘口并转换为实际让球数
        # 盘口规则：handicap_value为负数表示主队让球（如-2=主队让2球），正数表示主队受让（如+1=主队受让1球）
        handicap = conn.execute("""
            SELECT handicap_value FROM handicap 
            WHERE match_id = ?
        """, (match_id,)).fetchone()
        if not handicap:
            return False  # 无盘口时无法判断让球术语，直接返回未命中
        handicap_value = handicap["handicap_value"]
        let_goals = -handicap_value  # 转换为实际让球数（主队让球数，正数为让，负数为受让）

        # 4. 解析预测术语
        terms = [t.strip() for t in original_term.split("/")]
        hit = False

        # 5. 修正后的判断规则（重点优化让球逻辑）
        for term in terms:
            if term == "胜":
                # 不让球胜：实际净胜球 > 0
                hit = hit or (actual_goal_diff > 0)
            elif term == "平":
                # 不让球平：实际净胜球 == 0
                hit = hit or (actual_goal_diff == 0)
            elif term == "负":
                # 不让球负：实际净胜球 < 0
                hit = hit or (actual_goal_diff < 0)
            elif term == "让胜":
                # 让球胜：实际净胜球 > 让球数（如让2球，需净胜>2）
                hit = hit or (actual_goal_diff > let_goals)
            elif term == "让平":
                # 让球平：实际净胜球 == 让球数（如让2球，净胜=2）
                hit = hit or (actual_goal_diff == let_goals)
            elif term == "让负":
                # 让球负：实际净胜球 < 让球数（如让2球，净胜<2）
                hit = hit or (actual_goal_diff < let_goals)
            elif term == "双平":
                # 双平：不让球平（净胜0）或让球平（净胜=让球数）
                hit = hit or (actual_goal_diff == 0 or actual_goal_diff == let_goals)

        return hit
    except Exception as e:
        print(f"判断预测命中失败：{str(e)}")
        return False
    finally:
        conn.close()


def update_predictor_hits(match_id, result_id):
    """同步优化赛果更新时的命中判断逻辑"""
    conn = get_db_connection()
    if not conn:
        return
    try:
        # 获取该比赛的所有预测
        predictions = conn.execute("""
            SELECT prediction_id, predictor_id, original_term 
            FROM prediction 
            WHERE match_id = ?
        """, (match_id,)).fetchall()
        if not predictions:
            return

        # 获取赛果的实际净胜球
        res = conn.execute("""
            SELECT home_goals, away_goals, goal_diff 
            FROM result 
            WHERE result_id = ? AND match_id = ?
        """, (result_id, match_id)).fetchone()
        if not res:
            return
        actual_goal_diff = res["goal_diff"]

        # 获取盘口并转换为实际让球数
        handicap = conn.execute("""
            SELECT handicap_value FROM handicap 
            WHERE match_id = ?
        """, (match_id,)).fetchone()
        if not handicap:
            return  # 无盘口时不处理让球预测
        handicap_value = handicap["handicap_value"]
        let_goals = -handicap_value  # 实际让球数（主队让球数）

        # 逐个判断预测是否命中
        for pred in predictions:
            predictor_id = pred["predictor_id"]
            original_terms = [t.strip() for t in pred["original_term"].split("/")]
            hit = False

            for term in original_terms:
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
                elif term == "双平":
                    hit = hit or (actual_goal_diff == 0 or actual_goal_diff == let_goals)

            if hit:
                # 命中则更新预测者的总命中次数
                conn.execute("""
                    UPDATE predictor 
                    SET total_hits = total_hits + 1 
                    WHERE predictor_id = ?
                """, (predictor_id,))
        conn.commit()
    except Exception as e:
        print(f"更新预测者命中次数失败：{str(e)}")
        conn.rollback()
    finally:
        conn.close()


# ------------------------------
# 3. 球队历史战绩同步
# ------------------------------
def insert_team_history(match_id, result_id):
    """修正字段名和补充缺失字段，适配team_history表结构"""
    conn = get_db_connection()
    if not conn:
        return
    try:
        # 1. 获取比赛基本信息（新增league_id，用于填充表中必填的league_id）
        match = conn.execute("""
            SELECT home_team_id, away_team_id, betting_cycle_date, league_id 
            FROM match 
            WHERE match_id = ?
        """, (match_id,)).fetchone()
        if not match:
            return
        home_team_id = match["home_team_id"]
        away_team_id = match["away_team_id"]
        match_date = match["betting_cycle_date"]
        league_id = match["league_id"]  # 表中league_id为非空，必须获取

        # 2. 获取赛果信息
        res = conn.execute("""
            SELECT home_goals, away_goals, full_time_result 
            FROM result 
            WHERE result_id = ? AND match_id = ?
        """, (result_id, match_id)).fetchone()
        if not res:
            return
        home_goals = res["home_goals"]
        away_goals = res["away_goals"]
        full_time = res["full_time_result"]

        # 3. 插入主队历史记录（适配表结构字段）
        home_result = "胜" if full_time == "主胜" else ("平" if full_time == "平" else "负")
        conn.execute("""
            INSERT INTO team_history (
                team_id, 
                opponent_team_id,  -- 修正字段名：opponent_team_id（原opponent_id）
                is_home,  -- 新增：主队为1（主场）
                match_date,
                league_id,  -- 新增：赛事ID（表中必填）
                result,
                goal_scored,  -- 修正字段名：少个s（原goals_scored）
                goal_conceded,  -- 修正字段名：少个s（原goals_conceded）
                history_match_id  -- 修正字段名：用history_match_id关联比赛（原match_id）
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            home_team_id,
            away_team_id,  # 对手球队ID
            1,  # 主队是主场（1=主场）
            match_date,
            league_id,  # 赛事ID
            home_result,
            home_goals,  # 主队进球
            away_goals,  # 主队失球
            match_id  # 关联比赛ID（对应表中history_match_id）
        ))

        # 4. 插入客队历史记录（适配表结构字段）
        away_result = "胜" if full_time == "主负" else ("平" if full_time == "平" else "负")
        conn.execute("""
            INSERT INTO team_history (
                team_id, 
                opponent_team_id,  -- 修正字段名
                is_home,  -- 新增：客队为0（客场）
                match_date,
                league_id,  -- 新增：赛事ID
                result,
                goal_scored,  -- 修正字段名
                goal_conceded,  -- 修正字段名
                history_match_id  -- 修正字段名
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            away_team_id,
            home_team_id,  # 对手球队ID（主队）
            0,  # 客队是客场（0=客场）
            match_date,
            league_id,  # 赛事ID（与主队同一场比赛，赛事相同）
            away_result,
            away_goals,  # 客队进球
            home_goals,  # 客队失球
            match_id  # 关联比赛ID
        ))

        conn.commit()
        print("球队历史战绩插入成功")
    except Exception as e:
        print(f"插入球队历史战绩失败：{str(e)}")
        conn.rollback()
    finally:
        conn.close()


# ------------------------------
# 4. 存储优化主函数（整合更新逻辑）
# ------------------------------
def update_related_tables(action, **kwargs):
    """
    统一调用相关更新函数
    :param action: 操作类型（"save_prediction" 或 "save_result"）
    :param kwargs: 参数（根据action传递，如predictor_id、match_id等）
    """
    if action == "save_prediction":
        # 保存预测后：更新预测者总预测次数
        predictor_id = kwargs.get("predictor_id")
        if predictor_id:
            update_predictor_total(predictor_id)
    elif action == "save_result":
        # 保存赛果后：更新预测者命中次数 + 插入球队历史
        match_id = kwargs.get("match_id")
        result_id = kwargs.get("result_id")
        if match_id and result_id:
            update_predictor_hits(match_id, result_id)
            insert_team_history(match_id, result_id)


# ------------------------------
# 初始化team_history表（首次运行需执行）
# ------------------------------
def init_team_history_table():
    """创建包含match_id字段的球队历史战绩表"""
    conn = get_db_connection()
    if not conn:
        return
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS team_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL,  -- 球队ID
                match_id INTEGER NOT NULL,  -- 关联比赛ID（新增字段）
                opponent_id INTEGER NOT NULL,  -- 对手球队ID
                match_date DATE NOT NULL,  -- 比赛日期（竞彩周期）
                goals_scored INTEGER NOT NULL,  -- 进球数
                goals_conceded INTEGER NOT NULL,  -- 失球数
                result TEXT NOT NULL,  -- 对该球队而言的结果（胜/平/负）
                create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_id) REFERENCES team(team_id) ON DELETE CASCADE,
                FOREIGN KEY (match_id) REFERENCES match(match_id) ON DELETE CASCADE,
                FOREIGN KEY (opponent_id) REFERENCES team(team_id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        print("✅ 已创建 team_history 表（含match_id字段）")
    except Exception as e:
        print(f"创建team_history表失败：{str(e)}")
    finally:
        conn.close()


if __name__ == "__main__":
    # 首次运行时初始化表
    init_team_history_table()