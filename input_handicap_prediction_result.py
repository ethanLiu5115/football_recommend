import streamlit as st
import sqlite3
import os
from data_verification import verify_data_consistency, update_related_tables, update_predictor_total

# ------------------------------
# 数据库连接与基础函数（复用并扩展）
# ------------------------------
def get_db_connection():
    """建立与SQLite数据库的连接"""
    db_path = os.path.join("data", "football.db")
    if not os.path.exists(db_path):
        st.error("数据库文件不存在，请先运行db_create.py和初始化数据")
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_matches_for_selector():
    """获取所有比赛的信息，用于下拉选择（显示日期+编号+对阵）"""
    conn = get_db_connection()
    if not conn:
        return {}
    # 修改SQL查询：将match_datetime替换为betting_cycle_date
    matches = conn.execute('''
        SELECT 
            m.match_id, 
            m.match_no, 
            m.betting_cycle_date as match_date,  -- 替换字段
            h.team_name as home_name,
            a.team_name as away_name
        FROM match m
        JOIN team h ON m.home_team_id = h.team_id
        JOIN team a ON m.away_team_id = a.team_id
        ORDER BY m.betting_cycle_date DESC  -- 替换排序字段
    ''').fetchall()
    conn.close()
    # 格式化显示文本（日期+编号+对阵）
    match_dict = {}
    for m in matches:
        display_text = f"{m['match_date']} {m['match_no']}（{m['home_name']}vs{m['away_name']}）"
        match_dict[display_text] = m['match_id']
    return match_dict

# ------------------------------
# 1. 盘口输入模块相关函数
# ------------------------------
def is_valid_handicap(handicap_str):
    """验证盘口输入是否为整数（支持正负）"""
    if not handicap_str.strip():
        return False, "盘口不能为空"
    try:
        int(handicap_str)
        return True, "有效"
    except ValueError:
        return False, "请输入整数（如-4、0、+3）"

def is_handicap_exists(match_id, handicap_value):
    """检查某场比赛的某个盘口是否已存在（避免重复）"""
    conn = get_db_connection()
    if not conn:
        return False
    count = conn.execute('''
        SELECT COUNT(*) as c FROM handicap 
        WHERE match_id = ? AND handicap_value = ?
    ''', (match_id, handicap_value)).fetchone()["c"]
    conn.close()
    return count > 0

def save_handicap(match_id, handicap_value):
    """保存盘口信息到handicap表"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        conn.execute('''
            INSERT INTO handicap (match_id, handicap_value, verify_status)
            VALUES (?, ?, '一致')  -- verify_status默认"一致"
        ''', (match_id, handicap_value))
        conn.commit()
        conn.close()
        st.success(f"盘口 {handicap_value} 保存成功！")
        return True
    except Exception as e:
        conn.rollback()
        conn.close()
        st.error(f"盘口保存失败：{str(e)}")
        return False

# ------------------------------
# 2. 预测输入模块相关函数
# ------------------------------
def is_valid_prediction_term(term):
    """
    验证预测术语是否合法（支持单术语和双选/多选，用/分隔）
    基础合法术语：让胜、让平、让负、胜、平、负、双平
    组合规则：用/分隔，且每个子术语必须是基础合法术语（如“胜/平”“让胜/让平”）
    """
    base_terms = ["让胜", "让平", "让负", "胜", "平", "负", "双平"]
    if not term.strip():
        return False, "预测术语不能为空"

    # 处理组合术语（用/分割）
    sub_terms = [t.strip() for t in term.split("/")]

    # 检查每个子术语是否合法
    invalid_terms = [t for t in sub_terms if t not in base_terms]
    if invalid_terms:
        return False, f"以下术语不合法：{', '.join(invalid_terms)}，合法术语：{', '.join(base_terms)}"

    # 特殊规则：“双平”通常不与其他术语组合（可选限制，根据实际需求调整）
    if "双平" in sub_terms and len(sub_terms) > 1:
        return False, "“双平”为独立术语，不建议与其他术语组合"

    return True, "有效"


def get_or_create_predictor_id(predictor_name):
    """获取预测者ID，不存在则创建"""
    if not predictor_name.strip():
        st.error("预测者名称不能为空")
        return None
    conn = get_db_connection()
    if not conn:
        return None
    # 查找现有预测者
    predictor = conn.execute('''
        SELECT predictor_id FROM predictor 
        WHERE predictor_name = ?
    ''', (predictor_name,)).fetchone()
    if predictor:
        conn.close()
        return predictor["predictor_id"]
    # 创建新预测者
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictor (predictor_name, total_predictions, total_hits)
        VALUES (?, 0, 0)  -- 初始统计为0
    ''', (predictor_name,))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    st.success(f"新增预测者：{predictor_name}")
    return new_id


def save_prediction(match_id, predictor_id, original_term):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        # 1. 数据一致性验证（检查比赛是否存在）
        is_valid, msg = verify_data_consistency(match_id=match_id)
        if not is_valid:
            st.error(f"数据验证失败：{msg}")
            return False

        # 2. 插入预测记录
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prediction (match_id, predictor_id, original_term, translated_result)
            VALUES (?, ?, ?, NULL)
        ''', (match_id, predictor_id, original_term))
        conn.commit()

        # 3. 强制触发预测者总次数更新（直接调用函数，避免中间层问题）
        update_predictor_total(predictor_id)

        st.success(f"预测 [{original_term}] 保存成功！")
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"保存失败：{str(e)}")
        return False
    finally:
        conn.close()

# ------------------------------
# 3. 赛果输入模块相关函数
# ------------------------------
def is_result_exists(match_id):
    """检查某场比赛是否已存在赛果（避免重复）"""
    conn = get_db_connection()
    if not conn:
        return False
    count = conn.execute('''
        SELECT COUNT(*) as c FROM result 
        WHERE match_id = ?
    ''', (match_id,)).fetchone()["c"]
    conn.close()
    return count > 0

def calculate_result_info(home_goals, away_goals):
    """根据比分自动计算胜平负、净胜球和详细描述"""
    # 计算净胜球
    goal_diff = home_goals - away_goals
    # 计算全场赛果
    if home_goals > away_goals:
        full_time_result = "主胜"
    elif home_goals == away_goals:
        full_time_result = "平"
    else:
        full_time_result = "主负"
    # 生成详细描述（比分+赛果）
    result_detail = f"{home_goals}:{away_goals} {full_time_result}"
    return full_time_result, goal_diff, result_detail

def save_result(match_id, home_goals, away_goals):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        # 1. 验证数据一致性（检查比赛是否存在且未重复录入赛果）
        is_valid, msg = verify_data_consistency(match_id=match_id)
        if not is_valid:
            st.error(f"数据验证失败：{msg}")
            return False
        if is_result_exists(match_id):
            st.error("该比赛已录入赛果，不可重复录入")
            return False

        # 2. 插入赛果记录
        full_time_result, goal_diff, result_detail = calculate_result_info(home_goals, away_goals)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO result (
                match_id, home_goals, away_goals,
                full_time_result, goal_diff, result_detail
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (match_id, home_goals, away_goals, full_time_result, goal_diff, result_detail))
        conn.commit()
        result_id = cursor.lastrowid

        # 3. 触发预测者命中更新和球队历史同步
        update_related_tables(
            action="save_result",
            match_id=match_id,
            result_id=result_id
        )

        st.success(f"赛果保存成功！比分：{home_goals}:{away_goals}（{full_time_result}）")
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"保存失败：{str(e)}")
        return False
    finally:
        conn.close()

# ------------------------------
# 主界面（Tabs整合）
# ------------------------------
def main():
    st.set_page_config(page_title="比赛数据输入 - 扩展模块", layout="wide")
    st.title("足球比赛智能推荐系统 - 数据输入")
    st.subheader("盘口、预测与赛果管理")

    # 获取比赛列表（用于下拉选择）
    match_dict = get_matches_for_selector()
    if not match_dict:
        st.warning("暂无比赛数据，请先在「比赛基础信息」界面录入比赛")
        return

    # 用Tabs分模块
    tab1, tab2, tab3 = st.tabs(["盘口输入", "预测输入", "赛果输入"])

    # ------------------------------
    # 1. 盘口输入Tab
    # ------------------------------
    with tab1:
        st.info("### 盘口信息输入（关联已录入的比赛）")
        col1, col2 = st.columns(2)
        with col1:
            # 选择比赛
            selected_match_text = st.selectbox(
                "关联比赛（日期+编号+对阵）",
                list(match_dict.keys()),
                index=None,
                placeholder="选择需要录入盘口的比赛"
            )
            # 整数盘口（文本输入）
            handicap_input = st.text_input(
                "整数盘口",
                placeholder="请输入整数（如-4、0、+3）"
            )

            if st.button("保存盘口", use_container_width=True):
                # 验证
                if not selected_match_text:
                    st.error("请选择关联比赛")
                    return
                is_valid, msg = is_valid_handicap(handicap_input)
                if not is_valid:
                    st.error(msg)
                    return
                handicap_value = int(handicap_input)
                # 获取match_id
                match_id = match_dict[selected_match_text]
                # 检查重复
                if is_handicap_exists(match_id, handicap_value):
                    st.error(f"该比赛的 {handicap_value} 盘口已存在")
                    return
                # 保存
                save_handicap(match_id, handicap_value)

        with col2:
            st.info("""
            ### 盘口输入说明  
            1. 需先在「比赛基础信息」界面录入比赛，才能选择关联比赛  
            2. 盘口格式：**整数**（支持正负，如-4、0、+3）  
            3. 同一比赛的同一盘口只能保存一次  
            4. 保存后可在handicap表查看记录  
            """)

    # ------------------------------
    # 2. 预测输入Tab
    # ------------------------------
    with tab2:
        st.info("### 预测方案输入（支持单选/双选/多选）")
        col1, col2 = st.columns(2)
        with col1:
            # 选择比赛
            selected_match_text_pred = st.selectbox(
                "关联比赛（日期+编号+对阵）",
                list(match_dict.keys()),
                index=None,
                placeholder="选择需要预测的比赛",
                key="pred_match_updated"
            )

            # 预测者输入
            predictor_name = st.text_input("预测者名称", placeholder="输入预测者姓名或代号")

            # 预测术语（文本输入，支持组合）
            prediction_term = st.text_input(
                "预测术语",
                placeholder="输入术语（单选如“让平”，双选如“胜/平”，用/分隔）"
            )

            if st.button("保存预测", use_container_width=True):
                # 验证
                if not selected_match_text_pred:
                    st.error("请选择关联比赛")
                    return
                if not predictor_name.strip():
                    st.error("请输入预测者名称")
                    return
                is_valid, msg = is_valid_prediction_term(prediction_term)
                if not is_valid:
                    st.error(msg)
                    return

                # 保存
                match_id_pred = match_dict[selected_match_text_pred]
                predictor_id = get_or_create_predictor_id(predictor_name)
                if predictor_id:
                    save_prediction(match_id_pred, predictor_id, prediction_term)

        with col2:
            st.info("""
            ### 预测输入说明  
            1. 基础合法术语（7种）：  
               - 让球盘：让胜、让平、让负  
               - 不让球盘：胜、平、负  
               - 特殊：双平（平+让平）  

            2. 支持组合输入（用/分隔）：  
               - 双选：如“胜/平”“让平/让负”  
               - 多选：如“胜/平/负”（最多建议3个，避免无意义组合）  

            3. 规则限制：  
               - “双平”为独立术语，不建议与其他术语组合  
               - 组合术语需用半角/分隔（如“胜/平”而非“胜 平”）  

            4. 保存后可在prediction表查看原始输入术语  
            """)

    # ------------------------------
    # 3. 赛果输入Tab
    # ------------------------------
    with tab3:
        st.info("### 比赛结果输入（直接录入比分，系统自动计算其他信息）")
        col1, col2 = st.columns(2)
        with col1:
            # 选择比赛
            selected_match_text_res = st.selectbox(
                "关联比赛（日期+编号+对阵）",
                list(match_dict.keys()),
                index=None,
                placeholder="选择已结束的比赛",
                key="res_match"
            )

            # 比分输入（分开主队和客队进球数，确保非负整数）
            col_goals1, col_goals2 = st.columns(2)
            with col_goals1:
                home_goals_input = st.text_input("主队进球数", placeholder="如2")
            with col_goals2:
                away_goals_input = st.text_input("客队进球数", placeholder="如1")

            if st.button("保存赛果", use_container_width=True):
                # 验证1：选择比赛
                if not selected_match_text_res:
                    st.error("请选择关联比赛")
                    return
                # 验证2：进球数为非负整数
                try:
                    home_goals = int(home_goals_input.strip())
                    away_goals = int(away_goals_input.strip())
                    if home_goals < 0 or away_goals < 0:
                        st.error("进球数不能为负数")
                        return
                except ValueError:
                    st.error("请输入有效的非负整数（如0、2、5）")
                    return
                # 验证3：是否已存在赛果
                match_id_res = match_dict[selected_match_text_res]
                if is_result_exists(match_id_res):
                    st.error("该比赛已录入赛果，不可重复录入")
                    return
                # 保存赛果
                save_result(match_id_res, home_goals, away_goals)

        with col2:
            st.info("""
            ### 赛果输入说明  
            1. 直接录入真实比分（主队进球数:客队进球数），例如：  
               - 主队2球、客队1球 → 输入“2”和“1”  
               - 平局（如1:1）→ 输入“1”和“1”  
               - 主队0球、客队3球 → 输入“0”和“3”  

            2. 系统会自动计算：  
               - 全场赛果（主胜/平/主负）  
               - 净胜球（主队进球数 - 客队进球数）  
               - 详细描述（如“2:1 主胜”）  

            3. 约束：  
               - 进球数必须是非负整数（0、1、2...）  
               - 一场比赛只能录入一次赛果  

            4. 保存后可在result表查看完整信息（含具体比分）
            """)

if __name__ == "__main__":
    main()