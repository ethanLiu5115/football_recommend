import streamlit as st
import sqlite3
import os
from datetime import date

# 数据库连接函数
def get_db_connection():
    """建立与SQLite数据库的连接"""
    db_path = os.path.join("data", "football.db")
    if not os.path.exists(db_path):
        st.error("数据库文件不存在，请先运行db_create.py和db_init.py")
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # 使查询结果可通过列名访问
    return conn

# 获取现有球队列表（用于匹配验证）
def get_existing_teams():
    """从team表获取所有已存在的球队名称"""
    conn = get_db_connection()
    if not conn:
        return []
    teams = conn.execute("SELECT team_name FROM team").fetchall()
    conn.close()
    return [row["team_name"] for row in teams]

# 获取现有赛事列表（用于匹配验证）
def get_existing_leagues():
    """从league表获取所有已存在的赛事名称"""
    conn = get_db_connection()
    if not conn:
        return []
    leagues = conn.execute("SELECT league_name FROM league").fetchall()
    conn.close()
    return [row["league_name"] for row in leagues]

# 获取球队ID（不存在则创建）
def get_or_create_team_id(team_name):
    """
    查找球队ID，若不存在则创建新球队并返回新ID
    :param team_name: 球队名称
    :return: 球队ID（整数）
    """
    conn = get_db_connection()
    if not conn:
        return None
    # 查找现有球队
    team = conn.execute("SELECT team_id FROM team WHERE team_name = ?", (team_name,)).fetchone()
    if team:
        conn.close()
        return team["team_id"]
    # 不存在则创建新球队
    cursor = conn.cursor()
    cursor.execute("INSERT INTO team (team_name) VALUES (?)", (team_name,))
    conn.commit()
    new_team_id = cursor.lastrowid  # 获取刚插入的ID
    conn.close()
    st.success(f"已新增球队：{team_name}")
    return new_team_id

# 获取赛事ID（不存在则创建）
def get_or_create_league_id(league_name):
    """
    查找赛事ID，若不存在则创建新赛事并返回新ID
    :param league_name: 赛事名称
    :return: 赛事ID（整数）
    """
    conn = get_db_connection()
    if not conn:
        return None
    # 查找现有赛事
    league = conn.execute("SELECT league_id FROM league WHERE league_name = ?", (league_name,)).fetchone()
    if league:
        conn.close()
        return league["league_id"]
    # 不存在则创建新赛事
    cursor = conn.cursor()
    cursor.execute("INSERT INTO league (league_name) VALUES (?)", (league_name,))
    conn.commit()
    new_league_id = cursor.lastrowid  # 获取刚插入的ID
    conn.close()
    st.success(f"已新增赛事：{league_name}")
    return new_league_id

# 检查比赛编号是否已存在
def is_match_id_exists(match_no, betting_cycle_date):
    """检查同一竞彩周期内是否存在相同编号"""
    conn = get_db_connection()
    if not conn:
        return False
    count = conn.execute('''
        SELECT COUNT(*) as c FROM match 
        WHERE match_no = ? 
        AND betting_cycle_date = ?  -- 基于竞彩周期日期判断
    ''', (match_no, betting_cycle_date)).fetchone()["c"]
    conn.close()
    return count > 0

# 保存比赛信息到数据库
def save_match_info(match_no, betting_cycle_date, home_team, away_team, league):
    home_team_id = get_or_create_team_id(home_team)
    away_team_id = get_or_create_team_id(away_team)
    league_id = get_or_create_league_id(league)
    if not all([home_team_id, away_team_id, league_id]):
        st.error("关联数据获取失败，无法保存")
        return False

    conn = get_db_connection()
    if not conn:
        return False
    try:
        conn.execute("""
            INSERT INTO match (match_no, home_team_id, away_team_id, league_id, betting_cycle_date)
            VALUES (?, ?, ?, ?, ?)
        """, (match_no, home_team_id, away_team_id, league_id, betting_cycle_date))
        conn.commit()
        conn.close()
        st.success(f"比赛信息保存成功！编号：{match_no}（竞彩周期：{betting_cycle_date}）")
        return True
    except Exception as e:
        conn.rollback()
        conn.close()
        st.error(f"保存失败：{str(e)}")
        return False

# 主界面
def main():
    st.set_page_config(page_title="比赛数据输入", layout="wide")
    st.title("足球比赛智能推荐系统 - 数据输入")
    st.subheader("比赛基础信息（竞彩周期适配）")

    # 初始化默认竞彩周期日期（首次加载为当天，不自动刷新）
    if "default_betting_date" not in st.session_state:
        st.session_state.default_betting_date = date.today()  # 仅日期

    # 分栏布局
    col1, col2 = st.columns(2)
    with col1:
        # 1. 比赛编号（不变）
        match_no = st.text_input("比赛编号", placeholder="请输入3位数字（如006）")

        # 2. 竞彩周期日期（核心修改：用date_input选择日期，无时间）
        betting_cycle_date = st.date_input(
            "竞彩周期日期",
            value=st.session_state.default_betting_date,
            help="凌晨比赛（如11.12 02:00）属于11.11的竞彩周期，请选择11.11"
        )

        # 3. 主队（不变）
        existing_teams = get_existing_teams()
        home_team = st.text_input("主队", placeholder="输入球队名称，如：曼彻斯特城")
        if home_team and home_team in existing_teams:
            st.info(f"已匹配到球队：{home_team}")

        # 4. 客队（不变）
        away_team = st.text_input("客队", placeholder="输入球队名称，如：利物浦")
        if away_team and away_team in existing_teams:
            st.info(f"已匹配到球队：{away_team}")

        # 5. 赛事类型（不变）
        existing_leagues = get_existing_leagues()
        league = st.text_input("赛事类型", placeholder="输入赛事名称，如：英格兰足球超级联赛")
        if league and league in existing_leagues:
            st.info(f"已匹配到赛事：{league}")

        # 保存按钮（验证逻辑调整）
        if st.button("保存比赛信息", use_container_width=True):
            # 验证1：比赛编号是否为3位数字
            if not (match_no.isdigit() and len(match_no) == 3):
                st.error("请输入3位数字编号（如001）")
                return
            # 验证2：主队和客队是否相同
            if home_team == away_team:
                st.error("主队和客队不能相同")
                return
            # 验证3：球队和赛事名称非空
            if not (home_team.strip() and away_team.strip() and league.strip()):
                st.error("主队、客队和赛事名称不能为空")
                return
            # 验证4：“竞彩周期日期+编号”是否已存在
            if is_match_id_exists(match_no, betting_cycle_date):
                st.error(f"该竞彩周期（{betting_cycle_date}）已存在编号{match_no}，请修改")
                return
            # 保存数据
            save_match_info(match_no, betting_cycle_date, home_team, away_team, league)

    with col2:
        st.info("""
        ### 输入说明  
        1. 比赛编号：必须是3位数字（如001-999），同一竞彩周期内不可重复  
        2. 竞彩周期日期：  
           - 直接选择日期（无时间），默认当天  
           - 凌晨比赛规则：如11.12 00:30的比赛属于11.11的竞彩周期，需选择11.11  
        3. 球队/赛事输入：  
           - 输入已存在的名称会显示“已匹配”提示  
           - 输入新名称会自动创建新记录  
        4. 保存后可在match表查看记录（关联竞彩周期日期）  
        """)
        with st.expander("查看现有球队"):
            st.write(existing_teams)
        with st.expander("查看现有赛事"):
            st.write(existing_leagues)

if __name__ == "__main__":
    main()