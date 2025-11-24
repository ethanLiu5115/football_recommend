import pandas as pd
import numpy as np
import sqlite3
import warnings
import os
import sys

# 1. 配置项目根目录（确保能找到 config 文件夹）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 2. 环境适配（和 model_train_app.py 一致）
CURRENT_ENV = os.getenv("FOOTBALL_ENV", "dev")
# 3. 导入对应环境的 DB_PATH（关键：让函数能直接用 DB_PATH）
if CURRENT_ENV == "prod":
    from config.prod_config import DB_PATH
else:
    from config.dev_config import DB_PATH

warnings.filterwarnings('ignore')


# ===================== 1. 数据库工具函数 =====================
def get_db_connection(db_path):
    """创建数据库连接"""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        print(f"数据库连接失败：{str(e)}")
    return conn


def load_historical_data(db_path, start_date=None, end_date=None):
    """加载历史数据（含比赛+预测+赛果+预测者）- 用于训练"""
    conn = get_db_connection(db_path)
    if not conn:
        return pd.DataFrame()

    sql = '''
        SELECT 
            p.prediction_id, p.match_id, p.predictor_id, p.original_term, 
            p.prediction_type, p.predict_time,
            m.betting_cycle_date, m.handicap_value,
            r.home_goals, r.away_goals, r.half_full_result, r.goal_diff, r.total_goals,
            pr.total_predictions, pr.total_hits
        FROM prediction p
        JOIN match m ON p.match_id = m.match_id
        JOIN result r ON p.match_id = r.match_id
        JOIN predictor pr ON p.predictor_id = pr.predictor_id
    '''
    if start_date and end_date:
        sql += f" WHERE m.betting_cycle_date BETWEEN '{start_date}' AND '{end_date}'"

    try:
        df = pd.read_sql(sql, conn)
    except Exception as e:
        print(f"执行SQL查询时发生错误: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    # 数据类型转换
    df['betting_cycle_date'] = pd.to_datetime(df['betting_cycle_date'])
    df['predict_time'] = pd.to_datetime(df['predict_time'])
    df['handicap_value'] = df['handicap_value'].fillna(0)

    return df


def load_prediction_data(db_path, target_date):
    """加载指定日期的预测数据（无赛果）+ 历史数据（用于计算准确率）- 用于推理"""
    conn = get_db_connection(db_path)
    if not conn:
        return pd.DataFrame()

    # 1. 加载目标日期的待推理数据
    pred_sql = '''
        SELECT 
            p.prediction_id, p.match_id, p.predictor_id, p.original_term, 
            p.prediction_type, p.predict_time,
            m.betting_cycle_date, m.handicap_value,
            pr.total_predictions, pr.total_hits
        FROM prediction p
        JOIN match m ON p.match_id = m.match_id
        JOIN predictor pr ON p.predictor_id = pr.predictor_id
        WHERE m.betting_cycle_date = ?
        AND NOT EXISTS (SELECT 1 FROM result r WHERE r.match_id = p.match_id)
    '''

    # 2. 加载目标日期之前的历史数据（含赛果，用于计算最近3/5次准确率和命中率）
    history_sql = '''
        SELECT 
            p.predictor_id, p.prediction_type, p.predict_time, p.original_term,
            m.handicap_value,
            r.home_goals, r.away_goals, r.half_full_result, r.goal_diff, r.total_goals
        FROM prediction p
        JOIN match m ON p.match_id = m.match_id
        JOIN result r ON p.match_id = r.match_id
        WHERE m.betting_cycle_date < ?
    '''

    # 执行查询
    df_pred = pd.read_sql(pred_sql, conn, params=(target_date,))
    df_history = pd.read_sql(history_sql, conn, params=(target_date,))
    conn.close()

    # 数据类型转换
    df_pred['betting_cycle_date'] = pd.to_datetime(df_pred['betting_cycle_date'])
    df_pred['predict_time'] = pd.to_datetime(df_pred['predict_time'])
    df_pred['handicap_value'] = df_pred['handicap_value'].fillna(0)

    # 3. 处理历史数据：计算命中标签
    if not df_history.empty:
        df_history['is_hit'] = df_history.apply(
            lambda x: judge_prediction_hit(
                x['original_term'], x['prediction_type'],
                x['goal_diff'], x['handicap_value'],
                x['total_goals'], x['half_full_result']
            ), axis=1
        )
        df_history = df_history.sort_values(['predictor_id', 'predict_time']).reset_index(drop=True)

    # 4. 临时存储历史数据（供feature_engineering调用）
    df_pred['_history_data'] = [df_history.copy() for _ in range(len(df_pred))]

    return df_pred


# ===================== 2. 核心辅助函数（预测判断+编码） =====================
def judge_prediction_hit(original_term, prediction_type, goal_diff, handicap_value, total_goals, half_full_result):
    """完整命中判断逻辑（含让球胜平负/总进球数/半全场）"""
    terms = [t.strip() for t in original_term.split("/")]
    hit = False

    if prediction_type == "胜平负/让球胜平负":
        for term in terms:
            if term == "胜":
                hit = hit or (goal_diff > 0)
            elif term == "平":
                hit = hit or (goal_diff == 0)
            elif term == "负":
                hit = hit or (goal_diff < 0)
            elif term == "让胜":
                hit = hit or (goal_diff > handicap_value)
            elif term == "让平":
                hit = hit or (goal_diff == handicap_value)
            elif term == "让负":
                hit = hit or (goal_diff < handicap_value)
    elif prediction_type == "总进球数":
        for term in terms:
            clean_term = term.replace("球", "").strip()
            if clean_term == "7+":
                hit = hit or (total_goals >= 7)
            else:
                try:
                    target = int(clean_term)
                    hit = hit or (total_goals == target)
                except ValueError:
                    continue
    elif prediction_type == "半全场":
        for term in terms:
            if len(term) == 2 and term in ["胜胜", "胜平", "胜负", "平胜", "平平", "平负", "负胜", "负平", "负负"]:
                hit = hit or (term == half_full_result)
    return hit


def calculate_recent_acc(df_history, predictor_id, window):
    """计算单个预测者最近N次预测准确率（无数据返回np.nan，由特征阶段统一填默认值）"""
    # 第一步：判断df_history是否为空，或是否包含必要列
    if df_history.empty or 'predictor_id' not in df_history.columns:
        return np.nan

    # 第二步：筛选当前预测者的记录
    pred_history = df_history[df_history['predictor_id'] == predictor_id].copy()
    if pred_history.empty:
        return np.nan

    # 第三步：按时间排序，取最近window条记录
    pred_history = pred_history.sort_values('predict_time', ascending=False).head(window)
    if pred_history.empty:
        return np.nan

    # 第四步：计算准确率（确保is_hit列存在）
    if 'is_hit' not in pred_history.columns:
        return np.nan

    recent_acc = pred_history['is_hit'].mean()
    return recent_acc if not np.isnan(recent_acc) else np.nan


def encode_prediction_scheme(original_term, prediction_type):
    """
    预测方案编码（按三种类型分别实现）
    :return: 编码向量（numpy数组）
    """
    terms = [t.strip() for t in original_term.split("/")]

    if prediction_type == "胜平负/让球胜平负":
        # 编码顺序：[胜, 平, 负, 让胜, 让平, 让负]（6维One-Hot）
        encode = np.zeros(6)
        term_map = {"胜": 0, "平": 1, "负": 2, "让胜": 3, "让平": 4, "让负": 5}
        for term in terms:
            if term in term_map:
                encode[term_map[term]] = 1
        return encode

    elif prediction_type == "总进球数":
        # 编码顺序：[0球,1球,2球,3球,4球,5球,6球,7+球]（8维多标签）
        encode = np.zeros(8)
        term_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7+": 7}
        for term in terms:
            clean_term = term.replace("球", "").strip()
            if clean_term in term_map:
                encode[term_map[clean_term]] = 1
        return encode

    elif prediction_type == "半全场":
        # 编码顺序：[胜胜,胜平,胜负,平胜,平平,平负,负胜,负平,负负]（9维One-Hot）
        encode = np.zeros(9)
        term_map = {
            "胜胜": 0, "胜平": 1, "胜负": 2,
            "平胜": 3, "平平": 4, "平负": 5,
            "负胜": 6, "负平": 7, "负负": 8
        }
        for term in terms:
            if term in term_map:
                encode[term_map[term]] = 1
        return encode

    else:
        # 未知类型返回全0向量（避免报错）
        return np.zeros(6)


def get_predictor_performance_vec(df_valid, df_history, predictor_id, is_training):
    """计算单个预测者的历史表现向量（4维）- 稳健版"""
    # 第一步：验证传入的df_valid是否包含关键列（明确报错位置）
    required_cols = ['predictor_id', 'total_predictions', 'total_hits']
    missing_cols = [col for col in required_cols if col not in df_valid.columns]
    if missing_cols:
        raise KeyError(f"函数内缺少关键列：{missing_cols}（传入的df_valid列名：{df_valid.columns.tolist()}）")

    # 第二步：稳健筛选当前预测者的记录（避免链式索引风险）
    # 用布尔索引筛选，确保结果非空
    mask = df_valid['predictor_id'] == predictor_id
    pred_data = df_valid[mask].copy()  # 用copy避免SettingWithCopyWarning

    if pred_data.empty:
        return np.array([0.1, 0.1, 0.1, 0.1])

    # 第三步：稳健获取total_predictions和total_hits（避免iloc[0]隐性报错）
    total_pred = pred_data['total_predictions'].values[0]  # 比iloc[0]更稳健
    total_hits = pred_data['total_hits'].values[0]

    # 1. 历史总准确率
    historical_acc = total_hits / total_pred if total_pred > 0 else 0.1

    # 2. 总预测数（归一化）
    max_pred = df_valid['total_predictions'].max() if df_valid['total_predictions'].max() > 0 else 1
    norm_total_pred = total_pred / max_pred

    # 3. 最近3次准确率
    recent3_acc = calculate_recent_acc(df_history, predictor_id, window=3)

    # 4. 最近5次准确率
    recent5_acc = calculate_recent_acc(df_history, predictor_id, window=5)

    return np.array([historical_acc, norm_total_pred, recent3_acc, recent5_acc])


# ===================== 3. 特征工程公共函数（核心修改） =====================
def calculate_prediction_consistency(group):
    """计算预测一致性（与高表现预测者重合度）"""
    # 修正：用已有列计算历史准确率（total_hits / total_predictions）
    group['predictor_historical_acc'] = group['total_hits'] / group['total_predictions'].replace(0, 1)

    # 高表现预测者：历史准确率≥0.5 且 总预测数≥100
    high_perf_mask = (group['predictor_historical_acc'] >= 0.5) & (group['total_predictions'] >= 100)
    if high_perf_mask.sum() == 0:
        all_terms = group['original_term'].str.split('/').explode()
        top_terms = all_terms.value_counts().head(3).index.tolist()
    else:
        high_score_terms = group[high_perf_mask]['original_term'].str.split('/').explode()
        top_terms = high_score_terms.unique()

    def get_overlap_rate(term):
        user_terms = term.split('/')
        overlap = len(set(user_terms) & set(top_terms))
        return overlap / len(user_terms) if len(user_terms) > 0 else 0

    group['prediction_consistency'] = group['original_term'].apply(get_overlap_rate)
    return group


def calculate_type_hit_rate(df):
    """计算同类型预测历史命中率（中性值0.1）"""
    # 1. 全量预测者的同类型命中率
    global_type_hit = df.groupby('prediction_type')['is_hit'].mean().to_dict()
    df['global_type_hit_rate'] = df['prediction_type'].map(global_type_hit).fillna(0.1)

    # 2. 个人同类型命中率
    personal_type_hit = df.groupby(['predictor_id', 'prediction_type'])['is_hit'].mean().reset_index()
    personal_type_hit.columns = ['predictor_id', 'prediction_type', 'personal_type_hit_rate']
    df = df.merge(personal_type_hit, on=['predictor_id', 'prediction_type'], how='left')
    df['personal_type_hit_rate'] = df['personal_type_hit_rate'].fillna(0.1)

    # 融合得分
    df['type_hit_rate'] = (df['global_type_hit_rate'] + df['personal_type_hit_rate']) / 2
    return df


def calculate_match_pred_density(group):
    """计算同比赛预测密度（参与预测的有效预测者人数）"""
    pred_count = group['predictor_id'].nunique()
    group['match_pred_density'] = pred_count
    return group


def feature_engineering(df, is_training=True):
    """特征工程：以稳定的人维度统计量为中心，压缩预测者信息为少量、平滑、可泛化的数值特征"""
    df = df.copy()
    df = df.reset_index(drop=True)

    # ================== 1. 预测者 ID & 历史数据准备 ==================
    # 当前窗口中的预测者
    current_predictor_ids = df['predictor_id'].dropna().unique().tolist()

    # 从 predictor 表中补全历史预测者（保证与历史模型兼容）
    conn = get_db_connection(DB_PATH)
    if conn:
        try:
            all_pids = conn.execute("SELECT DISTINCT predictor_id FROM predictor").fetchall()
            all_pids = [p['predictor_id'] for p in all_pids]
            current_predictor_ids = list(set(current_predictor_ids + all_pids))
        finally:
            conn.close()
    current_predictor_ids.sort()

    # 历史数据：
    #   - 推理阶段：load_prediction_data 已经把历史放在 _history_data 里
    #   - 训练阶段：用到当前窗口结束日为止的全量历史
    if (not df.empty) and (not is_training) and ('_history_data' in df.columns):
        df_history = df['_history_data'].iloc[0].copy()
    else:
        end_date = None
        if not df.empty and 'betting_cycle_date' in df.columns:
            end_ts = pd.to_datetime(df['betting_cycle_date']).max()
            if pd.notna(end_ts):
                end_date = end_ts.strftime('%Y-%m-%d')
        df_history = load_historical_data(DB_PATH, start_date=None, end_date=end_date)

    # ================== 2. 构造“人维度”统计特征 ==================
    # 我们只保留少量平滑统计量：整体命中率 / 最近3&5次命中率 / 总预测次数归一化 / 按玩法命中率等
    overall_hit_map = {}
    overall_recent3_map = {}
    overall_recent5_map = {}
    overall_total_norm_map = {}
    type_hit_map = {}
    type_cnt_norm_map = {}
    global_type_hit_map = {}

    if df_history is not None and not df_history.empty:
        df_hist = df_history.copy()

        # 时间字段统一
        if 'predict_time' in df_hist.columns:
            df_hist['predict_time'] = pd.to_datetime(df_hist['predict_time'])

        # 兜底：若无 is_hit，则基于赛果计算
        if 'is_hit' not in df_hist.columns:
            df_hist['is_hit'] = df_hist.apply(
                lambda row: judge_prediction_hit(
                    original_term=row['original_term'],
                    prediction_type=row['prediction_type'],
                    goal_diff=row['goal_diff'],
                    handicap_value=row.get('handicap_value'),
                    total_goals=row.get('total_goals'),
                    half_full_result=row.get('half_full_result')
                ),
                axis=1
            ).astype(int)

        # 2.1 预测者整体统计（不区分玩法）
        grouped = df_hist.groupby('predictor_id')

        # 整体命中率
        overall_hit_map = grouped['is_hit'].mean().to_dict()
        # 历史预测次数（用于归一化）
        total_cnt_map = grouped.size().to_dict()
        max_total_cnt = max(total_cnt_map.values()) if len(total_cnt_map) > 0 else 1.0
        overall_total_norm_map = {pid: cnt / max_total_cnt for pid, cnt in total_cnt_map.items()}

        # 最近3/5次命中率（以最后 N 条为准）
        for pid, g in grouped:
            g = g.sort_values('predict_time')
            vals = g['is_hit'].values
            if len(vals) == 0:
                r3 = np.nan
                r5 = np.nan
            else:
                r3 = vals[-3:].mean() if len(vals) >= 1 else np.nan
                r5 = vals[-5:].mean() if len(vals) >= 1 else np.nan
            overall_recent3_map[pid] = float(r3) if not np.isnan(r3) else np.nan
            overall_recent5_map[pid] = float(r5) if not np.isnan(r5) else np.nan

        # 2.2 “预测者×玩法”统计
        if 'prediction_type' in df_hist.columns:
            type_group = df_hist.groupby(['predictor_id', 'prediction_type'])
            # 按玩法的个人命中率 & 预测次数
            tmp_type_hit = type_group['is_hit'].mean()
            tmp_type_cnt = type_group.size()

            type_hit_map = tmp_type_hit.to_dict()
            max_type_cnt = max(tmp_type_cnt.values) if len(tmp_type_cnt) > 0 else 1.0
            type_cnt_norm_map = {k: v / max_type_cnt for k, v in tmp_type_cnt.to_dict().items()}

            # 2.3 全局玩法命中率
            global_type_hit_map = df_hist.groupby('prediction_type')['is_hit'].mean().to_dict()

    # ================== 3. 将人维度统计映射到当前窗口的每条记录 ==================
    df_feat = df.copy()

    def _map_overall(map_dict, pid):
        return map_dict.get(pid, np.nan)

    def _map_type(map_dict, pid, ptype):
        return map_dict.get((pid, ptype), np.nan)

    df_feat['pred_overall_acc'] = df_feat['predictor_id'].map(lambda x: _map_overall(overall_hit_map, x))
    df_feat['pred_recent3_acc'] = df_feat['predictor_id'].map(lambda x: _map_overall(overall_recent3_map, x))
    df_feat['pred_recent5_acc'] = df_feat['predictor_id'].map(lambda x: _map_overall(overall_recent5_map, x))
    df_feat['pred_total_pred_norm'] = df_feat['predictor_id'].map(lambda x: _map_overall(overall_total_norm_map, x))

    df_feat['pred_type_hit_rate'] = df_feat.apply(
        lambda row: _map_type(type_hit_map, row['predictor_id'], row['prediction_type']),
        axis=1
    )
    df_feat['pred_type_total_pred_norm'] = df_feat.apply(
        lambda row: _map_type(type_cnt_norm_map, row['predictor_id'], row['prediction_type']),
        axis=1
    )

    df_feat['global_type_hit_rate'] = df_feat['prediction_type'].map(global_type_hit_map)

    # 缺失值统一填为中性值 0.1
    human_cols = [
        'pred_overall_acc', 'pred_recent3_acc', 'pred_recent5_acc',
        'pred_total_pred_norm', 'pred_type_hit_rate',
        'pred_type_total_pred_norm', 'global_type_hit_rate'
    ]
    for col in human_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0.1
        df_feat[col] = df_feat[col].fillna(0.1)

    # ================== 4. 预测一致性 & 比赛预测密度 ==================
    # 4.1 预测一致性：根据当前窗口内高表现预测者的方案重合度
    df_feat = calculate_prediction_consistency(df_feat)

    # 4.2 比赛预测密度：每场参与预测的预测者数量
    if 'match_id' in df_feat.columns:
        density_map = df_feat.groupby('match_id')['predictor_id'].nunique().to_dict()
        df_feat['match_pred_density'] = df_feat['match_id'].map(density_map).astype(float)
    else:
        df_feat['match_pred_density'] = 1.0

    # ================== 5. 玩法类型编码（全局） ==================
    type_map = {"胜平负/让球胜平负": 1, "总进球数": 2, "半全场": 3}
    df_feat['pred_type_identifier'] = df_feat['prediction_type'].map(type_map).fillna(1).astype(float)

    # ===== 新增：总进球数组合宽度和覆盖特征 =====
    def parse_goal_combo_stats(row):
        # Only applies to 总进球数
        if row['prediction_type'] != "总进球数":
            return 0.0, 0.0, 0.0
        terms = [t.strip() for t in str(row['original_term']).split('/')]
        goals = set()
        for term in terms:
            # Remove trailing "球" if present
            clean = term
            if clean.endswith("球"):
                clean = clean[:-1]
            clean = clean.strip()
            if clean == "7+":
                goals.add(7)
            else:
                try:
                    val = int(clean)
                    goals.add(val)
                except Exception:
                    continue
        if not goals:
            return 0.0, 0.0, 0.0
        combo_count = float(len(goals))
        combo_width = float(max(goals) - min(goals))
        is_ultra_wide = 1.0 if combo_count >= 4 else 0.0
        return combo_count, combo_width, is_ultra_wide
    df_feat[['goal_combo_count', 'goal_combo_width', 'goal_combo_is_ultra_wide']] = df_feat.apply(parse_goal_combo_stats, axis=1, result_type='expand')
    df_feat['goal_combo_count'] = df_feat['goal_combo_count'].astype(float)
    df_feat['goal_combo_width'] = df_feat['goal_combo_width'].astype(float)
    df_feat['goal_combo_is_ultra_wide'] = df_feat['goal_combo_is_ultra_wide'].astype(float)

    # ================== 6. 方案内容编码（不再按预测者展开，只保留“方案本身”的信息） ==================
    # 统一 6+8+9 维编码，缺失部分置 0
    scheme_cols_sp = ["scheme_sp_胜", "scheme_sp_平", "scheme_sp_负", "scheme_sp_让胜", "scheme_sp_让平", "scheme_sp_让负"]
    scheme_cols_goal = [
        "scheme_goal_0", "scheme_goal_1", "scheme_goal_2", "scheme_goal_3",
        "scheme_goal_4", "scheme_goal_5", "scheme_goal_6", "scheme_goal_7plus"
    ]
    scheme_cols_half = [
        "scheme_half_胜胜", "scheme_half_胜平", "scheme_half_胜负",
        "scheme_half_平胜", "scheme_half_平平", "scheme_half_平负",
        "scheme_half_负胜", "scheme_half_负平", "scheme_half_负负"
    ]
    scheme_cols_all = scheme_cols_sp + scheme_cols_goal + scheme_cols_half

    X_scheme = pd.DataFrame(0.0, index=df_feat.index, columns=scheme_cols_all)

    for idx, row in df_feat.iterrows():
        original_term = str(row['original_term']) if pd.notna(row['original_term']) else ""
        ptype = row['prediction_type']
        vec = encode_prediction_scheme(original_term, ptype)

        if ptype == "胜平负/让球胜平负" and len(vec) == len(scheme_cols_sp):
            X_scheme.loc[idx, scheme_cols_sp] = vec
        elif ptype == "总进球数" and len(vec) == len(scheme_cols_goal):
            X_scheme.loc[idx, scheme_cols_goal] = vec
        elif ptype == "半全场" and len(vec) == len(scheme_cols_half):
            X_scheme.loc[idx, scheme_cols_half] = vec
        # 其他类型保持全 0

    # ================== 7. 标签（训练模式） ==================
    y = None
    if is_training:
        # 7.1 检查 judge_prediction_hit 所需列
        required_base_cols = ['original_term', 'prediction_type', 'goal_diff']
        missing_base_cols = [col for col in required_base_cols if col not in df_feat.columns]
        if missing_base_cols:
            raise ValueError(f"训练模式下，数据缺少计算命中所需的基础列: {missing_base_cols}")

        pred_types = df_feat['prediction_type'].unique()
        optional_required_cols = []
        if '胜平负/让球胜平负' in pred_types:
            optional_required_cols.append('handicap_value')
        if '总进球数' in pred_types:
            optional_required_cols.append('total_goals')
        if '半全场' in pred_types:
            optional_required_cols.append('half_full_result')

        missing_optional_cols = [col for col in optional_required_cols if col not in df_feat.columns]
        if missing_optional_cols:
            raise ValueError(
                f"训练模式下，数据缺少计算命中所需的可选列（对应预测类型：{pred_types}）: {missing_optional_cols}"
            )

        # 7.2 生成 is_hit 标签
        df_feat['is_hit'] = df_feat.apply(
            lambda row: judge_prediction_hit(
                original_term=row['original_term'],
                prediction_type=row['prediction_type'],
                goal_diff=row['goal_diff'],
                handicap_value=row.get('handicap_value'),
                total_goals=row.get('total_goals'),
                half_full_result=row.get('half_full_result')
            ),
            axis=1
        ).astype(int)

        y = df_feat['is_hit'].astype(int)

    # ================== 8. 组装最终特征矩阵 ==================
    numeric_cols = [
        'pred_overall_acc', 'pred_recent3_acc', 'pred_recent5_acc',
        'pred_total_pred_norm', 'pred_type_hit_rate',
        'pred_type_total_pred_norm', 'global_type_hit_rate',
        'prediction_consistency', 'match_pred_density', 'pred_type_identifier',
        'goal_combo_count', 'goal_combo_width', 'goal_combo_is_ultra_wide'
    ]
    # 确保这些列存在
    for col in numeric_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0.1

    X_numeric = df_feat[numeric_cols].astype(float)
    X_joint = pd.concat([X_numeric, X_scheme], axis=1)

    # 缺失值统一处理
    X = X_joint.loc[:, ~X_joint.columns.duplicated()].fillna(0.1)
    feature_cols = X.columns.tolist()

    # 推理阶段清理临时历史列
    if (not is_training) and ('_history_data' in df_feat.columns):
        df_feat = df_feat.drop(columns=['_history_data'])

    return X, y, feature_cols, current_predictor_ids


# ===================== 新增：模型预测相关数据库函数 =====================
def init_model_pred_tables(db_path):
    """初始化模型预测记录表、历史统计表，以及按置信度区间/Top2的模型表现统计表（首次运行创建）"""
    conn = get_db_connection(db_path)
    if not conn:
        return False

    # 表1：model_prediction_records（存储模型预测记录，用于后续判断对错）
    create_pred_table_sql = '''
    CREATE TABLE IF NOT EXISTS model_prediction_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pred_date TEXT NOT NULL,              -- 预测日期（YYYY-MM-DD）
        match_id INTEGER NOT NULL,            -- 比赛ID
        original_term TEXT NOT NULL,          -- 预测方案（如：胜/让胜）
        prediction_type TEXT NOT NULL,        -- 预测类型（胜平负/让球胜平负等）
        confidence FLOAT NOT NULL,            -- 三模型/加权后的置信度
        is_hit INTEGER DEFAULT NULL,          -- 是否命中（1=命中，0=未命中，NULL=未结算）
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(pred_date, match_id, original_term, prediction_type)  -- 避免重复插入同一场同一方案
    )
    '''

    # 表2：model_historical_stats（存储模型整体累计统计，用于展示总体准确率）
    create_stats_table_sql = '''
    CREATE TABLE IF NOT EXISTS model_historical_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        total_predictions INTEGER DEFAULT 0,  -- 累计预测次数
        total_hits INTEGER DEFAULT 0,         -- 累计命中次数
        accuracy FLOAT DEFAULT 0.0,           -- 累计准确率
        last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    '''

    # ===== 新增 6 张“按预测者维度统计”的表（结构类似 predictor 统计表） =====
    # 统一结构说明：
    #   predictor_id       预测者ID
    #   prediction_type    玩法类型（胜平负/让球胜平负/进球数/半全场…）
    #   bucket_name        条件标签（比如 'TOP2' / '[0.5,0.6)' / '[0.6,0.7)' 等）
    #   total_predictions  在该条件下的预测次数
    #   total_hits         在该条件下的命中次数
    #   accuracy           命中率 = total_hits / total_predictions
    #   last_update_time   最近更新时间
    #
    # 这样你以后想扩展更多分桶，只要新增表或者新增 bucket_name 即可。

    # 1）Top2 置信度范围内的预测表现（比如每场比赛置信度最高的两个方案）
    create_top2_table_sql = '''
    CREATE TABLE IF NOT EXISTS model_pred_stats_top2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pred_date TEXT NOT NULL,                  -- 预测日期（YYYY-MM-DD）
        match_id INTEGER NOT NULL,                -- 比赛ID
        prediction_type TEXT NOT NULL,            -- 玩法类型（胜平负/让球胜平负、总进球数、半全场）
        original_term TEXT NOT NULL,              -- 模型最终挑选的预测内容（例如 3/4球、胜/让胜 等）
        confidence FLOAT NOT NULL,                -- 当前这条方案的模型置信度
        bucket_name TEXT NOT NULL DEFAULT 'TOP2', -- 条件标签（这里恒为 TOP2）
        is_hit INTEGER DEFAULT NULL,              -- 赛后结算是否命中（1=命中，0=没中，NULL=未结算）
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(pred_date, match_id, prediction_type, original_term, bucket_name)
    )
    '''

    # 下面 5 张是不同置信度区间的统计表
    # 你可以把 bucket_name 当成“桶名”，后续查询/展示会更直观

    create_bucket_05_06_sql = '''
    CREATE TABLE IF NOT EXISTS model_pred_stats_conf_05_06 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predictor_id INTEGER NOT NULL,
        prediction_type TEXT NOT NULL,
        bucket_name TEXT NOT NULL DEFAULT '[0.5,0.6)',
        total_predictions INTEGER DEFAULT 0,
        total_hits INTEGER DEFAULT 0,
        accuracy FLOAT DEFAULT 0.0,
        last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(predictor_id, prediction_type, bucket_name)
    )
    '''

    create_bucket_06_07_sql = '''
    CREATE TABLE IF NOT EXISTS model_pred_stats_conf_06_07 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predictor_id INTEGER NOT NULL,
        prediction_type TEXT NOT NULL,
        bucket_name TEXT NOT NULL DEFAULT '[0.6,0.7)',
        total_predictions INTEGER DEFAULT 0,
        total_hits INTEGER DEFAULT 0,
        accuracy FLOAT DEFAULT 0.0,
        last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(predictor_id, prediction_type, bucket_name)
    )
    '''

    create_bucket_07_08_sql = '''
    CREATE TABLE IF NOT EXISTS model_pred_stats_conf_07_08 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predictor_id INTEGER NOT NULL,
        prediction_type TEXT NOT NULL,
        bucket_name TEXT NOT NULL DEFAULT '[0.7,0.8)',
        total_predictions INTEGER DEFAULT 0,
        total_hits INTEGER DEFAULT 0,
        accuracy FLOAT DEFAULT 0.0,
        last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(predictor_id, prediction_type, bucket_name)
    )
    '''

    create_bucket_08_09_sql = '''
    CREATE TABLE IF NOT EXISTS model_pred_stats_conf_08_09 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predictor_id INTEGER NOT NULL,
        prediction_type TEXT NOT NULL,
        bucket_name TEXT NOT NULL DEFAULT '[0.8,0.9)',
        total_predictions INTEGER DEFAULT 0,
        total_hits INTEGER DEFAULT 0,
        accuracy FLOAT DEFAULT 0.0,
        last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(predictor_id, prediction_type, bucket_name)
    )
    '''

    create_bucket_09_10_sql = '''
    CREATE TABLE IF NOT EXISTS model_pred_stats_conf_09_10 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predictor_id INTEGER NOT NULL,
        prediction_type TEXT NOT NULL,
        bucket_name TEXT NOT NULL DEFAULT '[0.9,1.0]',
        total_predictions INTEGER DEFAULT 0,
        total_hits INTEGER DEFAULT 0,
        accuracy FLOAT DEFAULT 0.0,
        last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(predictor_id, prediction_type, bucket_name)
    )
    '''

    try:
        cursor = conn.cursor()
        # 原有两张表
        cursor.execute(create_pred_table_sql)
        cursor.execute(create_stats_table_sql)

        # 如果没有历史统计记录，初始化一条
        cursor.execute("SELECT 1 FROM model_historical_stats LIMIT 1")
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO model_historical_stats (total_predictions, total_hits, accuracy) "
                "VALUES (0, 0, 0.0)"
            )

        # 新增 6 张“预测者统计”表
        cursor.execute(create_top2_table_sql)

        # 兼容旧版本：若历史表中缺少以下列，则尝试补充（避免 no such column 报错）
        alter_stmts = [
            "ALTER TABLE model_pred_stats_top2 ADD COLUMN pred_date TEXT",
            "ALTER TABLE model_pred_stats_top2 ADD COLUMN match_id INTEGER",
            "ALTER TABLE model_pred_stats_top2 ADD COLUMN predictor_id INTEGER",
            "ALTER TABLE model_pred_stats_top2 ADD COLUMN original_term TEXT",
            "ALTER TABLE model_pred_stats_top2 ADD COLUMN confidence FLOAT"
        ]
        for sql_stmt in alter_stmts:
            try:
                cursor.execute(sql_stmt)
            except Exception:
                # 列已存在或其他非致命错误时直接忽略
                pass

        cursor.execute(create_bucket_05_06_sql)
        cursor.execute(create_bucket_06_07_sql)
        cursor.execute(create_bucket_07_08_sql)
        cursor.execute(create_bucket_08_09_sql)
        cursor.execute(create_bucket_09_10_sql)

        conn.commit()
        return True
    except Exception as e:
        print(f"初始化模型预测表失败：{str(e)}")
        return False
    finally:
        conn.close()


def save_prediction_to_db(db_path, pred_df_top2):
    """将置信度前2的预测记录存入数据库"""
    conn = get_db_connection(db_path)
    if not conn:
        return False

    try:
        cursor = conn.cursor()
        for _, row in pred_df_top2.iterrows():
            # 插入预测记录（忽略重复）
            cursor.execute('''
                INSERT OR IGNORE INTO model_prediction_records 
                (pred_date, match_id, original_term, prediction_type, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                row['betting_cycle_date'].strftime('%Y-%m-%d'),
                row['match_id'],
                row['original_term'],
                row['prediction_type'],
                round(row['confidence'], 3)
            ))
        conn.commit()
        return True
    except Exception as e:
        print(f"保存预测记录失败：{str(e)}")
        return False
    finally:
        conn.close()


def get_model_historical_stats(db_path):
    """获取模型历史统计（累计准确率等）"""
    conn = get_db_connection(db_path)
    if not conn:
        return {"total_predictions": 0, "total_hits": 0, "accuracy": 0.0}

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT total_predictions, total_hits, accuracy FROM model_historical_stats LIMIT 1")
        result = cursor.fetchone()
        if result:
            return {
                "total_predictions": result[0],
                "total_hits": result[1],
                "accuracy": round(result[2], 3)
            }
        else:
            return {"total_predictions": 0, "total_hits": 0, "accuracy": 0.0}
    except Exception as e:
        print(f"获取模型统计失败：{str(e)}")
        return {"total_predictions": 0, "total_hits": 0, "accuracy": 0.0}
    finally:
        conn.close()


# 新增：预测者特征模板工具（统一特征格式，适配新增预测者）
def get_predictor_feature_templates():
    """获取预测者特征模板（固定格式，适配所有预测者）"""
    # 1. 预测者表现特征（4类）
    perf_templates = [
        "{pid}_historical_acc",    # 历史总准确率
        "{pid}_norm_total_pred",   # 归一化总预测数
        "{pid}_recent3_acc",       # 最近3次准确率
        "{pid}_recent5_acc"        # 最近5次准确率
    ]
    # 2. 预测方案编码模板（按预测类型区分维度）
    scheme_templates = {
        "胜平负/让球胜平负": ["{pid}_胜", "{pid}_平", "{pid}_负", "{pid}_让胜", "{pid}_让平", "{pid}_让负"],
        "总进球数": ["{pid}_0球", "{pid}_1球", "{pid}_2球", "{pid}_3球", "{pid}_4球", "{pid}_5球", "{pid}_6球", "{pid}_7+球"],
        "半全场": ["{pid}_胜胜", "{pid}_胜平", "{pid}_胜负", "{pid}_平胜", "{pid}_平平", "{pid}_平负", "{pid}_负胜", "{pid}_负平", "{pid}_负负"]
    }
    return perf_templates, scheme_templates

def generate_predictor_features(predictor_ids, prediction_type, df_history, df_valid):
    """
    生成所有预测者的特征（含新增预测者）
    :param predictor_ids: 当前所有预测者ID（含新增）
    :param prediction_type: 预测类型（胜平负/总进球数/半全场）
    :param df_history: 全量历史数据（用于计算预测者表现）
    :param df_valid: 当前窗口/推理的预测数据（用于获取预测者总预测数/命中数）
    :return: 预测者特征DataFrame、特征列名列表
    """
    perf_templates, scheme_templates = get_predictor_feature_templates()
    all_features = []
    all_feature_names = []

    # 1. 生成每个预测者的特征
    for pid in predictor_ids:
        # 1.1 计算预测者表现特征（historical_acc、recent3_acc等）
        perf_vec = get_predictor_performance_vec(df_valid=df_valid, df_history=df_history, predictor_id=pid, is_training=True)
        # 生成表现特征列名（替换模板中的{pid}）
        perf_feature_names = [t.format(pid=f"pred_{pid}") for t in perf_templates]
        perf_features = pd.DataFrame([perf_vec], columns=perf_feature_names)

        # 1.2 生成预测方案编码特征（默认全0，后续按实际预测方案赋值）
        scheme_template = scheme_templates[prediction_type]
        scheme_feature_names = [t.format(pid=f"pred_{pid}") for t in scheme_template]
        scheme_features = pd.DataFrame(np.zeros((1, len(scheme_template))), columns=scheme_feature_names)

        # 1.3 合并当前预测者的所有特征
        pid_features = pd.concat([perf_features, scheme_features], axis=1)
        all_features.append(pid_features)
        all_feature_names.extend(perf_feature_names + scheme_feature_names)

    # 2. 合并所有预测者的特征（行：1行，列：所有预测者的特征）
    if not all_features:
        return pd.DataFrame(), []
    combined_features = pd.concat(all_features, axis=1)
    # 去重（避免同一预测者重复生成特征）
    combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
    return combined_features, combined_features.columns.tolist()

def align_features_with_predictors(X, current_predictor_ids, model_predictor_ids, model_features):
    """
    特征对齐：让新数据的特征与历史模型训练时的特征完全一致
    :param X: 新数据的特征矩阵（DataFrame）
    :param current_predictor_ids: 当前所有预测者ID（含新增）
    :param model_predictor_ids: 历史模型训练时的预测者ID
    :param model_features: 历史模型训练时的特征列表（从 prev_lr_features/prev_lgb_features 加载）
    :return: 对齐后的特征矩阵（与 model_features 完全一致）
    """
    # 1. 确保新数据的列名是字符串类型（避免因类型不一致导致的匹配失败）
    X = X.rename(columns=str)
    model_features = [str(col) for col in model_features]

    # 2. 初始化对齐后的特征矩阵（以历史模型的特征列表为基准）
    X_aligned = pd.DataFrame(index=X.index, columns=model_features)

    # 3. 填充特征值：
    # 3.1 先填充新数据和历史模型共有的特征
    common_features = [col for col in model_features if col in X.columns]
    X_aligned[common_features] = X[common_features].copy()

    # 3.2 填充新数据缺失的特征（用 0 填充，包括全局特征、旧预测者特征）
    missing_features = [col for col in model_features if col not in X.columns]
    X_aligned[missing_features] = 0.0

    # 4. 检查并处理数据类型（确保和历史模型一致，避免float/int不匹配）
    X_aligned = X_aligned.astype(np.float32)

    # 5. 最终检查：确保对齐后的特征和历史模型完全一致
    assert list(X_aligned.columns) == model_features, \
        f"特征对齐失败：对齐后特征 {list(X_aligned.columns)} 与历史模型特征 {model_features} 不一致"

    return X_aligned