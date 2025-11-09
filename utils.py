import pandas as pd
import sqlite3
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# 配置中文字体支持（解决中文显示方块问题）
plt.rcParams["font.family"] = ["STHeiti", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# --- 数据库连接 ---
def get_db_connection(db_path='data/football.db'):
    """创建并返回数据库连接"""
    return sqlite3.connect(db_path)


# --- 让球胜平负标签计算函数 ---
def calculate_handicap_result(home_goals, away_goals, home_handicap):
    """
    根据实际比分和主队让球数，计算让球胜平负结果。
    返回: 0 (让负), 1 (让平), 2 (让胜)
    """
    if pd.isna(home_goals) or pd.isna(away_goals) or pd.isna(home_handicap):
        return None

    adjusted_goal_diff = (home_goals) - (away_goals + home_handicap)
    if adjusted_goal_diff > 0:
        return 2  # 让胜
    elif adjusted_goal_diff == 0:
        return 1  # 让平
    else:
        return 0  # 让负


def generate_prediction_features():
    """动态生成预测者特征：包含命中率与预测选项的交互特征"""
    conn = get_db_connection()

    # 1. 获取所有预测者的ID、名称、命中率
    predictors = pd.read_sql("""
        SELECT predictor_id, predictor_name, total_hits, total_predictions 
        FROM predictor
    """, conn)
    predictors['hit_rate'] = predictors['total_hits'] / predictors['total_predictions'].replace(0, 1)
    predictor_map = {
        row['predictor_id']: (row['predictor_name'], row['hit_rate'])
        for _, row in predictors.iterrows()
    }

    # 2. 获取所有预测记录，并解析预测术语
    predictions = pd.read_sql("""
        SELECT match_id, predictor_id, original_term 
        FROM prediction
    """, conn)

    # 定义“预测术语”到“特征编码”的映射
    term_code_map = {
        "胜": "n3", "平": "n1", "负": "n0",
        "让胜": "h3", "让平": "h1", "让负": "h0",
        "双平": "dp"
    }
    # 获取所有可能的编码，用于确保每个预测者都有完整的交互特征
    all_codes = list(term_code_map.values())

    feature_rows = []
    for _, pred_row in predictions.iterrows():
        match_id = pred_row['match_id']
        pred_id = pred_row['predictor_id']
        term = pred_row['original_term']

        pred_name, hit_rate = predictor_map.get(pred_id, ("未知预测者", 0))

        # 初始化特征字典，先加入match_id
        feat = {'match_id': match_id}

        # --- 核心改进：创建交互特征 ---
        # 为这个预测者的每一个可能的预测选项都创建一个交互特征
        for code in all_codes:
            feat[f'pred_{pred_name}_{code}_interact'] = 0  # 默认为0

        # 解析该预测者在本场比赛的具体预测，并更新对应的交互特征
        predicted_codes = []
        for t in term.split('/'):
            t = t.strip()
            code = term_code_map.get(t, None)
            if code:
                predicted_codes.append(code)

        # 将该预测者的命中率赋值给所有他预测了的选项的交互特征
        for code in predicted_codes:
            feat[f'pred_{pred_name}_{code}_interact'] = hit_rate

        feature_rows.append(feat)

    # 处理空数据场景
    if not feature_rows:
        # 创建包含所有可能交互特征的空DataFrame
        dummy_cols = ['match_id']
        for pred_name, _ in predictor_map.values():
            for code in all_codes:
                dummy_cols.append(f'pred_{pred_name}_{code}_interact')
        # 如果没有任何预测者，加入一个默认的
        if not dummy_cols:
             dummy_cols = ['match_id', 'pred_未知预测者_n3_interact']
        features_df = pd.DataFrame(columns=dummy_cols)
    else:
        features_df = pd.DataFrame(feature_rows).fillna(0)
        # 由于一场比赛可能有多个预测者，需要按match_id合并，取最大值（0或命中率）
        features_df = features_df.groupby('match_id').max().reset_index()

    conn.close()
    return features_df


# --- 数据加载 ---
def load_processed_training_data():
    """加载训练数据：比赛+赛果+盘口+动态生成的预测者特征"""
    conn = get_db_connection()

    # 简化盘口查询：直接关联handicap表（无时间筛选，因为盘口固定）
    query = """
        SELECT 
            m.match_id,
            m.betting_cycle_date,
            r.full_time_result,
            r.home_goals,
            r.away_goals,
            h.handicap_value  -- 直接获取固定盘口
        FROM match m
        JOIN result r ON m.match_id = r.match_id
        JOIN handicap h ON m.match_id = h.match_id  -- 直接关联，无时间筛选
        WHERE r.full_time_result IS NOT NULL
        ORDER BY m.betting_cycle_date
    """
    core_df = pd.read_sql(query, conn)
    conn.close()

    if core_df.empty:
        return core_df

    # 1. 计算胜平负标签
    result_map = {"主胜": 2, "平": 1, "主负": 0}
    core_df['no_handicap_result'] = core_df['full_time_result'].map(result_map)

    # 2. 计算让球胜平负标签
    core_df['handicap_result'] = core_df.apply(
        lambda row: calculate_handicap_result(
            row['home_goals'],
            row['away_goals'],
            row['handicap_value']
        ),
        axis=1
    )

    # 过滤掉因缺少盘口数据而无法计算让球结果的比赛
    initial_count = len(core_df)
    core_df = core_df.dropna(subset=['handicap_result'])
    if len(core_df) < initial_count:
        print(f"警告：有 {initial_count - len(core_df)} 场比赛因缺少盘口数据被过滤。")

    # 3. 生成并合并预测者特征
    features_df = generate_prediction_features()
    if not features_df.empty:
        core_df = core_df.merge(features_df, on='match_id', how='left').fillna(0)
    else:
        # 若特征为空，创建空列避免后续错误
        feature_cols = get_feature_columns(core_df)
        for col in feature_cols:
            core_df[col] = 0

    # 确保标签是整数类型
    core_df['no_handicap_result'] = core_df['no_handicap_result'].astype(int)
    core_df['handicap_result'] = core_df['handicap_result'].astype(int)

    return core_df


def load_upcoming_matches_data():
    """加载未开始比赛的数据和动态生成的预测者特征"""
    conn = get_db_connection()

    # 简化盘口查询：直接关联handicap表（无时间筛选）
    query = """
        SELECT 
            m.match_id,
            m.betting_cycle_date,
            t1.team_name as home_team,
            t2.team_name as away_team,
            h.handicap_value  -- 加载固定盘口
        FROM match m
        JOIN team t1 ON m.home_team_id = t1.team_id
        JOIN team t2 ON m.away_team_id = t2.team_id
        JOIN handicap h ON m.match_id = h.match_id  -- 直接关联，无时间筛选
        WHERE m.match_id NOT IN (SELECT match_id FROM result)
        ORDER BY m.betting_cycle_date
    """
    upcoming_df = pd.read_sql(query, conn)
    conn.close()

    # 动态生成特征并合并
    if not upcoming_df.empty:
        features_df = generate_prediction_features()
        if not features_df.empty:
            upcoming_df = upcoming_df.merge(features_df, on='match_id', how='left').fillna(0)
        else:
            feature_cols = get_feature_columns(upcoming_df)
            for col in feature_cols:
                upcoming_df[col] = 0

    return upcoming_df


# --- 特征列提取 ---
def get_feature_columns(df):
    """从DataFrame中自动识别并返回特征列"""
    return [col for col in df.columns if col.startswith("pred_") and
            (col.endswith("_hit_rate") or any(x in col for x in ["_n3", "_n1", "_n0", "_h3", "_h1", "_h0", "_dp"]))]


# --- 模型保存与加载 ---
def save_model(model, filename):
    """保存XGBoost模型"""
    model.save_model(filename)


def load_model(filename):
    """加载XGBoost模型（兼容分类器）"""
    model = xgb.XGBClassifier()
    model.load_model(filename)
    return model


# --- 可视化函数 ---
def plot_accuracy_trend(history_df):
    """绘制准确率趋势图并返回matplotlib figure对象"""
    if history_df.empty:
        return None

    history_df['date'] = pd.to_datetime(history_df['date'])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_df["train_days"], history_df["no_acc"], label="胜平负准确率", marker="o", linewidth=2)
    ax.plot(history_df["train_days"], history_df["h_acc"], label="让球胜平负准确率", marker="s", linewidth=2)
    ax.set_xlabel("训练窗口大小（天）")
    ax.set_ylabel("准确率")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    return fig


def plot_feature_weights(weights_df, selected_date, top_n=10):
    """绘制特征权重图并返回matplotlib figure对象"""
    if weights_df.empty:
        return None

    weights_df['date'] = pd.to_datetime(weights_df['date'])
    daily_weights = weights_df[weights_df["date"].dt.date == selected_date].sort_values('weight', ascending=False).head(
        top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(daily_weights)), daily_weights["weight"], color='skyblue')
    ax.set_yticks(range(len(daily_weights)))
    pred_names = daily_weights["feature_name"].str.extract(r'pred_(.+?)_')[0].fillna("未知")
    ax.set_yticklabels([f"{name}" for name in pred_names], fontsize=10)
    ax.set_xlabel("特征重要性", fontsize=10)
    ax.set_title(f"特征权重 Top {top_n} ({selected_date})", fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}", ha="left", va="center", fontsize=9)

    return fig