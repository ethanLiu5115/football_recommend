import streamlit as st
import pandas as pd
import os
import pickle
from utils import (
    load_upcoming_matches_data, get_feature_columns, load_model
)

# 配置中文字体（与utils.py保持一致，确保推理页面中文显示正常）
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# --- 检查模型是否存在 ---
def check_model_exists():
    """检查训练好的模型和缓存是否存在"""
    required_files = [
        'trained_models/model_no_latest.json',
        'trained_models/model_h_latest.json',
        'trained_models/model_cache_no.pkl',
        'trained_models/model_cache_h.pkl'
    ]
    return all(os.path.exists(file) for file in required_files)


# --- 加权预测函数 ---
def weighted_predict(X, model_cache):
    """多模型加权预测（与训练时保持一致）"""
    if len(model_cache) == 1:
        return model_cache[0].predict(X), model_cache[0].predict_proba(X).max(axis=1)
    # 概率加权平均
    pred_probs = [m.predict_proba(X) for m in model_cache]
    weights = [0.2, 0.3, 0.5][-len(model_cache):]
    avg_probs = sum(prob * w for prob, w in zip(pred_probs, weights))
    return avg_probs.argmax(axis=1), avg_probs.max(axis=1)


# --- 预测核心函数 ---
def predict_upcoming():
    # 加载未开始比赛数据
    upcoming_df = load_upcoming_matches_data()
    if upcoming_df.empty:
        st.warning("⚠️ 数据库中没有找到未开始的比赛（无赛果且有预测数据的比赛）")
        return

    # 加载模型和缓存
    model_no = load_model('trained_models/model_no_latest.json')
    model_h = load_model('trained_models/model_h_latest.json')
    model_cache_no = pickle.load(open('trained_models/model_cache_no.pkl', 'rb'))
    model_cache_h = pickle.load(open('trained_models/model_cache_h.pkl', 'rb'))

    # 提取特征
    feature_cols = get_feature_columns(upcoming_df)
    X_inference = upcoming_df[feature_cols].fillna(0)

    # 执行预测
    pred_no, conf_no = weighted_predict(X_inference, model_cache_no)
    pred_h, conf_h = weighted_predict(X_inference, model_cache_h)

    # 结果映射（与标签定义一致）
    no_result_map = {0: "负", 1: "平", 2: "胜"}
    h_result_map = {0: "让负", 1: "让平", 2: "让胜"}

    # 整理结果
    results_df = upcoming_df[['match_id', 'betting_cycle_date', 'home_team', 'away_team']].copy()
    results_df['胜平负预测'] = [no_result_map[p] for p in pred_no]
    results_df['胜平负置信度'] = conf_no.round(3)
    results_df['让球胜平负预测'] = [h_result_map[p] for p in pred_h]
    results_df['让球胜平负置信度'] = conf_h.round(3)

    # 高亮高置信度结果（置信度≥0.6）
    def highlight_high_confidence(val):
        return 'background-color: #d4edda' if val >= 0.6 else ''

    st.subheader("预测结果汇总")
    styled_df = results_df.style.applymap(
        highlight_high_confidence,
        subset=['胜平负置信度', '让球胜平负置信度']
    )
    st.dataframe(styled_df, use_container_width=True)

    # 下载功能
    csv = results_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 下载预测结果（CSV）",
        data=csv,
        file_name=f"比赛预测结果_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


def main():
    st.set_page_config(page_title="比赛预测 - 足球比赛预测系统", layout="wide")
    st.title("🔮 未开始比赛预测")

    # 检查模型是否存在
    if not check_model_exists():
        st.error("❌ 未找到训练好的模型！请先运行 train_app.py 完成训练")
        return

    # 预测控制区
    st.header("开始预测")
    st.write("""
    预测说明：
    1. 自动加载数据库中未结束的比赛（无赛果）。
    2. 使用最新训练的3个模型加权预测，综合考虑预测者的单选、双选及双平倾向。
    3. 显示预测结果和置信度（≥0.6高亮标注）。
    4. 支持下载预测结果CSV文件。
    """)

    if st.button("生成预测结果", type="primary", use_container_width=True):
        with st.spinner("预测中... 正在处理未开始的比赛"):
            predict_upcoming()


if __name__ == "__main__":
    main()