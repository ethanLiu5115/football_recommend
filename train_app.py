import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
import pickle
from utils import (
    load_processed_training_data, get_feature_columns,
    plot_accuracy_trend, plot_feature_weights, save_model
)

# --- 确保目录存在 ---
os.makedirs('trained_models', exist_ok=True)
os.makedirs('training_logs', exist_ok=True)


def rolling_train():
    from bayes_opt import BayesianOptimization
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')
    import pickle
    import os

    # 确保模型和日志目录存在
    os.makedirs('trained_models', exist_ok=True)
    os.makedirs('training_logs', exist_ok=True)

    core_df = load_processed_training_data()
    if core_df.empty:
        st.error("数据库中没有找到已完成的比赛数据，无法进行训练。")
        return

    date_list = sorted(core_df["betting_cycle_date"].unique())
    feature_cols = get_feature_columns(core_df)

    # -------------------------- 第一步：贝叶斯优化调参 --------------------------
    st.subheader("🔍 贝叶斯优化调参（实时展示过程）")
    tune_data = core_df.iloc[:int(len(core_df) * 0.8)]
    if len(tune_data) < 5:
        st.warning("调参数据不足（<5条），使用默认参数训练")
        best_params_no = {"max_depth": 4, "eta": 0.2, "alpha": 0.1, "lambda": 0.1, "objective": "multi:softprob",
                          "num_class": 3, "seed": 42}
        best_params_h = best_params_no.copy()
    else:
        X_tune = tune_data[feature_cols].fillna(0)
        y_tune_no = tune_data["no_handicap_result"].values
        y_tune_h = tune_data["handicap_result"].values

        opt_history_no, opt_history_h = [], []

        def objective_no(max_depth, eta, alpha, lambda_, num_round):
            params = {"max_depth": int(max_depth), "eta": eta, "alpha": alpha, "lambda": lambda_,
                      "objective": "multi:softprob", "num_class": 3, "eval_metric": "merror", "seed": 42, "silent": 1}
            dtrain = xgb.DMatrix(X_tune, label=y_tune_no)
            cv_results = xgb.cv(params, dtrain, num_boost_round=int(num_round), nfold=3, early_stopping_rounds=5,
                                verbose_eval=False)
            score = 1 - cv_results["test-merror-mean"].iloc[-1]
            opt_history_no.append({"iter": len(opt_history_no) + 1, "score": score})
            return score

        def objective_h(max_depth, eta, alpha, lambda_, num_round):
            params = {"max_depth": int(max_depth), "eta": eta, "alpha": alpha, "lambda": lambda_,
                      "objective": "multi:softprob", "num_class": 3, "eval_metric": "merror", "seed": 42, "silent": 1}
            dtrain = xgb.DMatrix(X_tune, label=y_tune_h)
            cv_results = xgb.cv(params, dtrain, num_boost_round=int(num_round), nfold=3, early_stopping_rounds=5,
                                verbose_eval=False)
            score = 1 - cv_results["test-merror-mean"].iloc[-1]
            opt_history_h.append({"iter": len(opt_history_h) + 1, "score": score})
            return score

        st.write("📊 胜平负模型调参中...")
        optimizer_no = BayesianOptimization(f=objective_no,
                                            pbounds={"max_depth": (2, 6), "eta": (0.05, 0.3), "alpha": (0, 3),
                                                     "lambda_": (0, 3), "num_round": (50, 200)},
                                            random_state=42, verbose=1)
        optimizer_no.maximize(init_points=3, n_iter=10)
        best_params_no = optimizer_no.max["params"]
        best_params_no.update({"max_depth": int(best_params_no["max_depth"]),
                               "num_round": int(best_params_no["num_round"])})
        best_params_no["objective"] = "multi:softprob"
        best_params_no["num_class"] = 3

        st.write("📊 让球胜平负模型调参中...")
        optimizer_h = BayesianOptimization(f=objective_h,
                                           pbounds={"max_depth": (2, 6), "eta": (0.05, 0.3), "alpha": (0, 3),
                                                    "lambda_": (0, 3), "num_round": (50, 200)},
                                           random_state=42, verbose=1)
        optimizer_h.maximize(init_points=3, n_iter=10)
        best_params_h = optimizer_h.max["params"]
        best_params_h.update({"max_depth": int(best_params_h["max_depth"]),
                              "num_round": int(best_params_h["num_round"])})
        best_params_h["objective"] = "multi:softprob"
        best_params_h["num_class"] = 3

        st.success("✅ 调参完成！最优参数如下：")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**胜平负模型**")
            for k, v in best_params_no.items():
                st.write(f"- {k}: {v}")
        with col2:
            st.write("**让球胜平负模型**")
            for k, v in best_params_h.items():
                st.write(f"- {k}: {v}")

    # -------------------------- 第二步：初始化训练组件 --------------------------
    model_cache_no, model_cache_h = [], []
    train_history = pd.DataFrame(columns=["date", "train_days", "no_acc", "h_acc", "no_train_loss", "h_train_loss"])
    feature_weights = pd.DataFrame(columns=["date", "feature_name", "weight"])
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = len(date_list)
    no_label_map = {0: "负", 1: "平", 2: "胜"}
    h_label_map = {0: "让负", 1: "让平", 2: "让胜"}

    def weighted_predict(X, cache, dmatrix=True):
        if len(cache) == 0:
            return np.array([])
        pred_probs_list = []
        for m in cache:
            prob = m.predict(X) if dmatrix else m.predict(xgb.DMatrix(X))
            pred_probs_list.append(prob)
        weights = [0.2, 0.3, 0.5][-len(cache):]
        avg_probs = sum(prob * w for prob, w in zip(pred_probs_list, weights))
        y_pred = avg_probs.argmax(axis=1).astype(int)
        return y_pred

    # -------------------------- 第三步：滚动训练循环 --------------------------
    for i in range(len(date_list)):
        test_date = date_list[i]
        test_df = core_df[core_df["betting_cycle_date"] == test_date]
        train_dates = date_list[:i] if i > 0 else [test_date]
        train_df = core_df[core_df["betting_cycle_date"].isin(train_dates)]

        st.write(f"**[迭代 {i + 1}/{total_steps}] 处理日期: {test_date}**")
        st.write(f"训练集样本数：{len(train_df)} | 测试集样本数：{len(test_df)}")
        if train_df.empty or test_df.empty:
            st.warning("训练集或测试集为空，跳过本轮迭代。")
            continue

        X_train = train_df[feature_cols].fillna(0)
        y_train_no = train_df["no_handicap_result"].values
        y_train_h = train_df["handicap_result"].values
        X_test = test_df[feature_cols].fillna(0)
        y_test_no = test_df["no_handicap_result"].values
        y_test_h = test_df["handicap_result"].values

        dtrain_no = xgb.DMatrix(X_train, label=y_train_no)
        dtrain_h = xgb.DMatrix(X_train, label=y_train_h)
        dtest_no = xgb.DMatrix(X_test)
        dtest_h = xgb.DMatrix(X_test)

        params_no = best_params_no.copy()
        params_h = best_params_h.copy()
        params_no.pop("num_round", None)
        params_h.pop("num_round", None)

        model_no = xgb.train(params_no, dtrain_no, num_boost_round=best_params_no["num_round"])
        model_h = xgb.train(params_h, dtrain_h, num_boost_round=best_params_h["num_round"])

        no_train_loss = float(model_no.eval(dtrain_no).split()[1].split(':')[1])
        h_train_loss = float(model_h.eval(dtrain_h).split()[1].split(':')[1])

        model_cache_no.append(model_no)
        model_cache_h.append(model_h)
        if len(model_cache_no) > 3: model_cache_no.pop(0)
        if len(model_cache_h) > 3: model_cache_h.pop(0)

        y_pred_no = weighted_predict(dtest_no, model_cache_no)
        y_pred_h = weighted_predict(dtest_h, model_cache_h)

        # -------------------------- 计算指标 --------------------------
        acc_no, acc_h = np.nan, np.nan
        no_cm, no_report, h_cm, h_report = None, None, None, None
        # 新增：提前初始化 target_names，避免后续引用错误
        target_names_no, target_names_h = [], []

        if len(y_pred_no) == len(y_test_no) and len(y_pred_no) > 0:
            acc_no = accuracy_score(y_test_no, y_pred_no)
            no_cm = confusion_matrix(y_test_no, y_pred_no)
            existing_classes_no = sorted(list(set(y_test_no) | set(y_pred_no)))
            target_names_no = [no_label_map[cls] for cls in existing_classes_no]
            no_report = classification_report(y_test_no, y_pred_no, labels=existing_classes_no,
                                              target_names=target_names_no, output_dict=True, zero_division=0)
            st.write("胜平负准确率：", acc_no)
        else:
            st.warning("胜平负样本数不匹配，无法计算指标")

        if len(y_pred_h) == len(y_test_h) and len(y_pred_h) > 0:
            acc_h = accuracy_score(y_test_h, y_pred_h)
            h_cm = confusion_matrix(y_test_h, y_pred_h)
            existing_classes_h = sorted(list(set(y_test_h) | set(y_pred_h)))
            target_names_h = [h_label_map[cls] for cls in existing_classes_h]
            h_report = classification_report(y_test_h, y_pred_h, labels=existing_classes_h, target_names=target_names_h,
                                             output_dict=True, zero_division=0)
            st.write("让球准确率：", acc_h)
        else:
            st.warning("让球胜平负样本数不匹配，无法计算指标")

        # 更新训练历史
        new_history_entry = pd.DataFrame({
            "date": [test_date], "train_days": [i + 1],
            "no_acc": [acc_no], "h_acc": [acc_h],
            "no_train_loss": [no_train_loss], "h_train_loss": [h_train_loss]
        })
        train_history = pd.concat([train_history, new_history_entry], ignore_index=True)

        # -------------------------- 提取特征权重 --------------------------
        weights_no = pd.Series(model_no.get_score(importance_type='weight'), name='no_weight').fillna(0)
        weights_h = pd.Series(model_h.get_score(importance_type='weight'), name='h_weight').fillna(0)
        combined_weights = pd.concat([weights_no, weights_h], axis=1).fillna(0)
        combined_weights['avg_weight'] = (combined_weights['no_weight'] + combined_weights['h_weight']) / 2
        avg_weights = combined_weights['avg_weight'].to_dict()

        weight_df = pd.DataFrame({
            "date": [test_date] * len(avg_weights),
            "feature_name": list(avg_weights.keys()),
            "weight": list(avg_weights.values())
        })
        feature_weights = pd.concat([feature_weights, weight_df], ignore_index=True)

        # 实时评估展示
        progress = (i + 1) / total_steps
        progress_bar.progress(progress)
        status_text.text(f"正在处理：{test_date}（进度：{progress:.1%}）")
        st.success(f"### 日期 {test_date} 训练结果")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("胜平负准确率", f"{acc_no:.3f}")
        with col2:
            st.metric("让球准确率", f"{acc_h:.3f}")
        with col3:
            st.metric("胜平负训练损失", f"{no_train_loss:.3f}")
        with col4:
            st.metric("让球训练损失", f"{h_train_loss:.3f}")

        # -------------------------- 详细报告 --------------------------
        with st.expander("📋 胜平负详细报告", expanded=False):
            if no_cm is not None:
                cm_classes = sorted(list(set(y_test_no)))
                cm_index = [no_label_map[cls] for cls in cm_classes]
                st.dataframe(pd.DataFrame(no_cm, index=cm_index, columns=cm_index))
                st.text("混淆矩阵")

            # **核心修复**：将 report_df 的生成和展示逻辑完全放在 if no_report is not None: 内部
            if no_report is not None and len(target_names_no) > 0:
                report_df = pd.DataFrame({
                    "精确率": [no_report[label]["precision"] for label in target_names_no],
                    "召回率": [no_report[label]["recall"] for label in target_names_no],
                    "F1值": [no_report[label]["f1-score"] for label in target_names_no]
                }, index=target_names_no)
                st.dataframe(report_df.round(3))
                st.text("分类指标")
            else:
                st.info("无有效分类报告数据")

        with st.expander("📋 让球胜平负详细报告", expanded=False):
            if h_cm is not None:
                cm_classes = sorted(list(set(y_test_h)))
                cm_index = [h_label_map[cls] for cls in cm_classes]
                st.dataframe(pd.DataFrame(h_cm, index=cm_index, columns=cm_index))
                st.text("混淆矩阵")

            # **核心修复**：同理，让球胜平负部分也做同样处理
            if h_report is not None and len(target_names_h) > 0:
                report_df = pd.DataFrame({
                    "精确率": [h_report[label]["precision"] for label in target_names_h],
                    "召回率": [h_report[label]["recall"] for label in target_names_h],
                    "F1值": [h_report[label]["f1-score"] for label in target_names_h]
                }, index=target_names_h)
                st.dataframe(report_df.round(3))
                st.text("分类指标")
            else:
                st.info("无有效分类报告数据")

        st.divider()

    # -------------------------- 训练完成后处理 --------------------------
    progress_bar.empty()
    status_text.empty()

    if 'model_no' in locals() and 'model_h' in locals():
        model_no.save_model('trained_models/model_no_latest.json')
        model_h.save_model('trained_models/model_h_latest.json')
        pickle.dump(model_cache_no, open('trained_models/model_cache_no.pkl', 'wb'))
        pickle.dump(model_cache_h, open('trained_models/model_cache_h.pkl', 'wb'))

        log_filename = f"training_logs/滚动训练日志_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        if not train_history.empty:
            train_history.to_csv(log_filename, index=False, encoding="utf-8-sig")

        st.session_state['train_history'] = train_history
        st.session_state['feature_weights'] = feature_weights
        st.info(f"🎉 所有训练完成！\n- 模型保存：trained_models/\n- 日志：{log_filename}")
    else:
        st.error("模型未成功初始化，检查数据有效性！")


def main():
    st.set_page_config(page_title="模型训练 - 足球比赛预测系统", layout="wide")
    st.title("⚽ 模型训练（单天训练版）")

    # 初始化Session State
    if 'train_history' not in st.session_state:
        st.session_state['train_history'] = pd.DataFrame()
    if 'feature_weights' not in st.session_state:
        st.session_state['feature_weights'] = pd.DataFrame()

    # 训练控制区
    st.header("🚀 开始训练")
    st.write("""
    训练说明：
    1. 自动加载数据库中所有已完成的比赛数据（含盘口）
    2. 支持单天数据训练（训练集=测试集）
    3. 胜平负 + 让球胜平负 双模型并行训练
    4. 采用3模型加权平滑预测，自动学习双平（dp）等特征的权重
    5. 实时显示训练进度和准确率
    """)

    if st.button("启动训练", type="primary", use_container_width=True):
        with st.spinner("训练中... 请耐心等待（时间取决于数据量）"):
            rolling_train()

    # 训练结果可视化区
    st.header("📈 训练结果可视化")
    if st.session_state['train_history'].empty:
        st.info("请先启动训练以查看可视化结果")
    else:
        tab1, tab2 = st.tabs(["准确率趋势", "特征权重分析"])
        with tab1:
            fig = plot_accuracy_trend(st.session_state['train_history'])
            if fig:
                st.pyplot(fig)
                # 显示详细训练日志
                st.subheader("训练日志详情")
                st.dataframe(st.session_state['train_history'].round(3), use_container_width=True)
        with tab2:
            weights_df = st.session_state['feature_weights']
            if not weights_df.empty:
                # 日期选择器
                weights_df['date'] = pd.to_datetime(weights_df['date'])
                min_date = weights_df["date"].min().date()
                max_date = weights_df["date"].max().date()
                selected_date = st.date_input("选择查看日期", max_date, min_value=min_date, max_value=max_date)

                # TopN选择器
                top_n = st.slider("显示Top N预测者", min_value=5, max_value=30, value=10)

                # 绘制权重图
                fig = plot_feature_weights(weights_df, selected_date, top_n)
                if fig:
                    st.pyplot(fig)

                    # 显示权重详情表（可观察双平特征的权重）
                    st.subheader("特征权重详情")
                    daily_weights = weights_df[weights_df["date"].dt.date == selected_date].sort_values('weight',
                                                                                                        ascending=False)
                    st.dataframe(daily_weights[['feature_name', 'weight']].round(4), use_container_width=True)


if __name__ == "__main__":
    main()