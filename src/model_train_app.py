import json
import logging
import os
import sys
import joblib
import pandas as pd
import numpy as np
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, log_loss
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import warnings
from pymoo.config import Config
Config.warnings['not_compiled'] = False

warnings.filterwarnings('ignore')

# 全局字体设置：优先使用 macOS 常见中文字体，避免图表中文字显示为小方框
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',        # macOS 系统中文默认字体
    'Hiragino Sans GB',   # 较新的中文黑体
    'Heiti TC',           # 旧版黑体
    'Songti SC',          # 宋体系列
    'STHeiti',            # 兼容早期系统
    'SimHei',             # Windows 常见黑体
    'Arial Unicode MS',   # 跨平台备用
    'DejaVu Sans'         # 最后兜底
]
plt.rcParams['axes.unicode_minus'] = False

# ===================== 全局配置 =====================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 环境适配
CURRENT_ENV = os.getenv("FOOTBALL_ENV", "dev")
if CURRENT_ENV == "prod":
    from config.prod_config import DB_PATH
else:
    from config.dev_config import DB_PATH

# 目录配置（按环境区分模型目录；metrics / visualization 暂共用）
if CURRENT_ENV == "prod":
    MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_models")
else:
    # 开发环境单独使用 developed_models 目录，避免和生产模型混在一起
    MODEL_DIR = os.path.join(PROJECT_ROOT, "developed_models")

METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")
VIS_DIR = os.path.join(PROJECT_ROOT, "visualization")

# 创建目录
for dir_path in [MODEL_DIR, METRICS_DIR, VIS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 导入公共工具函数
from utils import (
    load_historical_data, load_prediction_data,
    feature_engineering, init_model_pred_tables,
    get_model_historical_stats, save_prediction_to_db,
    get_db_connection, align_features_with_predictors,
    judge_prediction_hit  # 确保导入命中判断函数
)

# 补充基础指标计算函数
def calculate_base_metrics(y_true, y_pred):
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 3),
        'precision': round(precision_score(y_true, y_pred), 3),
        'recall': round(recall_score(y_true, y_pred), 3),
        'f1': round(f1_score(y_true, y_pred), 3),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

# ===================== 新增：模型迭代管理配置 =====================
BEST_MODEL_CONFIG = os.path.join(MODEL_DIR, "best_model_config.json")
MODEL_RETAIN_COUNT = 3  # 保留模型数量：1个最优 + 2个最新

def load_best_model_config():
    """加载最优模型配置"""
    if not os.path.exists(BEST_MODEL_CONFIG):
        return {
            "model_date": "",
            "lgb_f1": 0.0,
            "lr_f1": 0.0,
            "lgb_path": "",
            "lr_path": "",
            "scaler_path": "",
            "lgb_features_path": "",
            "lr_features_path": "",
            "window_end": ""
        }
    with open(BEST_MODEL_CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)

def save_best_model_config(config):
    """保存最优模型配置"""
    with open(BEST_MODEL_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def get_recent_model_dates(top_n=2):
    """获取最近训练的n个模型日期（按时间降序）"""
    model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('lgb_model_')]
    # 提取模型日期（格式：YYYYMMDD_HHMMSS）
    model_dates = [f.split('_')[2] + '_' + f.split('_')[3].split('.')[0] for f in model_files]
    # 去重+降序排序
    model_dates = sorted(list(set(model_dates)), reverse=True)
    return model_dates[:top_n]

def clean_old_models():
    """清理冗余模型：只保留最优+最近2个（确保模型、特征、scaler文件同步保留）"""
    best_config = load_best_model_config()
    recent_dates = get_recent_model_dates(top_n=2)

    # 需保留的模型日期：最优模型日期 + 最近2个模型日期（去重）
    keep_dates = set(recent_dates)
    # 确保最佳模型日期存在且格式正确
    best_model_date = best_config.get("model_date")
    if best_model_date:
        keep_dates.add(best_model_date)
        logger.info(f"最佳模型日期 {best_model_date} 已加入保留列表")
    keep_dates = list(keep_dates)
    logger.info(f"需保留的模型日期：{keep_dates}")

    # 遍历所有模型相关文件
    for root, _, files in os.walk(MODEL_DIR):
        for file in files:
            # 只处理模型相关文件（排除其他无关文件）
            if not any(file.startswith(prefix) for prefix in [
                'lgb_model_', 'lr_model_',
                'lgb_features_', 'lr_features_',
                'scaler_', 'model_predictors_', 'model_pred_type_'
            ]):
                continue

            # 核心修复：按文件类型修正日期提取索引
            try:
                parts = file.split('_')
                if file.startswith(('lgb_model_', 'lr_model_', 'lgb_features_', 'lr_features_', 'model_predictors_')):
                    # 格式：xxx_xxx_YYYYMMDD_HHMMSS.pkl → 取索引2、3
                    if len(parts) < 4:
                        raise IndexError("文件命名格式错误，缺少足够的下划线分割部分")
                    date_part1 = parts[2]
                    date_part2 = parts[3].split('.')[0]
                    file_date = f"{date_part1}_{date_part2}"
                elif file.startswith('model_pred_type_'):
                    # 格式：model_pred_type_YYYYMMDD_HHMMSS.pkl → 取索引3、4
                    if len(parts) < 5:
                        raise IndexError("文件命名格式错误，缺少足够的下划线分割部分")
                    date_part1 = parts[3]
                    date_part2 = parts[4].split('.')[0]
                    file_date = f"{date_part1}_{date_part2}"
                elif file.startswith('scaler_'):
                    # 格式：scaler_YYYYMMDD_HHMMSS.pkl → 取索引1、2
                    if len(parts) < 3:
                        raise IndexError("文件命名格式错误，缺少足够的下划线分割部分")
                    date_part1 = parts[1]
                    date_part2 = parts[2].split('.')[0]
                    file_date = f"{date_part1}_{date_part2}"
                else:
                    logger.warning(f"跳过未知文件类型：{file}")
                    continue

                # 验证日期格式（可选，增强健壮性）
                if len(file_date.split('_')) != 2 or len(file_date.replace('_', '')) != 14:
                    logger.warning(f"文件 {file} 的日期格式不正确（应为YYYYMMDD_HHMMSS），跳过")
                    continue

            except Exception as e:
                logger.error(f"提取文件 {file} 日期失败：{str(e)}，跳过该文件")
                continue

            # 判断是否需要删除（不在保留日期列表中）
            if file_date not in keep_dates:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logger.info(f"删除冗余文件：{file_path}")
                except Exception as e:
                    logger.error(f"删除文件 {file_path} 失败：{str(e)}")

# 新增：共线性处理函数
def remove_high_correlation_features(X, threshold=0.8):
    """剔除高相关特征（相关系数绝对值>threshold）"""
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
    X_cleaned = X.drop(columns=to_drop)
    return X_cleaned, to_drop

# 针对总进球数玩法当前不做任何硬编码的置信度惩罚
def adjust_goal_combo_confidence(df):
    """
    针对总进球数玩法当前不做任何硬编码的置信度惩罚。
    保留这个函数主要是为了接口兼容，后续如果基于数据分析需要，可以在这里实现数据驱动的调整逻辑。
    目前直接返回原始置信度。
    """
    # 如果没有 confidence 列，直接返回
    if 'confidence' not in df.columns:
        return df
    # 当前不做调整，直接返回原 DataFrame
    return df

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("model_train")

# ===================== 1. 模型训练与调参（核心修改：特征选择） =====================
def hebo_lgb_tune(train_data, show_curve=True):
    """HEBO优化LightGBM参数，并在需要时绘制logloss收敛曲线"""
    param_config = [
        {'name': 'max_depth', 'type': 'num', 'lb': 3, 'ub': 6},
        {'name': 'num_leaves', 'type': 'int', 'lb': 16, 'ub': 32},
        {'name': 'learning_rate', 'type': 'num', 'lb': 0.01, 'ub': 0.08},
        {'name': 'reg_alpha', 'type': 'num', 'lb': 0.001, 'ub': 0.2},
        {'name': 'reg_lambda', 'type': 'num', 'lb': 0.001, 'ub': 0.2},
        # subsample / colsample 更靠近 1.0，让每棵树看更多样本和特征，从而有更多机会尝试“人维度”特征
        {'name': 'subsample', 'type': 'num', 'lb': 0.8, 'ub': 1.0},
        {'name': 'colsample_bytree', 'type': 'num', 'lb': 0.8, 'ub': 1.0},
        # 新增：控制叶子最小样本数，让树更愿意在稀疏的人特征上继续切分
        {'name': 'min_data_in_leaf', 'type': 'int', 'lb': 10, 'ub': 80},
    ]

    space = DesignSpace()
    space.parse(param_config)

    def objective(params):
        params = params.iloc[0].to_dict()
        int_params = ['num_leaves', 'max_depth', 'min_data_in_leaf']
        for param in int_params:
            if param in params:
                params[param] = int(params[param])

        params.update({
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
            'seed': 42,
            'feature_pre_filter': False
        })
        cv_results = lgb.cv(
            params,
            train_data,
            num_boost_round=200,
            nfold=3,
            stratified=True,
        )

        target_key = 'valid binary_logloss-mean'
        if target_key in cv_results and len(cv_results[target_key]) > 0:
            vals = cv_results[target_key]
            return float(np.min(vals))
        else:
            return 1e9

    hebo = HEBO(space, model_name='gp')
    for _ in range(20):
        try:
            suggest = hebo.suggest()
            loss = objective(suggest)
            hebo.observe(suggest, np.array([loss]))
            print(f"HEBO iteration {_ + 1}, Suggested params: {suggest.to_dict()}, Loss: {loss:.4f}")
        except Exception as e:
            print(f"HEBO iteration {_ + 1} failed with error: {e}")
            continue

    best_params = hebo.best_x.iloc[0].to_dict()
    int_params = ['num_leaves', 'max_depth', 'min_data_in_leaf']
    for param in int_params:
        if param in best_params:
            best_params[param] = int(best_params[param])
    # 显式同步 feature_fraction，鼓励每棵树使用更多特征
    if 'colsample_bytree' in best_params and 'feature_fraction' not in best_params:
        best_params['feature_fraction'] = best_params['colsample_bytree']

    best_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbose': -1,
        'seed': 42,
        'bagging_freq': 5,
        'feature_fraction_seed': 42,
        'bagging_seed': 42,
    })

    # 使用最佳参数重新做一遍CV，记录logloss随迭代轮数的变化，并找到最优 boosting 轮数
    logloss_curve = None
    best_boost_round = 200  # 默认为 200 轮，若CV成功则用最优轮数覆盖
    try:
        cv_results = lgb.cv(
            best_params,
            train_data,
            num_boost_round=200,
            nfold=3,
            stratified=True,
        )
        target_key = 'valid binary_logloss-mean'
        if target_key in cv_results and len(cv_results[target_key]) > 0:
            logloss_curve = cv_results[target_key]
            # 方案A：使用整个曲线中的最小 logloss 对应的轮数作为最优 boosting 轮数
            best_boost_round = int(np.argmin(logloss_curve)) + 1
    except Exception as e:
        logger.warning(f"LightGBM CV绘制logloss曲线失败: {e}")

    if show_curve and logloss_curve is not None:
        st.markdown("### 📉 LightGBM CV Logloss 收敛曲线（最佳参数）")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(logloss_curve) + 1), logloss_curve, marker='o', linewidth=1)
        ax.set_xlabel('迭代轮数（num_boost_round）', fontsize=12)
        ax.set_ylabel('验证集 logloss', fontsize=12)
        ax.set_title('CV Logloss vs Boosting Round', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    return best_params, best_boost_round

def train_base_models(X, y, current_predictor_ids, prediction_type):
    """训练流程：支持新增预测者，保存预测者ID和特征模板"""
    from lightgbm import LGBMClassifier

    # 分层抽样划分训练集/测试集（8:2）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------------- 步骤1-4：原训练逻辑保留，新增特征筛选时考虑所有预测者 ----------------------
    sample_weight = np.array([2 if lbl == 1 else 1 for lbl in y_train])
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
    best_lgb_params, best_boost_round = hebo_lgb_tune(train_data, show_curve=True)

    # 临时模型获取特征重要性
    lgb_clf_temp = LGBMClassifier(**best_lgb_params, n_estimators=best_boost_round, random_state=42)
    lgb_clf_temp.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_test, y_test)], eval_metric='binary_logloss')

    # 提取特征重要性（覆盖所有预测者的特征），仅用于后续可视化和分析
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': lgb_clf_temp.feature_importances_
    }).sort_values('importance', ascending=False)

    # 使用所有特征作为 LightGBM 的训练特征（当前工程特征维度本身不高）
    all_lgb_features = X.columns.tolist()

    # 剔除高相关特征（仅对 LR 做一层共线性处理）
    X_train_lr = X_train.copy()
    X_test_lr = X_test.copy()
    X_train_lr_cleaned, dropped_cols = remove_high_correlation_features(X_train_lr)
    X_test_lr_cleaned = X_test_lr.drop(columns=dropped_cols)
    final_lr_features = X_train_lr_cleaned.columns.tolist()

    # 重新训练最终模型
    X_train_lgb = X_train[all_lgb_features].copy()
    X_test_lgb = X_test[all_lgb_features].copy()

    lgb_clf_final = LGBMClassifier(**best_lgb_params, n_estimators=best_boost_round, random_state=42)
    lgb_clf_final.fit(
        X_train_lgb,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_test_lgb, y_test)],
        eval_metric='binary_logloss'
    )

    scaler = StandardScaler()
    X_train_lr_scaled = scaler.fit_transform(X_train_lr_cleaned)
    X_test_lr_scaled = scaler.transform(X_test_lr_cleaned)

    lr_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=42
    )
    lr_model.fit(X_train_lr_scaled, y_train)

    # ---------------------- 步骤5：保存模型时新增预测者ID和特征模板 ----------------------
    model_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 保存核心模型文件
    joblib.dump(lr_model, os.path.join(MODEL_DIR, f'lr_model_{model_date}.pkl'))
    joblib.dump(lgb_clf_final, os.path.join(MODEL_DIR, f'lgb_model_{model_date}.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'scaler_{model_date}.pkl'))
    joblib.dump(final_lr_features, os.path.join(MODEL_DIR, f'lr_features_{model_date}.pkl'))
    joblib.dump(all_lgb_features, os.path.join(MODEL_DIR, f'lgb_features_{model_date}.pkl'))
    # 新增：保存训练时的预测者ID和预测类型（用于后续对齐）
    joblib.dump(current_predictor_ids, os.path.join(MODEL_DIR, f'model_predictors_{model_date}.pkl'))
    joblib.dump(prediction_type, os.path.join(MODEL_DIR, f'model_pred_type_{model_date}.pkl'))

    # 计算指标（原逻辑保留）
    lgb_pred = lgb_clf_final.predict(X_test_lgb)
    lr_pred = lr_model.predict(X_test_lr_scaled)
    lr_metrics = calculate_base_metrics(y_test, lr_pred)
    lgb_metrics = calculate_base_metrics(y_test, lgb_pred)

    # 概率输出：用于后续 logloss / bucket 分析 & LGB vs Ensemble 对比
    lr_proba = lr_model.predict_proba(X_test_lr_scaled)[:, 1]
    lgb_proba = lgb_clf_final.predict_proba(X_test_lgb)[:, 1]
    ensemble_proba = (lgb_proba + lr_proba) / 2

    # 计算稳定性指标（增加NaN检查）
    def get_stability_metrics(scores):
        scores = scores[~np.isnan(scores)]  # 去除NaN值
        if len(scores) == 0:
            return 0.0, 0.0
        return round(scores.mean(), 3), round(scores.std(), 3)

    # 稳定性指标（原逻辑保留）
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    def safe_f1_score(y_true, y_pred): return f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    lgb_cv_scores = cross_val_score(
        LGBMClassifier(**best_lgb_params, n_estimators=best_boost_round, random_state=42),
        X[all_lgb_features],
        y,
        cv=skf,
        scoring=make_scorer(safe_f1_score)
    )
    lr_cv_scores = cross_val_score(LogisticRegression(**lr_model.get_params()), scaler.transform(X[final_lr_features]), y, cv=skf, scoring=make_scorer(safe_f1_score))
    lr_mean, lr_var = get_stability_metrics(lr_cv_scores)
    lgb_mean, lgb_var = get_stability_metrics(lgb_cv_scores)

    stability_metrics = {
        'lr_cv_f1_mean': lr_mean, 'lr_cv_var': lr_var,
        'lgb_cv_f1_mean': lgb_mean, 'lgb_cv_var': lgb_var,
        'lgb_top300_features': all_lgb_features,  # 现在含义是“LightGBM实际使用的全部特征”
        'lr_final_features': final_lr_features,
        'model_predictors': current_predictor_ids, 'prediction_type': prediction_type
    }

    metrics = {
        'lr_metrics': lr_metrics, 'lgb_metrics': lgb_metrics, 'stability_metrics': stability_metrics,
        'train_date': model_date, 'feature_cols': X.columns.tolist(),
        'model_predictors': current_predictor_ids, 'prediction_type': prediction_type
    }
    joblib.dump(metrics, os.path.join(METRICS_DIR, f'metrics_{model_date}.pkl'))

    # 返回结果（新增预测者ID和预测类型）
    return {
        'lr_model': lr_model, 'lgb_model': lgb_clf_final, 'scaler': scaler,
        'metrics': metrics, 'X_test': X_test, 'y_test': y_test,
        'lr_pred': lr_pred,
        'lgb_pred': lgb_pred,
        'lgb_proba': lgb_proba,
        'lr_proba': lr_proba,
        'ensemble_proba': ensemble_proba,
        'lgb_top300_features': all_lgb_features, 'lr_final_features': final_lr_features,
        'model_date': model_date, 'model_predictors': current_predictor_ids, 'prediction_type': prediction_type
    }

# ===================== 核心修改：微调模型函数（适配特征对齐） =====================
def fine_tune_model(prev_model_path, prev_scaler_path, prev_lr_features, prev_lgb_features, X_new_lr, X_new_lgb, y_new):
    """微调模型（接收对齐后的LR/LGB特征，确保与历史模型一致）"""
    # 加载历史模型和scaler
    lr_model = joblib.load(prev_model_path)
    lgb_model = joblib.load(prev_model_path.replace('lr_', 'lgb_'))
    scaler = joblib.load(prev_scaler_path)

    # 严格验证特征一致性（避免对齐遗漏）
    assert list(X_new_lr.columns) == prev_lr_features, f"LR特征不一致：历史{len(prev_lr_features)}个，当前{len(X_new_lr.columns)}个"
    assert list(X_new_lgb.columns) == prev_lgb_features, f"LGB特征不一致：历史{len(prev_lgb_features)}个，当前{len(X_new_lgb.columns)}个"

    # 标准化新数据（LR）
    X_new_lr_scaled = scaler.transform(X_new_lr).astype(np.float64)

    # 微调Logistic Regression（SGD）
    sgd_model = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1.0,
        random_state=42,
        warm_start=True,
        learning_rate='constant',
        eta0=0.01,
        max_iter=100
    )
    if hasattr(lr_model, 'coef_') and hasattr(lr_model, 'intercept_'):
        sgd_model.coef_ = lr_model.coef_
        sgd_model.intercept_ = lr_model.intercept_
    sgd_model.partial_fit(X_new_lr_scaled, y_new, classes=[0, 1])

    # 微调LightGBM
    lgb_model.set_params(learning_rate=0.05)
    sample_weight = np.array([2 if lbl == 1 else 1 for lbl in y_new])
    lgb_model.fit(
        X_new_lgb,
        y_new,
        sample_weight=sample_weight,
    )

    # 评估微调后模型
    lr_pred = sgd_model.predict(X_new_lr_scaled)
    lgb_pred = lgb_model.predict(X_new_lgb)
    lgb_proba = lgb_model.predict_proba(X_new_lgb)[:, 1]

    # 计算指标
    lr_metrics = calculate_base_metrics(y_new, lr_pred)
    lgb_metrics = calculate_base_metrics(y_new, lgb_pred)

    # 稳定性指标
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def safe_f1_score(y_true, y_pred):
        try:
            return f1_score(y_true, y_pred)
        except:
            return 0.0

    # LR稳定性评估
    lr_cv_scores = cross_val_score(sgd_model, X_new_lr_scaled, y_new, cv=skf, scoring=make_scorer(safe_f1_score))
    # LightGBM稳定性评估（用模型蓝图）
    lgb_params = lgb_model.get_params()
    lgb_cv_scores = cross_val_score(
        estimator=lgb.LGBMClassifier(**lgb_params),
        X=X_new_lgb,
        y=y_new,
        cv=skf,
        scoring=make_scorer(safe_f1_score),
        fit_params={'sample_weight': sample_weight}
    )

    # 计算稳定性指标（增加NaN检查）
    def get_stability_metrics(scores):
        scores = scores[~np.isnan(scores)]
        if len(scores) == 0:
            return 0.0, 0.0
        return round(scores.mean(), 3), round(scores.std(), 3)

    lr_mean, lr_var = get_stability_metrics(lr_cv_scores)
    lgb_mean, lgb_var = get_stability_metrics(lgb_cv_scores)

    stability_metrics = {
        'lr_cv_f1_mean': lr_mean,
        'lr_cv_var': lr_var,
        'lgb_cv_f1_mean': lgb_mean,
        'lgb_cv_var': lgb_var,
        'lgb_top300_features': prev_lgb_features,  # 沿用历史特征列表
        'lr_final_features': prev_lr_features       # 沿用历史特征列表
    }

    return {
        'lr_model': sgd_model,
        'lgb_model': lgb_model,
        'scaler': scaler,
        'lr_metrics': lr_metrics,
        'lgb_metrics': lgb_metrics,
        'stability_metrics': stability_metrics,
        'lr_pred': lr_pred,
        'lgb_pred': lgb_pred,
        'lgb_proba': lgb_proba,
        'lgb_top300_features': prev_lgb_features,
        'lr_final_features': prev_lr_features
    }

# ===================== 2. 全量训练流程（替代滚动训练） =====================
def train_global_model(start_date, end_date):
    """
    使用指定时间区间内的全量历史数据进行训练与评估：
    - 先按 betting_cycle_date 划分为若干时间窗口，对每个窗口单独训练/验证，做“时序切片分析”；
    - 然后仍然使用全量数据做一次性训练，保持原有 best model 保存逻辑不变。
    """
    logger.info("===== 开始执行全量训练 =====")
    init_model_pred_tables(DB_PATH)

    # 1. 加载全量历史数据

    # 1. 加载全量历史数据
    full_df = load_historical_data(DB_PATH, start_date, end_date)
    if full_df.empty:
        st.error("❌ 未加载到有效训练数据")
        return pd.DataFrame()
    full_df = full_df.sort_values('betting_cycle_date').reset_index(drop=True)
    min_date = full_df['betting_cycle_date'].min()
    max_date = full_df['betting_cycle_date'].max()

    # 训练历史记录（这里仍然用与原来滚动训练相同的字段，方便后续复用可视化）
    train_history = pd.DataFrame(columns=[
        'window_start', 'window_end', 'sample_count', 'valid_predictors_count',
        'lr_f1', 'lgb_f1', 'lr_cv_var', 'lgb_cv_var', 'best_model', 'is_new_best'
    ])

    # 当前最优模型信息展示
    best_config = load_best_model_config()
    st.sidebar.markdown("### 🏆 当前最优模型")
    st.sidebar.info(f"""
    训练窗口：{best_config['window_end'] or '无'}
    模型日期：{best_config['model_date'] or '无'}
    LightGBM F1：{best_config['lgb_f1']:.3f}
    逻辑回归 F1：{best_config['lr_f1']:.3f}
    """)

    # 2. 只保留有赛果的比赛
    required_result_cols = ['home_goals', 'away_goals']
    if not all(col in full_df.columns for col in required_result_cols):
        missing_cols = [col for col in required_result_cols if col not in full_df.columns]
        st.error(f"❌ 全量数据缺少必要的赛果列: {missing_cols}，无法训练。")
        return train_history

    full_df = full_df[
        (full_df['home_goals'].notna()) & (full_df['home_goals'] != '') &
        (full_df['away_goals'].notna()) & (full_df['away_goals'] != '')
    ].copy()

    if len(full_df) < 50:
        st.warning(f"⚠️ 全量有效数据量过少（{len(full_df)}条），不足以进行训练。")
        return train_history

    # ==================== 3. 按时间窗口切分 + 窗口级训练/验证 ====================
    st.markdown("### ⏱ 按时间窗口的模型表现分析")

    # 3.1 构造时间窗口（按 betting_cycle_date 的唯一日期切成最多 4 段）
    full_df = full_df.sort_values('betting_cycle_date').reset_index(drop=True)
    date_series = full_df['betting_cycle_date'].dt.date
    unique_dates = sorted(date_series.unique())

    window_logs = []
    bucket_log_rows = []
    human_imp_list = []

    if len(unique_dates) >= 3:
        max_windows = 4
        # 至少 3 段，最多 4 段，但不能超过日期总数
        n_windows = min(max_windows, len(unique_dates))
        if n_windows < 3:
            n_windows = max(1, n_windows)
        date_splits = np.array_split(unique_dates, n_windows)

        for idx, date_chunk in enumerate(date_splits):
            if len(date_chunk) == 0:
                continue

            win_start = date_chunk[0]
            win_end = date_chunk[-1]

            window_mask = date_series.isin(date_chunk)
            df_win = full_df[window_mask].copy()

            if len(df_win) < 30:
                st.info(f"⚠️ 时间窗口 {idx + 1}（{win_start} ~ {win_end}）样本数过少（{len(df_win)}），跳过窗口训练。")
                continue

            try:
                X_w, y_w, feature_cols_w, current_predictor_ids_w = feature_engineering(df_win, is_training=True)
                valid_predictors_count_w = len(current_predictor_ids_w)
            except Exception as e:
                st.warning(f"⚠️ 时间窗口 {idx + 1}（{win_start} ~ {win_end}）特征工程失败：{str(e)}")
                continue

            current_pred_type_w = df_win['prediction_type'].iloc[0]
            st.write(f"🔧 窗口 {idx + 1} 训练：{win_start} ~ {win_end}，样本 {len(df_win)}，预测者 {valid_predictors_count_w}")

            # 使用窗口内数据从零开始训练模型
            train_result_w = train_base_models(X_w, y_w, current_predictor_ids_w, current_pred_type_w)

            # 3.2 计算该窗口的 logloss（验证集）
            y_test_w = train_result_w['y_test']
            lgb_proba_w = train_result_w['lgb_proba']
            try:
                lgb_logloss_w = log_loss(y_test_w, lgb_proba_w)
            except Exception:
                lgb_logloss_w = np.nan

            lgb_f1_w = train_result_w['metrics']['lgb_metrics']['f1']
            lr_f1_w = train_result_w['metrics']['lr_metrics']['f1']
            lr_cv_var_w = train_result_w['metrics']['stability_metrics']['lr_cv_var']
            lgb_cv_var_w = train_result_w['metrics']['stability_metrics']['lgb_cv_var']

            window_logs.append({
                'window_idx': idx + 1,
                'window_start': str(win_start),
                'window_end': str(win_end),
                'sample_count': len(df_win),
                'valid_predictors_count': valid_predictors_count_w,
                'lr_f1': lr_f1_w,
                'lgb_f1': lgb_f1_w,
                'lr_cv_var': lr_cv_var_w,
                'lgb_cv_var': lgb_cv_var_w,
                'lgb_logloss': lgb_logloss_w
            })

            # 3.3 各 bucket 高置信命中率（基于该窗口验证集）
            y_array_w = np.asarray(y_test_w)
            high_thresholds = [0.6, 0.7, 0.8, 0.9]
            for th in high_thresholds:
                mask = lgb_proba_w >= th
                selected = int(mask.sum())
                if selected > 0:
                    hit_rate = float(y_array_w[mask].mean())
                else:
                    hit_rate = np.nan
                bucket_log_rows.append({
                    'window_idx': idx + 1,
                    'window_start': str(win_start),
                    'window_end': str(win_end),
                    'threshold': th,
                    'sample_count': selected,
                    'hit_rate': hit_rate
                })

            # 3.4 人维度特征重要性（以特征名前缀 pred_ 作为“预测者相关”）
            lgb_top_features_w = train_result_w.get('lgb_top300_features', [])
            lgb_model_w = train_result_w.get('lgb_model', None)
            if lgb_model_w is not None and len(lgb_top_features_w) == len(lgb_model_w.feature_importances_):
                fi_df = pd.DataFrame({
                    'feature': lgb_top_features_w,
                    'importance': lgb_model_w.feature_importances_
                })
                # 只保留像 pred_123_xxx 这种带预测者ID前缀的特征，避免把 pred_type_identifier 之类全局特征算做人特征
                human_fi = fi_df[
                    fi_df['feature'].astype(str).str.match(r'^pred_\d+_')
                ].copy()
                if not human_fi.empty:
                    human_fi = human_fi.sort_values('importance', ascending=False).head(15)
                    human_fi['window_idx'] = idx + 1
                    human_imp_list.append(human_fi)

        # 3.x 汇总可视化
        if len(window_logs) > 0:
            logs_df = pd.DataFrame(window_logs)
            logs_df_sorted = logs_df.sort_values('window_end')

            # 3.x.1 logloss vs 时间
            st.markdown("#### 📉 各时间窗口 LightGBM 验证集 logloss 随时间变化")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(logs_df_sorted['window_end'], logs_df_sorted['lgb_logloss'], marker='o', linestyle='-')
            ax.set_xlabel('窗口结束日期', fontsize=12)
            ax.set_ylabel('LightGBM 验证集 logloss', fontsize=12)
            ax.set_title('logloss vs 时间（按 betting_cycle_date 窗口）', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)
            for i, row in logs_df_sorted.iterrows():
                if not np.isnan(row['lgb_logloss']):
                    ax.text(row['window_end'], row['lgb_logloss'] + 0.005,
                            f"{row['lgb_logloss']:.3f}", ha='center', fontsize=9)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("#### 📊 各时间窗口基础表现概览")
            st.dataframe(logs_df.round(3), width='stretch')

            # 持久化窗口级日志
            logs_save_path = os.path.join(
                METRICS_DIR,
                f'window_cv_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
            logs_df.to_csv(logs_save_path, index=False, encoding='utf-8-sig')

        if len(bucket_log_rows) > 0:
            bucket_df = pd.DataFrame(bucket_log_rows)
            bucket_df_display = bucket_df.copy()
            def _fmt_hit_rate(x):
                return f"{x * 100:.1f}%" if (x is not None and not np.isnan(x)) else "无样本"
            bucket_df_display['hit_rate'] = bucket_df_display['hit_rate'].apply(_fmt_hit_rate)

            st.markdown("#### 🎯 各时间窗口高置信度 bucket 真实命中率")
            st.dataframe(bucket_df_display, width='stretch')

        if len(human_imp_list) > 0:
            st.markdown("#### 🧍 各时间窗口“人维度”特征重要性（Top15，仅特征名前缀 pred_）")
            for human_df in human_imp_list:
                win_idx = int(human_df['window_idx'].iloc[0])
                # 找到对应窗口的起止日期
                win_row = None
                if 'logs_df' in locals():
                    match_rows = logs_df[logs_df['window_idx'] == win_idx]
                    if not match_rows.empty:
                        win_row = match_rows.iloc[0]
                win_title_suffix = ""
                if win_row is not None:
                    win_title_suffix = f"（{win_row['window_start']} ~ {win_row['window_end']}）"

                fig, ax = plt.subplots(figsize=(8, 6))
                human_plot_df = human_df.sort_values('importance', ascending=True)
                sns.barplot(x='importance', y='feature', data=human_plot_df, ax=ax, color='#e67e22')
                ax.set_xlabel('特征重要性得分', fontsize=12)
                ax.set_ylabel('特征名称', fontsize=10)
                ax.set_title(f'窗口 {win_idx} 人维度 Top15 特征{win_title_suffix}', fontsize=14, fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

    else:
        st.info("⚠️ betting_cycle_date 唯一日期少于 3 天，暂不进行时间窗口切分分析。")

    # ==================== 4. 使用全量数据从零开始训练基础模型（保持原逻辑） ====================
    try:
        X, y, feature_cols, current_predictor_ids = feature_engineering(full_df, is_training=True)
        valid_predictors_count = len(current_predictor_ids)
        logger.info(f"特征工程完成：总特征数{len(feature_cols)}，有效预测者数{valid_predictors_count}")
    except Exception as e:
        st.error(f"特征工程失败：{str(e)}")
        return train_history

    st.write(
        f"✅ 全量训练数据量：{len(full_df)}条，"
        f"有效预测者数：{valid_predictors_count}，总特征数：{len(feature_cols)}"
    )

    current_pred_type = full_df['prediction_type'].iloc[0]
    st.write("🔧 使用全量数据从零开始训练基础模型...")
    train_result = train_base_models(X, y, current_predictor_ids, current_pred_type)

    # 5. 模型迭代对比 + 仅对新best model做可视化
    new_lgb_f1 = train_result['metrics']['lgb_metrics']['f1']
    new_lr_f1 = train_result['metrics']['lr_metrics']['f1']
    historical_best_lgb = best_config["lgb_f1"]

    # 新策略：考虑“性能 + 时效性”的双重约束
    # 规则：
    #   - 如果还没有历史最优模型（historical_best_lgb <= 0），任何新模型都视为最优；
    #   - 否则：
    #       1）只要新模型的 F1 没有比历史最优差太多（<= base_margin），就允许用新模型覆盖旧模型；
    #       2）如果新模型明显更差，则检查历史最优模型是否“过旧”（超过 force_refresh_days），
    #          若过旧则为了时效性强制刷新为新模型。
    base_margin = 0.005        # 允许新模型比历史最优略低的性能容忍度
    force_refresh_days = 10    # 历史最优模型允许“不过期”的天数阈值

    # 默认假设本次不是新最优，并初始化 days_gap
    is_new_best = False
    days_gap = None

    if historical_best_lgb <= 0:
        # 没有历史最优模型时，当前一定是最优
        is_new_best = True
        days_gap = 0
    else:
        # 1）性能维度：新模型没有比历史最优差太多，允许覆盖
        if new_lgb_f1 >= historical_best_lgb - base_margin:
            is_new_best = True
        else:
            # 2）时效维度：历史最优模型是否已经“过旧”
            old_end_str = best_config.get("window_end", "")
            try:
                old_end = datetime.strptime(old_end_str, "%Y-%m-%d").date()
                if pd.isna(max_date):
                    # 若当前训练数据最大日期异常，则视作极旧，触发刷新
                    days_gap = 9999
                else:
                    days_gap = (max_date.date() - old_end).days
            except Exception:
                # window_end 缺失或解析失败时，视作极旧
                days_gap = 9999

            if days_gap is not None and days_gap >= force_refresh_days:
                is_new_best = True
                logger.info(
                    f"历史最优模型已过期 {days_gap} 天，"
                    f"即使新模型 F1 略低 (new={new_lgb_f1:.3f}, best={historical_best_lgb:.3f})，"
                    f"仍强制将当前模型设为新的最优模型"
                )

    date_tag = max_date.strftime("%Y%m%d") if not pd.isna(max_date) else end_date.replace('-', '')

    if is_new_best or not best_config.get("model_date"):
        # 更新最优模型配置
        new_best_config = {
            "model_date": train_result['model_date'],
            "lgb_f1": new_lgb_f1,
            "lr_f1": new_lr_f1,
            "lgb_path": os.path.join(MODEL_DIR, f'lgb_model_{train_result["model_date"]}.pkl'),
            "lr_path": os.path.join(MODEL_DIR, f'lr_model_{train_result["model_date"]}.pkl'),
            "scaler_path": os.path.join(MODEL_DIR, f'scaler_{train_result["model_date"]}.pkl'),
            "lgb_features_path": os.path.join(MODEL_DIR, f'lgb_features_{train_result["model_date"]}.pkl'),
            "lr_features_path": os.path.join(MODEL_DIR, f'lr_features_{train_result["model_date"]}.pkl'),
            "model_predictors": current_predictor_ids,
            "prediction_type": train_result['prediction_type'],
            "window_end": max_date.strftime("%Y-%m-%d") if not pd.isna(max_date) else end_date
        }
        save_best_model_config(new_best_config)
        best_config = new_best_config
        if historical_best_lgb > 0:
            st.success(
                f"🎉 新模型被设为最优模型！LightGBM F1 从 {historical_best_lgb:.3f} "
                f"更新为 {new_lgb_f1:.3f}（允许下降阈值 {base_margin:.3f}）"
            )
        else:
            st.success(
                f"🎉 已生成首个最优模型！LightGBM F1 = {new_lgb_f1:.3f}"
            )
    else:
        delta = new_lgb_f1 - historical_best_lgb
        extra_msg = ""
        # 当 days_gap 被正确计算时，补充展示“距离历史最优窗口已过去多少天”
        if days_gap is not None:
            extra_msg = f"，当前训练数据结束日期距离历史最优模型窗口结束已过去 {days_gap} 天"
        st.info(
            f"ℹ️ 新模型暂未替换历史最优（当前 LightGBM F1: {new_lgb_f1:.3f}，"
            f"历史最优: {historical_best_lgb:.3f}，差值 {delta:+.3f}，"
            f"允许下降阈值 {base_margin:.3f}{extra_msg}）"
        )

    # 每次训练都生成可视化评估报告（全量视角）
    visualize_metrics(train_result, date_tag)

    # 6. 清理冗余模型文件（只保留最优 + 最近2个）
    clean_old_models()

    # 7. 记录全量训练历史（单行，保持原结构）
    best_model = 'LightGBM' if new_lgb_f1 > new_lr_f1 else '逻辑回归'
    history_entry = pd.DataFrame({
        'window_start': [min_date.strftime('%Y-%m-%d') if not pd.isna(min_date) else start_date],
        'window_end': [max_date.strftime('%Y-%m-%d') if not pd.isna(max_date) else end_date],
        'sample_count': [len(full_df)],
        'valid_predictors_count': [len(current_predictor_ids)],
        'lr_f1': [new_lr_f1],
        'lgb_f1': [new_lgb_f1],
        'lr_cv_var': [train_result['metrics']['stability_metrics']['lr_cv_var']],
        'lgb_cv_var': [train_result['metrics']['stability_metrics']['lgb_cv_var']],
        'best_model': [best_model],
        'is_new_best': [1 if is_new_best else 0]
    })
    train_history = pd.concat([train_history, history_entry], ignore_index=True)

    # 保存训练历史
    train_history.to_csv(
        os.path.join(METRICS_DIR, f'training_history_{datetime.now().strftime("%Y%m%d")}.csv'),
        index=False,
        encoding='utf-8-sig'
    )

    return train_history

# ===================== 3. 当日推理 =====================
def predict_today(target_date):
    st.markdown(f"### 🎯 当日推理：{target_date}")
    # 规范化推理日期字符串，后续写入统计表时复用
    target_date_str = str(target_date)
    # 数据库表初始化校验
    if not init_model_pred_tables(DB_PATH):
        st.error("❌ 数据库表初始化失败")
        return pd.DataFrame(), pd.DataFrame()

    # 1. 仅使用最佳模型进行推理
    best_config = load_best_model_config()
    model_dates = []

    # 推理阶段优先使用“最佳模型”单模型进行预测
    if best_config.get("model_date"):
        model_dates = [best_config["model_date"]]
    else:
        # 若尚未产生最佳模型，则退化为使用最近一次训练产生的模型
        recent_dates = get_recent_model_dates(top_n=1)
        if recent_dates:
            model_dates = [recent_dates[0]]
            st.warning("⚠️ 未找到最优模型配置，临时使用最近一次训练模型进行推理")
        else:
            st.error("❌ 当前不存在任何可用模型，请先完成一次全量训练")
            return pd.DataFrame(), pd.DataFrame()

    st.info(f"本次推理使用的模型日期：{model_dates[0]}（单一最佳模型）")

    # 2. 加载推理数据
    pred_df = load_prediction_data(DB_PATH, target_date)
    if pred_df.empty:
        st.error("❌ 未加载到预测数据（检查日期或数据库连接）")
        return pd.DataFrame(), pd.DataFrame()

    # <--- 新增代码：关联 match 表获取 match_no
    conn = get_db_connection(DB_PATH)
    if conn:
        try:
            # 查询 match 表，获取 match_id 和 match_no
            match_query = "SELECT match_id, match_no FROM match;"
            match_df = pd.read_sql(match_query, conn)

            # 将 match_no 合并到 pred_df 中
            # 使用左连接，确保即使没有找到对应的 match_no，pred_df 的记录也不会丢失
            pred_df = pd.merge(pred_df, match_df, on='match_id', how='left')

            # 处理可能的缺失值（如果有的话）
            pred_df['match_no'] = pred_df['match_no'].fillna('N/A')

        except Exception as e:
            st.warning(f"⚠️ 关联 match 表获取比赛编号失败: {e}")
        finally:
            conn.close()

    current_predictor_ids = pred_df['predictor_id'].unique().tolist()
    conn = get_db_connection(DB_PATH)
    if conn:
        all_pids = [p['predictor_id'] for p in conn.execute("SELECT DISTINCT predictor_id FROM predictor").fetchall()]
        current_predictor_ids = list(set(current_predictor_ids + all_pids))
        conn.close()

    # 3. 特征工程
    try:
        X, _, feature_cols, _ = feature_engineering(pred_df, is_training=False)
        # <--- 新增调试代码
        st.subheader("🔍 调试信息：特征矩阵")
        st.write(f"特征矩阵形状: {X.shape}")
        st.write("特征矩阵前5行:")
        st.dataframe(X.head())
        # <--- 调试代码结束
    except Exception as e:
        st.error(f"❌ 特征工程失败：{str(e)[:150]}")
        return pd.DataFrame(), pd.DataFrame()

    # 4. 多模型推理
    model_preds = []

    # <--- 关键修改 1: 在循环前初始化变量为 None
    lgb_model = None
    lgb_features = None
    X_aligned_for_viz = None  # 同样为可视化用的 X_aligned 初始化

    for idx, model_date in enumerate(model_dates):
        try:
            # 加载模型及配套配置
            temp_lgb_model = joblib.load(os.path.join(MODEL_DIR, f'lgb_model_{model_date}.pkl'))
            lr_model = joblib.load(os.path.join(MODEL_DIR, f'lr_model_{model_date}.pkl'))
            scaler = joblib.load(os.path.join(MODEL_DIR, f'scaler_{model_date}.pkl'))
            temp_lgb_features = joblib.load(os.path.join(MODEL_DIR, f'lgb_features_{model_date}.pkl'))
            lr_features = joblib.load(os.path.join(MODEL_DIR, f'lr_features_{model_date}.pkl'))
            model_predictors = joblib.load(os.path.join(MODEL_DIR, f'model_predictors_{model_date}.pkl'))

            # 特征对齐
            X_aligned = align_features_with_predictors(
                X=X,
                current_predictor_ids=current_predictor_ids,
                model_predictor_ids=model_predictors,
                model_features=temp_lgb_features
            )

            # 特征筛选和缩放
            X_lgb = X_aligned[temp_lgb_features].copy()
            X_lr = X_aligned[lr_features].copy() if all(col in X_aligned.columns for col in lr_features) else X_aligned
            X_lr_scaled = scaler.transform(X_lr).astype('float64')

            # 计算置信度
            lr_proba = lr_model.predict_proba(X_lr_scaled)[:, 1]
            lgb_proba = temp_lgb_model.predict_proba(X_lgb)[:, 1]

            # <--- 新增调试代码
            st.write(f"📊 模型 {idx + 1} 预测概率:")
            st.write(f"LR 模型概率 (前10个): {lr_proba[:10]}")
            st.write(f"LGB 模型概率 (前10个): {lgb_proba[:10]}")
            # <--- 调试代码结束

            # 线上置信度：当前版本改为纯 LightGBM 概率
            model_confidence = lgb_proba

            model_preds.append(pd.DataFrame({
                'predictor_id': pred_df['predictor_id'].values,
                'match_id': pred_df['match_id'].values,
                'match_no': pred_df['match_no'].values,  # <--- 关键：在这里加入 match_no
                'betting_cycle_date': pred_df['betting_cycle_date'],
                'original_term': pred_df['original_term'].values,
                'prediction_type': pred_df['prediction_type'].values,
                f'confidence_model_{idx + 1}': model_confidence
            }))
            st.info(f"✅ 加载模型{idx + 1}（日期：{model_date}）成功")

            # <--- 关键修改 2: 只有当模型成功加载时，才更新用于可视化的全局变量
            # 我们用最后一个成功加载的模型来做可视化
            lgb_model = temp_lgb_model
            lgb_features = temp_lgb_features
            X_aligned_for_viz = X_aligned

        except Exception as e:
            import traceback
            st.warning(f"⚠️ 加载模型{idx + 1}（日期：{model_date}）失败：{str(e)[:100]}")
            st.code(traceback.format_exc()[:500], language='python')
            continue

    # 校验：若无可用模型，直接返回
    if len(model_preds) == 0:
        st.error("❌ 无可用模型完成推理，请检查模型文件或路径")
        return pd.DataFrame(), pd.DataFrame()

    # 5. 预测结果融合
    merged_pred = model_preds[0]
    for p in model_preds[1:]:
        merged_pred = pd.merge(
            merged_pred,
            p,
            on=['predictor_id', 'match_id', 'betting_cycle_date', 'original_term', 'prediction_type', 'match_no'],
            how='outer'
        )
    confidence_cols = [col for col in merged_pred.columns if col.startswith('confidence_model_')]
    merged_pred['confidence'] = merged_pred[confidence_cols].mean(axis=1).round(3)

    # 针对“总进球数”玩法的组合投注做一层置信度惩罚，避免 2/3球、3/4球 等与超宽区间组合拿到类似置信度
    merged_pred = adjust_goal_combo_confidence(merged_pred)

    # 新增：为 Top2 统计准备全局最高置信度的前2条记录（跨比赛与玩法）
    merged_pred_top2 = (
        merged_pred
        .sort_values('confidence', ascending=False)
        .head(2)
        .copy()
    )

    # 6. 先算每场比赛、每种玩法最终选取的方案（同一比赛内每种玩法取置信度最高的一条）
    best_idx = (
        merged_pred
        .groupby(['match_id', 'prediction_type'])['confidence']
        .idxmax()
        .dropna()
        .astype(int)
    )
    match_best_all = merged_pred.loc[best_idx].reset_index(drop=True)

    # 按置信度排序，Top10 用于后面的图表 + 卡片展示
    match_best_pred = match_best_all.sort_values('confidence', ascending=False).head(10)

    # 日期展示字段
    match_best_all['display_date'] = match_best_all['betting_cycle_date'].dt.strftime('%Y-%m-%d')
    match_best_pred['display_date'] = match_best_pred['betting_cycle_date'].dt.strftime('%Y-%m-%d')

    # 合并盘口信息（如果有）
    if 'handicap_value' in pred_df.columns:
        match_info = pred_df[['match_id', 'handicap_value']].drop_duplicates('match_id')
        match_best_all = pd.merge(match_best_all, match_info, on='match_id', how='left')
        match_best_pred = pd.merge(match_best_pred, match_info, on='match_id', how='left')

    # 填默认值
    match_best_all = match_best_all.fillna({
        'handicap_value': '无'
    })
    match_best_pred = match_best_pred.fillna({
        'handicap_value': '无'
    })

    # 7. 新增：预测可解释性可视化
    st.markdown("### 📋 全部比赛最终预测（每场每玩法取置信度最高的方案）")

    all_display_cols = ['display_date', 'match_no', 'prediction_type', 'original_term', 'confidence']
    existing_cols = [c for c in all_display_cols if c in match_best_all.columns]

    st.dataframe(
        match_best_all[existing_cols].sort_values('confidence', ascending=False).reset_index(drop=True),
        width='stretch'
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'预测分析报告（{target_date}）', fontsize=16, fontweight='bold', y=0.98)

    # 7.1 Top10预测置信度排序
    y_labels = [f"比赛#{row['match_id']}" for _, row in match_best_pred.iterrows()]
    axes[0, 0].barh(range(len(match_best_pred)), match_best_pred['confidence'], color='#e74c3c', alpha=0.8)
    axes[0, 0].set_yticks(range(len(match_best_pred)))
    axes[0, 0].set_yticklabels(y_labels, fontsize=10)
    axes[0, 0].set_xlabel('置信度', fontsize=12)
    axes[0, 0].set_title('Top10预测置信度排序', fontweight='bold')
    axes[0, 0].axvline(x=0.8, color='green', linestyle='--', alpha=0.8, label='高置信阈值（0.8）')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='x', alpha=0.3)

    # 7.2 置信度分布
    axes[0, 1].hist(
        match_best_pred['confidence'],
        bins=5,
        color='#3498db',
        alpha=0.8,
        edgecolor='black'
    )
    axes[0, 1].set_xlabel('置信度区间', fontsize=12)
    axes[0, 1].set_ylabel('预测数量', fontsize=12)
    axes[0, 1].set_title('Top10预测置信度分布', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 7.3 Top10 置信度走势（当前版本仅使用单模型，不再展示模型共识度）
    axes[1, 0].plot(
        range(1, len(match_best_pred) + 1),
        match_best_pred['confidence'],
        marker='o',
        linestyle='-',
        linewidth=2,
        alpha=0.8
    )
    axes[1, 0].set_xlabel('Top 排名（1 = 最高置信度）', fontsize=12)
    axes[1, 0].set_ylabel('置信度', fontsize=12)
    axes[1, 0].set_title('Top10 置信度走势', fontweight='bold')
    axes[1, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='高置信阈值（0.8）')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # <--- 关键修改 3: 在使用 lgb_model 和 lgb_features 之前，先检查它们是否为 None
    if lgb_model is not None and lgb_features is not None and X_aligned_for_viz is not None:
        # 7.4 最高置信度预测的核心特征贡献
        top_pred = match_best_pred.iloc[0]
        # 使用为可视化保存的 X_aligned_for_viz
        X_top = X_aligned_for_viz[X_aligned_for_viz.index == top_pred.name][lgb_features].copy()
        feature_importance = pd.DataFrame({
            'feature': lgb_features[:10],
            'importance': lgb_model.feature_importances_[:10]
        }).sort_values('importance', ascending=True)
        axes[1, 1].barh(
            feature_importance['feature'],
            feature_importance['importance'],
            color='#2ecc71',
            alpha=0.8
        )
        axes[1, 1].set_xlabel('特征重要性得分', fontsize=12)
        axes[1, 1].set_title(f'最高置信度预测（比赛#{top_pred["match_id"]}）核心特征', fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
    else:
        # 如果变量未被正确初始化，则在第四个子图上显示提示信息
        axes[1, 1].text(0.5, 0.5, '无法加载LGB模型以生成特征重要性图',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[1, 1].transAxes,
                        fontsize=14,
                        color='red')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')  # 关闭坐标轴

    # 调整布局+展示图表
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)


    # 7.x 新增：自动入库模型预测明细（每场每玩法最佳 + 全局Top2）
    try:
        conn = get_db_connection(DB_PATH)
        if conn:
            cursor = conn.cursor()

            def _insert_model_record(row):
                """将一条模型预测结果写入 model_prediction_records（若已存在则忽略）"""
                pred_dt = row.get('betting_cycle_date')
                # 兼容 Timestamp / datetime / 字符串
                if pd.isna(pred_dt):
                    pred_date_str = str(target_date)
                else:
                    if hasattr(pred_dt, 'strftime'):
                        pred_date_str = pred_dt.strftime('%Y-%m-%d')
                    else:
                        pred_date_str = str(pred_dt)[:10]

                cursor.execute(
                    """
                    INSERT OR IGNORE INTO model_prediction_records
                    (pred_date, match_id, original_term, prediction_type, confidence)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        pred_date_str,
                        int(row['match_id']),
                        str(row['original_term']),
                        str(row['prediction_type']),
                        float(row['confidence'])
                    )
                )

            # 7.x.1 每场比赛、每种玩法置信度最高的方案
            for _, r in match_best_all.iterrows():
                _insert_model_record(r)

            # 7.x.2 当天全局置信度最高的 Top2 方案（跨比赛 + 跨玩法）
            for _, r in merged_pred_top2.iterrows():
                _insert_model_record(r)

            # 7.x.3 记录当日全局置信度最高的 Top2 预测到 model_pred_stats_top2 明细表
            # 表结构在 utils.init_model_pred_tables 中定义，包含：pred_date, match_id,
            # prediction_type, original_term, confidence, bucket_name 等字段，
            # 用于每天仅保留全局 Top2 预测明细（不再存 predictor_id）。
            try:
                for _, r in merged_pred_top2.iterrows():
                    # 使用 target_date_str 作为预测日期键，确保“一天只保留全局 Top2”
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO model_pred_stats_top2
                        (pred_date, match_id, prediction_type, original_term, confidence, bucket_name)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            target_date_str,
                            int(r['match_id']),
                            str(r['prediction_type']),
                            str(r['original_term']),
                            float(r['confidence']),
                            'TOP2'
                        )
                    )
            except Exception as e:
                st.warning(f"⚠️ 写入 model_pred_stats_top2 Top2 记录失败：{str(e)[:150]}")

            conn.commit()
            st.success("✅ 已将当日模型预测明细自动写入 model_prediction_records（去重后）")
    except Exception as e:
        st.warning(f"⚠️ 自动写入模型预测明细失败：{str(e)[:150]}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

    # 8. Top10预测展示（带“加入记录”按钮）
    # ... (这部分代码没有问题，此处省略)
    st.markdown("### 🏆 当日Top10高置信度预测")

    def get_confidence_style(conf):
        if conf >= 0.8:
            return 'background-color: #d4edda; color: #155724; font-weight: bold'
        elif conf >= 0.6:
            return 'background-color: #fff3cd; color: #856404; font-weight: bold'
        else:
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold'

    for idx, (_, row) in enumerate(match_best_pred.iterrows()):
        cols = st.columns([1.2, 1, 2, 2, 1.4])
        with cols[0]:
            st.write(f"📅 {row['display_date']}")
        with cols[1]:
            st.write(f"#{row['match_no']}")
        with cols[2]:
            st.write(row['prediction_type'])
        with cols[3]:
            st.write(row['original_term'])
        with cols[4]:
            st.markdown(
                f"<div style='{get_confidence_style(row['confidence'])}; padding: 4px; border-radius: 4px'>{row['confidence']:.3f}</div>",
                unsafe_allow_html=True
            )
        st.markdown("---")

    # 9. 模型历史表现（基于 Top2 记录）
    st.markdown("### 📈 模型历史表现（Top2 命中率）")

    total_rows = 0
    labeled_rows = 0
    hit_rows = 0

    conn = get_db_connection(DB_PATH)
    if conn:
        try:
            cursor = conn.cursor()
            # 表总记录数
            cursor.execute("SELECT COUNT(*) FROM model_pred_stats_top2")
            row = cursor.fetchone()
            total_rows = row[0] if row and row[0] is not None else 0

            # is_hit 字段已填写的记录数（不为 NULL）
            cursor.execute("SELECT COUNT(*) FROM model_pred_stats_top2 WHERE is_hit IS NOT NULL")
            row = cursor.fetchone()
            labeled_rows = row[0] if row and row[0] is not None else 0

            # 命中记录数（is_hit = 1）
            cursor.execute("SELECT COUNT(*) FROM model_pred_stats_top2 WHERE is_hit = 1")
            row = cursor.fetchone()
            hit_rows = row[0] if row and row[0] is not None else 0

        except Exception as e:
            st.warning(f"⚠️ 统计 Top2 命中率失败：{str(e)[:150]}")
        finally:
            conn.close()

    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.metric("Top2 已结算次数", labeled_rows, delta=0, help="is_hit 字段不为 NULL 的记录数（已结算）")
    with stat_cols[1]:
        st.metric("Top2 命中次数", hit_rows, delta=0, help="is_hit = 1 的记录数（命中次数）")
    with stat_cols[2]:
        acc = hit_rows / labeled_rows if labeled_rows > 0 else 0.0
        st.metric("Top2 命中率（已结算）", f"{acc:.1%}", delta=0, help="命中次数 / 已结算次数（is_hit=1 / is_hit不为NULL）")

    # 10. 可选：自动入库Top2高置信预测
    # ... (这部分代码没有问题，此处省略)
    # if len(match_best_pred) >= 2:
    #     auto_top2 = match_best_pred.head(2)
    #     if save_prediction_to_db(DB_PATH, auto_top2):
    #         st.success(f"✅ 已自动将Top2高置信预测存入数据库")
    #     else:
    #         st.warning("⚠️ 自动入库Top2预测失败，可手动点击“加入记录”补录")
    # else:
    #     st.warning("⚠️ 可用预测不足2条，跳过自动入库")

    return match_best_pred, pd.DataFrame()

# ===================== 4. Streamlit可视化评估（适配全量特征） =====================
def visualize_metrics(train_result, date_tag):
    """可视化基础性能+特征重要性+特征选择结果"""
    lr_metrics = train_result['metrics']['lr_metrics']
    lgb_metrics = train_result['metrics']['lgb_metrics']
    stability_metrics = train_result['metrics']['stability_metrics']
    y_test = train_result['y_test']
    lr_pred = train_result['lr_pred']
    lgb_pred = train_result['lgb_pred']
    lgb_proba = train_result['lgb_proba']
    lr_proba = train_result.get('lr_proba', None)
    ensemble_proba = train_result.get('ensemble_proba', None)
    feature_cols = train_result['metrics']['feature_cols']
    lgb_top300_features = train_result.get('lgb_top300_features', [])
    lr_final_features = train_result.get('lr_final_features', [])


    # 创建4x2子图
    fig, axes = plt.subplots(4, 2, figsize=(18, 22))
    fig.suptitle(f'模型评估报告（{date_tag}）', fontsize=18, fontweight='bold', y=0.98)

    # ---------------------- 1. 基础性能指标对比 ----------------------
    metrics_names = ['准确率', '精确率', '召回率', 'F1值']
    lr_values = [lr_metrics['accuracy'], lr_metrics['precision'], lr_metrics['recall'], lr_metrics['f1']]
    lgb_values = [lgb_metrics['accuracy'], lgb_metrics['precision'], lgb_metrics['recall'], lgb_metrics['f1']]

    x = np.arange(len(metrics_names))
    width = 0.35
    axes[0, 0].bar(x - width / 2, lr_values, width, label='逻辑回归', color='#3498db', alpha=0.8)
    axes[0, 0].bar(x + width / 2, lgb_values, width, label='LightGBM', color='#e74c3c', alpha=0.8)
    axes[0, 0].set_xlabel('指标类型', fontsize=12)
    axes[0, 0].set_ylabel('数值', fontsize=12)
    axes[0, 0].set_title('基础性能指标对比', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics_names)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    # 标注数值
    for i, (lr_val, lgb_val) in enumerate(zip(lr_values, lgb_values)):
        axes[0, 0].text(i - width / 2, lr_val + 0.01, f'{lr_val:.3f}', ha='center', fontsize=10)
        axes[0, 0].text(i + width / 2, lgb_val + 0.01, f'{lgb_val:.3f}', ha='center', fontsize=10)

    # ---------------------- 2. LightGBM混淆矩阵 ----------------------
    cm = np.array(lgb_metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar_kws={'label': '预测方案数'})
    axes[0, 1].set_xlabel('预测标签', fontsize=12)
    axes[0, 1].set_ylabel('真实标签', fontsize=12)
    axes[0, 1].set_title('预测方案命中情况混淆矩阵（0=未命中，1=命中）', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticklabels(['未命中', '命中'])
    axes[0, 1].set_yticklabels(['未命中', '命中'])

    # ---------------------- 3. 稳定性指标 ----------------------
    model_names = ['逻辑回归', 'LightGBM']
    cv_vars = [stability_metrics['lr_cv_var'], stability_metrics['lgb_cv_var']]
    cv_means = [stability_metrics['lr_cv_f1_mean'], stability_metrics['lgb_cv_f1_mean']]

    bars = axes[1, 0].bar(model_names, cv_vars, color=['#3498db', '#e74c3c'], alpha=0.8)
    axes[1, 0].set_xlabel('模型类型', fontsize=12)
    axes[1, 0].set_ylabel('F1值方差（越小越稳定）', fontsize=12)
    axes[1, 0].set_title('5折交叉验证稳定性', fontsize=14, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='稳定阈值（0.1）')
    axes[1, 0].legend()
    # 标注数值
    for bar, var, mean in zip(bars, cv_vars, cv_means):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                        f'方差：{var:.3f}\n均值F1：{mean:.3f}', ha='center', fontsize=10)

    # ---------------------- 3.1 概率层面的 logloss 对比（LGB vs Ensemble） ----------------------
    logloss_rows = []

    try:
        lgb_logloss = log_loss(y_test, lgb_proba)
        logloss_rows.append({'模型': 'LightGBM', 'logloss': lgb_logloss})
    except Exception:
        lgb_logloss = None

    if ensemble_proba is not None:
        try:
            ens_logloss = log_loss(y_test, ensemble_proba)
            logloss_rows.append({'模型': 'Ensemble(LGB+LR)/2', 'logloss': ens_logloss})
        except Exception:
            pass

    if lr_proba is not None:
        try:
            lr_logloss = log_loss(y_test, lr_proba)
            logloss_rows.append({'模型': 'Logistic Regression', 'logloss': lr_logloss})
        except Exception:
            pass

    if len(logloss_rows) > 0:
        logloss_df = pd.DataFrame(logloss_rows)
        st.markdown("#### 📉 概率层面的 logloss 对比（验证集）")
        st.dataframe(logloss_df.round(4), width='stretch')

    # ---------------------- 4. 预测分布对比 ----------------------
    axes[1, 1].hist(lr_pred, bins=2, alpha=0.6, label='逻辑回归', color='#3498db', density=True, rwidth=0.7)
    axes[1, 1].hist(lgb_pred, bins=2, alpha=0.6, label='LightGBM', color='#e74c3c', density=True, rwidth=0.7)
    axes[1, 1].hist(y_test, bins=2, alpha=0.4, label='真实结果', color='#2ecc71', density=True, rwidth=0.7)
    axes[1, 1].set_xlabel('标签（0=未命中，1=命中）', fontsize=12)
    axes[1, 1].set_ylabel('密度', fontsize=12)
    axes[1, 1].set_title('预测方案分布与真实结果对比', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['未命中', '命中'])
    axes[1, 1].legend()

    # ---------------------- 5. LightGBM Top20特征重要性 ----------------------
    if len(lgb_top300_features) > 0 and hasattr(train_result['lgb_model'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': lgb_top300_features[:20],
            'importance': train_result['lgb_model'].feature_importances_[:20]
        }).sort_values('importance', ascending=True)
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=axes[2, 0], color='#e74c3c')
        axes[2, 0].set_xlabel('特征重要性得分', fontsize=12)
        axes[2, 0].set_ylabel('特征名称', fontsize=10)
        axes[2, 0].set_title('LightGBM Top20核心特征（基于全部特征）', fontsize=14, fontweight='bold')
        axes[2, 0].grid(axis='x', alpha=0.3)

    # ---------------------- 6. PR曲线+投注阈值选择 ----------------------
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, thresholds = precision_recall_curve(y_test, lgb_proba)
    average_precision = average_precision_score(y_test, lgb_proba)

    axes[2, 1].plot(recall, precision, color='#e74c3c', lw=2, label=f'平均精确率（AP）={average_precision:.3f}')
    axes[2, 1].set_xlabel('召回率（预测方案命中覆盖率）', fontsize=12)
    axes[2, 1].set_ylabel('精确率（预测方案实际命中率）', fontsize=12)
    axes[2, 1].set_title('预测方案PR曲线与投注阈值选择', fontsize=14, fontweight='bold')
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.3)

    # 标注核心阈值（精确率≥0.8）
    target_precision = 0.8
    closest_idx = np.argmin(np.abs(precision - target_precision))
    best_threshold = 0.1
    if closest_idx < len(thresholds):
        best_threshold = thresholds[closest_idx]
        axes[2, 1].scatter(recall[closest_idx], precision[closest_idx], color='red', s=80, zorder=5)
        axes[2, 1].annotate(
            f'推荐阈值={best_threshold:.3f}\n精确率={precision[closest_idx]:.3f}\n覆盖率={recall[closest_idx]:.3f}',
            xy=(recall[closest_idx], precision[closest_idx]),
            xytext=(recall[closest_idx] + 0.1, precision[closest_idx] - 0.1),
            arrowprops=dict(arrowstyle='->', color='red', lw=2)
        )

    # ---------------------- 6.1 高置信度区间真实命中率统计（LGB vs Ensemble） ----------------------
    # 在验证集上统计不同阈值下的样本数和真实命中率，帮助判断“只押高置信度方案”的实战价值
    y_array = np.asarray(y_test)
    high_thresholds = [0.6, 0.7, 0.8, 0.9]

    def collect_bucket_stats(proba, name):
        rows = []
        for th in high_thresholds:
            mask = proba >= th
            selected = int(mask.sum())
            if selected > 0:
                hit_rate = float(y_array[mask].mean())
            else:
                hit_rate = np.nan
            rows.append({
                '模型': name,
                '阈值': th,
                '样本数': selected,
                '真实命中率': hit_rate if not np.isnan(hit_rate) else None
            })
        return rows

    all_rows = []
    # LightGBM 必填
    all_rows += collect_bucket_stats(lgb_proba, 'LightGBM')

    # Ensemble 可选
    if ensemble_proba is not None:
        all_rows += collect_bucket_stats(ensemble_proba, 'Ensemble(LGB+LR)/2')

    # LR 概率（更多是 sanity check）
    if lr_proba is not None:
        all_rows += collect_bucket_stats(lr_proba, 'Logistic Regression')

    if len(all_rows) > 0:
        stats_df = pd.DataFrame(all_rows)
        stats_df_display = stats_df.copy()
        stats_df_display['真实命中率'] = stats_df_display['真实命中率'].apply(
            lambda x: f"{x * 100:.1f}%" if x is not None else "无样本"
        )
        st.markdown("#### 🎯 高置信度区间真实命中率（验证集 LGB vs Ensemble 对比）")
        st.dataframe(stats_df_display, width='stretch')

    # ---------------------- 7. 特征选择结果对比 ----------------------
    feature_count_data = pd.DataFrame({
        '模型': ['全部特征', 'Logistic Regression最终'],
        '特征数量': [len(feature_cols), len(lr_final_features)]
    })
    sns.barplot(
        x='模型',
        y='特征数量',
        data=feature_count_data,
        ax=axes[3, 0],
        palette=['#95a5a6', '#3498db']
    )
    axes[3, 0].set_xlabel('特征阶段', fontsize=12)
    axes[3, 0].set_ylabel('特征数量', fontsize=12)
    axes[3, 0].set_title('特征选择过程对比', fontsize=14, fontweight='bold')
    axes[3, 0].grid(axis='y', alpha=0.3)
    for i, count in enumerate(feature_count_data['特征数量']):
        axes[3, 0].text(i, count + 1, str(count), ha='center', fontsize=12, fontweight='bold')

    # ---------------------- 8. 逻辑回归核心特征（Top10） ----------------------
    if len(lr_final_features) > 0 and hasattr(train_result['lr_model'], 'coef_'):
        lr_coef = pd.DataFrame({
            'feature': lr_final_features[:10],
            'coef_abs': np.abs(train_result['lr_model'].coef_[0][:10])
        }).sort_values('coef_abs', ascending=True)
        sns.barplot(x='coef_abs', y='feature', data=lr_coef, ax=axes[3, 1], color='#3498db')
        axes[3, 1].set_xlabel('系数绝对值（重要性）', fontsize=12)
        axes[3, 1].set_ylabel('特征名称', fontsize=10)
        axes[3, 1].set_title('Logistic Regression Top10核心特征', fontsize=14, fontweight='bold')
        axes[3, 1].grid(axis='x', alpha=0.3)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存+显示
    save_path = os.path.join(VIS_DIR, f'metrics_{date_tag}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    st.pyplot(fig)

    # 计算一个典型高置信阈值（0.8）下的命中率，用于总结
    high_mask = lgb_proba >= 0.8
    high_sample_cnt = int(high_mask.sum())
    high_hit_rate = float(np.asarray(y_test)[high_mask].mean()) if high_sample_cnt > 0 else None

    # 核心结论
    st.markdown("### 📋 核心评估结论")
    st.markdown(
        f"- **LightGBM**：F1值 **{lgb_metrics['f1']}**，稳定性方差 **{stability_metrics['lgb_cv_var']}**（{'合格' if stability_metrics['lgb_cv_var'] < 0.1 else '需优化'}），使用全部特征（共{len(lgb_top300_features)}个）")
    st.markdown(
        f"- **逻辑回归**：F1值 **{lr_metrics['f1']}**，稳定性方差 **{stability_metrics['lr_cv_var']}**（{'合格' if stability_metrics['lr_cv_var'] < 0.1 else '需优化'}），使用去相关后的特征（{len(lr_final_features)}个）")
    st.markdown(f"- **最优模型**：**{('LightGBM' if lgb_metrics['f1'] > lr_metrics['f1'] else '逻辑回归')}**")
    st.markdown(f"- **推荐投注阈值**：**{best_threshold:.3f}**（对应精确率≥{target_precision}）")

    if high_hit_rate is not None:
        st.markdown(f"- **高置信度区间（阈值≥0.8）**：样本数 **{high_sample_cnt}**，真实命中率 **{high_hit_rate:.1%}**")
    else:
        st.markdown(f"- **高置信度区间（阈值≥0.8）**：当前验证集无样本，暂无法评估命中率")

    st.markdown(f"- **核心特征示例**：LightGBMTop1={lgb_top300_features[0] if len(lgb_top300_features) > 0 else '无'}，LRTop1={lr_final_features[0] if len(lr_final_features) > 0 else '无'}")

# ===================== 5. Streamlit主界面 =====================
def main():
    st.set_page_config(page_title="竞彩预测模型训练系统", layout="wide")
    st.title("⚽ 竞彩预测模型训练与推理系统")
    st.markdown("---")

    # 侧边栏配置
    st.sidebar.header("📋 功能选择")
    function_option = st.sidebar.radio("选择功能", ["全量训练", "当日推理"])

    # 初始化Session State
    if 'train_history' not in st.session_state:
        st.session_state['train_history'] = pd.DataFrame()

    # ---------------------- 功能1：全量训练 ----------------------
    if function_option == "全量训练":
        st.header("🚀 全量训练配置")
        env_label = "生产环境" if CURRENT_ENV == "prod" else "开发环境"
        st.caption(f"当前环境：{env_label}（FOOTBALL_ENV={CURRENT_ENV}）")

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("训练起始日期", value=pd.to_datetime('2025-10-11'))
        with col2:
            default_end_date = (datetime.now() - timedelta(days=1)).date()
            end_date = st.date_input("训练结束日期", value=default_end_date)

        st.markdown("---")

        if st.button("启动模型训练", type="primary", width='stretch'):
            if start_date > end_date:
                st.error("❌ 起始日期不能晚于结束日期")
                return

            with st.spinner("🔄 正在执行全量训练... 请耐心等待"):
                train_history = train_global_model(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )

            st.session_state['train_history'] = train_history

            if not train_history.empty:
                st.markdown("### 📈 训练历史汇总")
                st.dataframe(train_history.round(3), width='stretch')

                # 训练趋势图（即使目前每次只有一行，也保留，方便后续扩展）
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(train_history['window_end'], train_history['lr_f1'], marker='o', label='逻辑回归F1',
                        color='#3498db')
                ax.plot(train_history['window_end'], train_history['lgb_f1'], marker='s', label='LightGBM F1',
                        color='#e74c3c')
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='合格线（0.5）')
                ax.set_xlabel('训练数据结束日期')
                ax.set_ylabel('F1值')
                ax.set_title('全量训练F1值趋势')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)

    # ---------------------- 功能2：当日推理 ----------------------
    elif function_option == "当日推理":
        st.header("🎯 当日推理配置")
        env_label = "生产环境" if CURRENT_ENV == "prod" else "开发环境"
        st.caption(f"当前环境：{env_label}（FOOTBALL_ENV={CURRENT_ENV}）")

        target_date = st.date_input("选择推理日期", value=pd.to_datetime(datetime.now().strftime('%Y-%m-%d')))
        target_date_str = target_date.strftime('%Y-%m-%d')

        st.markdown("---")

        if st.button("执行当日推理", type="primary", width='stretch'):
            with st.spinner("🔍 正在执行推理..."):
                match_best_pred, two_combos = predict_today(target_date_str)

    # ---------------------- 侧边栏信息 ----------------------
    st.sidebar.markdown("---")
    st.sidebar.info(f"""
    📌 环境配置：{CURRENT_ENV}
    📁 数据库路径：{DB_PATH}
    📊 已训练模型数：{len(get_recent_model_dates(top_n=10))}
    📈 特征配置：预测者-方案对联合特征 + 辅助特征
    🎯 模型架构：LightGBM（全部特征）+ Logistic Regression（去相关后特征）
    🔧 推理策略：仅最优模型（单模型推理）
    """)

if __name__ == "__main__":
    main()