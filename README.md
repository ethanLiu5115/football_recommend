# ⚽ 竞彩预测模型训练与推理系统

一个基于 **LightGBM + Logistic Regression** 的足球竞彩预测系统，用来从“人（预测者）× 方案”的数据中自动学习命中规律，支持：

- 全量历史数据训练（带时间窗口切片评估）
- 最优模型管理（自动选择当前最佳模型）
- 当日比赛推理与推荐 Top10 高置信方案
- 自动记录模型预测命中率 & Top2 全局高置信预测
- 基于 Streamlit 的可视化界面

---

## 🧱 项目结构

```text
FootballRecommend/
├── config/
│   ├── dev_config.py               # 开发环境配置（dev DB / 模型 / 日志路径）
│   └── prod_config.py              # 生产环境配置（prod DB / 模型 / 日志路径）
├── data/
│   ├── dev/
│   │   └── football_dev.db         # 开发环境 SQLite 数据库
│   └── prod/
│       └── football_prod.db        # 生产环境 SQLite 数据库
├── metrics/                        # 训练过程生成的评估指标、窗口日志等
├── trained_models/                 # 训练好的模型 & 特征列表 & best_model_config.json
├── scripts/                        # 辅助脚本（数据导入、清洗等）
├── src/
│   ├── model_train_app.py          # 主入口（Streamlit App：训练 + 推理）
│   ├── utils.py                    # 公共工具函数（特征工程、DB 工具、统计表初始化等）
│   └── data_input_integrated.py    # 信息录入入口（Streamlit App：竞彩比赛、预测、赛果录入）
├── requirements.txt
└── README.md