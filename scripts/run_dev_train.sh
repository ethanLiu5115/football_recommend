#!/usr/bin/env bash

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

# 设置环境为 dev 并启动训练/推理界面
export FOOTBALL_ENV=dev
streamlit run src/model_train_app.py