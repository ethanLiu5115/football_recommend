#!/usr/bin/env bash

# 切到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}" || exit 1

# 开发环境：FOOTBALL_ENV=dev
export FOOTBALL_ENV=dev

# 启动“信息录入”界面
streamlit run src/data_input_integrated.py