import subprocess
import sys


def check_and_install_packages():
    """
    检查必要库是否安装，未安装则自动通过pip安装（适配Mac环境）
    输出安装日志（成功/失败的库）
    """
    # 需要检查的库（注意：sqlite3是Python标准库，无需pip安装）
    required_packages = [
        "pandas",
        "streamlit",
        "xgboost"
    ]
    # 记录安装结果
    install_log = {
        "success": [],
        "failed": []
    }

    print("===== 开始环境检查与配置 =====")

    for package in required_packages:
        try:
            # 尝试导入库，检查是否已安装
            __import__(package)
            install_log["success"].append(f"{package} 已安装")
            print(f"✅ {package} 已安装")
        except ImportError:
            # 未安装则通过pip安装（使用当前Python环境的pip）
            print(f"❌ {package} 未安装，开始安装...")
            try:
                # Mac环境下使用当前Python解释器的pip安装
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package],
                    stdout=subprocess.DEVNULL,  # 隐藏安装详情，仅捕获错误
                    stderr=subprocess.STDOUT
                )
                install_log["success"].append(f"{package} 安装成功")
                print(f"✅ {package} 安装成功")
            except subprocess.CalledProcessError:
                install_log["failed"].append(f"{package} 安装失败")
                print(f"❌ {package} 安装失败，请手动安装")

    # 单独处理sqlite3（标准库，无需安装）
    try:
        import sqlite3
        install_log["success"].append("sqlite3 已安装（标准库）")
        print("✅ sqlite3 已安装（Python标准库）")
    except ImportError:
        install_log["failed"].append("sqlite3 缺失（请检查Python环境）")
        print("❌ sqlite3 缺失（Python标准库异常，请重新安装Python）")

    # 输出安装总结
    print("\n===== 环境配置总结 =====")
    print("成功安装/已存在的库：")
    for item in install_log["success"]:
        print(f"- {item}")
    if install_log["failed"]:
        print("\n安装失败的库：")
        for item in install_log["failed"]:
            print(f"- {item}")
    else:
        print("\n所有必要库均已正确安装！")


if __name__ == "__main__":
    check_and_install_packages()