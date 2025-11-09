import sqlite3
import os

def init_basic_data():
    """
    初始化基础数据：
    - 向team表插入10支常见球队（仅名称）
    - 向league表插入5个赛事（仅名称）
    """
    # 数据库路径
    db_path = os.path.join("data", "football.db")
    if not os.path.exists(db_path):
        print("错误：数据库文件不存在，请先运行db_create.py创建数据库")
        return

    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 插入球队数据（10支，仅名称）
    teams = [
        ("广岛三箭",),  # 注意：元组末尾加逗号，确保格式正确
        ("江原FC",),
        ("町田泽维",),
        ("墨尔本城",),
        ("首尔FC",),
        ("成都蓉城",),
        ("布里兰",),
        ("上海海港",),
        ("多哈萨德",),
        ("吉达国民",)
    ]
    # 插入语句匹配team表结构（仅team_name字段）
    cursor.executemany('''
    INSERT OR IGNORE INTO team (team_name)
    VALUES (?)
    ''', teams)  # OR IGNORE：避免重复插入（若表中已有相同名称）
    print(f"插入球队数据成功，共{len(teams)}支球队")

    # 2. 插入赛事数据（5个，仅名称）
    leagues = [
        ("亚冠精英",),
        ("欧冠",),
        ("英冠",),
        ("葡超",),
        ("西甲",)
    ]
    # 插入语句匹配league表结构（仅league_name字段）
    cursor.executemany('''
    INSERT OR IGNORE INTO league (league_name)
    VALUES (?)
    ''', leagues)
    print(f"插入赛事数据成功，共{len(leagues)}个赛事")

    # 提交并关闭连接
    conn.commit()
    conn.close()
    print("基础数据初始化完成！")

if __name__ == "__main__":
    init_basic_data()