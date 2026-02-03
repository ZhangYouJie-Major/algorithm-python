"""
月末归档脚本

将本月完成的题目归档到 archive 目录

用法：
    python scripts/archive_month.py
"""

import os
import shutil
from datetime import datetime
from pathlib import Path


def get_current_month_archive():
    """获取当月归档目录"""
    now = datetime.now()
    year = now.year
    month = now.month

    # 月份名称映射
    month_names = {
        1: "01-january", 2: "02-february", 3: "03-march",
        4: "04-april", 5: "05-may", 6: "06-june",
        7: "07-july", 8: "08-august", 9: "09-september",
        10: "10-october", 11: "11-november", 12: "12-december"
    }

    archive_dir = f"archive/{year}/{month_names[month]}"
    return archive_dir


def get_recent_files(days=30):
    """获取最近修改的文件"""
    problems_dir = Path("src/problems")
    recent_files = []

    now = datetime.now().timestamp()

    for py_file in problems_dir.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue

        # 获取文件修改时间
        mtime = py_file.stat().st_mtime
        if now - mtime <= days * 24 * 3600:  # 30天内
            recent_files.append(py_file)

    return recent_files


def archive_month():
    """执行月度归档"""
    archive_dir = get_current_month_archive()
    os.makedirs(archive_dir, exist_ok=True)

    # 获取本月文件
    recent_files = get_recent_files(days=30)

    if not recent_files:
        print("本月没有新题目需要归档")
        return

    # 创建归档信息文件
    summary_path = os.path.join(archive_dir, "README.md")

    # 复制文件到归档目录
    archived_count = 0
    for file in recent_files:
        # 保持原有的子目录结构
        rel_path = file.relative_to("src/problems")
        dest_path = os.path.join(archive_dir, rel_path)

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(file, dest_path)
        archived_count += 1
        print(f"归档: {file.name}")

    # 写入归档总结
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# {datetime.now().strftime('%Y年%m月')} 刷题存档\n\n")
        f.write(f"## 统计\n\n")
        f.write(f"- 归档题目数: {archived_count}\n")
        f.write(f"- 归档时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 题目列表\n\n")

        for file in recent_files:
            rel_path = file.relative_to("src/problems")
            difficulty = rel_path.parts[0]  # easy/medium/hard
            f.write(f"- [{difficulty.upper()}] {file.name}\n")

    print(f"\n✅ 归档完成: {archive_dir}")
    print(f"共归档 {archived_count} 个题目")


if __name__ == "__main__":
    archive_month()
