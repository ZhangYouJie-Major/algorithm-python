"""
刷题统计脚本

统计当前刷题进度和月度历史

用法：
    python scripts/stats.py
"""

import os
from pathlib import Path
from datetime import datetime
from collections import Counter


def count_problems():
    """统计各难度题目数量"""
    problems_dir = Path("src/problems")
    stats = Counter()

    for py_file in problems_dir.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue

        # 获取难度分类
        parts = py_file.relative_to(problems_dir).parts
        if parts[0] in ["easy", "medium", "hard"]:
            stats[parts[0]] += 1

    return stats


def get_recent_solutions(limit=10):
    """获取最近完成的题目"""
    problems_dir = Path("src/problems")
    files = []

    for py_file in problems_dir.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue

        files.append((py_file, py_file.stat().st_mtime))

    # 按修改时间排序
    files.sort(key=lambda x: x[1], reverse=True)
    return files[:limit]


def count_archive_stats():
    """统计归档目录"""
    archive_dir = Path("archive")
    stats = {}

    if not archive_dir.exists():
        return stats

    for year_dir in archive_dir.iterdir():
        if year_dir.is_dir():
            year = year_dir.name
            stats[year] = len(list(year_dir.rglob("*.py")))

    return stats


def print_stats():
    """打印统计信息"""
    print("=" * 50)
    print("📊 刷题统计")
    print("=" * 50)

    # 当前题目统计
    stats = count_problems()
    total = sum(stats.values())

    print("\n📁 当前题目库:")
    print(f"  简单 (Easy):    {stats['easy']:3d} 题")
    print(f"  中等 (Medium):  {stats['medium']:3d} 题")
    print(f"  困难 (Hard):    {stats['hard']:3d} 题")
    print(f"  {'─' * 20}")
    print(f"  总计:           {total:3d} 题")

    # 归档统计
    archive_stats = count_archive_stats()
    if archive_stats:
        print("\n📦 历史归档:")
        for year, count in sorted(archive_stats.items()):
            print(f"  {year}: {count} 题")

    # 最近完成的题目
    print("\n🕐 最近完成:")
    recent = get_recent_solutions()
    for i, (file, mtime) in enumerate(recent, 1):
        time_str = datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M")
        print(f"  {i}. {file.name:30s} ({time_str})")

    print("\n" + "=" * 50)
    print(f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)


if __name__ == "__main__":
    print_stats()
