"""
新建题目脚本

用法：
    python scripts/new_problem.py <题号> <题目英文名> <难度>

示例：
    python scripts/new_problem.py 1 two_sum easy
"""

import sys
import os
from datetime import datetime


def create_problem_file(problem_num: str, problem_name: str, difficulty: str = "medium"):
    """创建新题目文件"""

    # 验证难度
    valid_difficulties = ["easy", "medium", "hard"]
    if difficulty.lower() not in valid_difficulties:
        difficulty = "medium"

    # 文件名
    filename = f"{problem_num}_{problem_name}.py"
    target_dir = f"src/problems/{difficulty.lower()}/"
    filepath = os.path.join(target_dir, filename)

    # 检查文件是否已存在
    if os.path.exists(filepath):
        print(f"⚠️  文件已存在: {filepath}")
        return False

    # 确保目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 读取模板
    template_path = "src/templates/problem_template.py"
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # 替换模板内容
    content = template.replace(
        "class Solution:",
        f"class Solution:\n    \"\"\"\n    LeetCode {problem_num}. {problem_name.replace('_', ' ').title()}\n    难度: {difficulty.lower()}\n    创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n    \"\"\"\n    pass\n\n    def solve"
    )

    # 写入文件
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ 题目文件已创建: {filepath}")
    return True


def main():
    if len(sys.argv) < 3:
        print("用法: python new_problem.py <题号> <题目英文名> [难度]")
        print("示例: python new_problem.py 1 two_sum easy")
        sys.exit(1)

    problem_num = sys.argv[1]
    problem_name = sys.argv[2]
    difficulty = sys.argv[3] if len(sys.argv) > 3 else "medium"

    create_problem_file(problem_num, problem_name, difficulty)


if __name__ == "__main__":
    main()
