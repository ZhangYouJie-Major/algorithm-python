# 算法刷题存档

> 个人算法题练习存档库

## 📁 目录结构

```
algorithm-python/
├── src/
│   ├── problems/           # 日常刷题目录
│   │   ├── easy/           # 简单题
│   │   ├── medium/         # 中等题
│   │   ├── hard/           # 困难题
│   │   └── by-topic/       # 按算法类型分类
│   └── templates/          # 代码模板
│
├── archive/                # 月度历史存档
│   ├── 2024/
│   └── 2025/
│
├── docs/                   # 解题文档
└── scripts/                # 辅助脚本
```

## 🚀 快速开始

### 1. 创建新题目

```bash
python scripts/new_problem.py <题号> <题目英文名> [难度]

# 示例
python scripts/new_problem.py 1 two_sum easy
```

### 2. 查看统计

```bash
python scripts/stats.py
```

### 3. 月末归档

```bash
python scripts/archive_month.py
```

## 📊 当前进度

| 难度 | 数量 |
|------|------|
| Easy | - |
| Medium | - |
| Hard | - |

## 🏷️ 算法分类

- [数组 (Array)](src/problems/by-topic/array/)
- [动态规划 (DP)](src/problems/by-topic/dp/)
- [图论 (Graph)](src/problems/by-topic/graph/)
- [树 (Tree)](src/problems/by-topic/tree/)
- [滑动窗口](src/problems/by-topic/sliding-window/)
- [双指针](src/problems/by-topic/two-pointers/)
- [贪心](src/problems/by-topic/greedy/)
- [回溯](src/problems/by-topic/backtracking/)
- [堆](src/problems/by-topic/heap/)
- [数学](src/problems/by-topic/math/)
- [字符串](src/problems/by-topic/string/)
- [二分查找](src/problems/by-topic/binary-search/)

## 📝 存档规则

1. **命名规范**: `{题号}_{题目英文名}.py`
2. **每月归档**: 月末执行归档脚本，将当月完成的题目移入 `archive/` 目录
3. **分类存放**: 新题按难度放在 `easy/medium/hard/` 目录
4. **保留源码**: `src/problems/` 始终保留最新代码副本

---

**开始时间**: 2025-02-03
**更新频率**: 每日更新
