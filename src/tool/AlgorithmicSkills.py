import itertools
import math, heapq, re
from bisect import bisect_right, bisect_left
from builtins import str
from collections import deque, Counter, defaultdict
from functools import cache
from itertools import pairwise, accumulate, combinations, permutations
from queue import PriorityQueue
from typing import List, Dict, Optional, Set
from src.tool.Construct import ListNode, TreeNode

# 双指针

# 二分
"""
    1、二分查找函数  from bisect import bisect_right, bisect_left

"""

# 前缀和
"""
    1、一维前缀和
    对于数组a 定义他的前缀和
    s[0] = 0
    s[1] = a[0]
    s[2] = a[0] + a[1]
    ......
    s[i] = a[0] + a[1] + ... + a[i-1]
    于是我们可以推导出
    a[left] + a[1] + ... + a[right] -> (a[0] + a[1] + ... + a[right]) - (a[0] + a[1] + ... + a[left-1])
    -> s[right+1] -  s[left]
    s = [0] * (n + 1)
    for i in range(n):
        s[i+1] = s[i] + nums[i]
    nums[i] + nums[i + 1] ... + nums[j] = s[j+1] -s[i]        

    2、二维前缀和
    
    s = [[0] * (n + 1) for _ in range(m + 1)]
    s[i + 1][j + 1] = s[i][j + 1] + s[i + 1][j] - s[i][j] + nums[i][j]
    
    for i in range(m):
        for j in range(n):
            s[i + 1][j + 1] = s[i][j + 1] + s[i + 1][j] - s[i][j] + matrix[i][j]
            
    从row1,col1 到 row2,col2 的sum  s[row2+1][col2+1] - s[row2+1][col1] - s[row1][col2+1] + s[row1][col1]
    

    
"""

# 差分
"""
    1、一维差分
    
    2、二维差分
"""

# 滑动窗口
"""
    1、定长的滑动窗口
    
    2、不定长的滑动窗口
    
    3、滑动窗口解决子数组问题
        3.1 

"""


# 记忆化搜索到动态规划
# class MemoizedSearch:


class DynamicProgramming:
    """
        capacity 背包容量
        w[i]: 第i个物品的体积
        v[i]: 第i个物品的价值
    """

    def zero_one_knapsack(self, capacity: int, w: List[int], v: List[int]) -> int:
        n = len(w)

        @cache
        def dfs(i, c):
            if i < 0:
                return 0
            if c < w[i]:
                return dfs(i - 1, c)
            return max(dfs(i - 1, c), dfs(i - 1, c - w[i]) + v[i])

        return dfs(n - 1, capacity)

    def unbounded_knapsack(self, capacity: int, w: List[int], v: List[int]) -> int:
        n = len(w)

        @cache
        def dfs(i, c):
            if i < 0:
                return 0
            if c < w[i]:
                return dfs(i - 1, c)
            return max(dfs(i - 1, c), dfs(i, c - w[i]) + v[i])

        return dfs(n - 1, capacity)


class SlidingWindowsSkill:
    nums = []
    """
        滑动窗口解决子数组问题
        1、答案在整个数组中间某段区间内才能是合法的 一般是求至多、小于这种情况枚举右端点的同时，去看对应的合法的左端点个数，
        也就是 right不断加 1 枚举时，每次保证 [left:right] 区间都是合法的，那么这段区间的子数组个数答案就是 right−left+1个
        2、答案是整个数组时都是合法的
        一般是 求至少、大于 ，这种情况下只需在枚举右端点的同时，保证 left左侧是合法的就可以，
        这时候区间 [[0,1,2,...,left−1]:right] 其实都是合法的，那么以该 right 为右端点的子数组个数答案为 left个、
        3、答案在整个数组中间某段区间内才能是合法的。，但与第一种情况不同的是一般都是 求恰好 ，这种情况下需要把「求恰好」 转换成为 「求至多、小于」。
         具体思路为：例如求恰好 k 个，就用最多 k 个 - 最多 k−1 个 = 恰好 k 个  => cal(k) - cal(k-1)
    """

    def check(self, i: int) -> int:
        nums = self.nums
        ans = left = 0
        for right, x in enumerate(self.nums):
            y = nums[left]
            left += 1
            while left <= right and '条件不合法':
                # xxx 根据题意改编条件
                left += 1
            ans += right - left + 1
        return ans

    def check1(self, i: int) -> int:
        nums = self.nums
        ans = left = 0
        for right, x in enumerate(self.nums):
            y = nums[left]
            left += 1
            while left <= right and '条件合法':
                # xxx 根据题意改编条件
                left += 1
            ans += left
        return ans

    def check3(self, i: int) -> int:
        return self.check1(i) - self.check1(i - 1)


class BinarySearchSkill:
    def lower_bound(self, nums: List[int], target: int) -> int:
        """返回第一个 >= target 的下标，若不存在返回 len(a)"""
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    """
        | 需求          | 一行代码                          |
        | ----------- | -----------------------------     |
        | ≥x 的第一个下标   | `bisect_left(a, x)`           |
        | >x 的第一个下标   | `bisect_right(a, x)`          |
        | <x 的最后一个下标 | `bisect_left(a, x) - 1`       |
        | ≤x 的最后一个下标  | `bisect_right(a, x) - 1`      |
        | `<x` 的元素个数  | `bisect_left(a, x)`           |
        | `≤x` 的元素个数  | `bisect_right(a, x)`          |
        | `≥x` 的元素个数  | `len(a) - bisect_left(a, x)`  |
        | `>x` 的元素个数  | `len(a) - bisect_right(a, x)` |
    """


class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        s = [[0] * (n + 1) for _ in range(m + 1)]
        """
            初始化 m+1 n+1
        """
        for i in range(m):
            for j in range(n):
                s[i + 1][j + 1] = s[i][j + 1] + s[i + 1][j] - s[i][j] + matrix[i][j]
        self.s = s

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.s[row2 + 1][col2 + 1] - self.s[row2 + 1][col1] - self.s[row1][col2 + 1] + self.s[row1][col1]
