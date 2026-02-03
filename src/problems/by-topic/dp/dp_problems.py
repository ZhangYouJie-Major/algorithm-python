"""
动态规划 (Dynamic Programming) 题目集合

包含所有使用动态规划解决的题目
"""

from typing import List
from functools import cache
import math


class Solution:
    """动态规划题目合集"""

    # ==================== 股票问题 ====================

    def maxProfit(self, prices: List[int]) -> int:
        """
        121. 买卖股票的最佳时机
        只能买卖一次
        """
        ans = 0
        min_price = prices[0]
        for p in prices:
            ans = max(ans, p - min_price)
            min_price = min(min_price, p)
        return ans

    def maxProfit2(self, prices: List[int]) -> int:
        """
        122. 买卖股票的最佳时机 II
        可以买卖多次
        """
        n = len(prices)
        f0 = 0
        pre0 = 0
        f1 = -prices[0]
        for i in range(1, n):
            pre0, f0, f1 = f0, max(f0, f1 + prices[i]), max(f1, f0 - prices[i])
        return f0

    def maxProfit3(self, prices: List[int]) -> int:
        """
        123. 买卖股票的最佳时机 III
        最多买卖两次
        """
        n = len(prices)
        buy_1 = buy_2 = -prices[0]
        sell_1 = sell_2 = 0
        for i in range(1, n):
            buy_1 = max(buy_1, -prices[i])
            sell_1 = max(sell_1, buy_1 + prices[i])
            buy_2 = max(buy_2, sell_1 - prices[i])
            sell_2 = max(sell_2, buy_2 + prices[i])
        return sell_2

    def maxProfit4(self, k: int, prices: List[int]) -> int:
        """
        188. 买卖股票的最佳时机 IV
        最多买卖k次
        """
        f = [[-math.inf] * 2 for _ in range(k + 2)]
        for j in range(1, k + 2):
            f[j][0] = 0
        for i, p in enumerate(prices):
            for j in range(1, k + 2):
                f[j][0] = max(f[j][0], f[j - 1][1] + p)
                f[j][1] = max(f[j][1], f[j][0] - p)
        return f[k + 1][0]

    # ==================== 打家劫舍系列 ====================

    def rob(self, nums: List[int]) -> int:
        """
        198. 打家劫舍
        不能打劫相邻的房屋
        """
        n = len(nums)
        if n <= 2:
            return max(nums[0], nums[-1])
        f = [0] * n
        f[0], f[1] = nums[0], max(nums[1], nums[0])
        for i in range(2, n):
            f[i] = max(f[i - 1], f[i - 2] + nums[i])
        return f[-1]

    # ==================== 子序列问题 ====================

    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        """
        1143. 最长公共子序列
        """
        m, n = len(text1), len(text2)
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i, x in enumerate(text1):
            for j, y in enumerate(text2):
                if x == y:
                    f[i + 1][j + 1] = f[i][j] + 1
                else:
                    f[i + 1][j + 1] = max(f[i][j + 1], f[i + 1][j])
        return f[m][n]

    def lengthOfLIS(self, nums: List[int]) -> int:
        """
        300. 最长递增子序列
        """
        from bisect import bisect_left
        g = []
        for x in nums:
            index = bisect_left(g, x)
            if index == len(g):
                g.append(x)
            else:
                g[index] = x
        return len(g)

    def longestPalindromeSubseq(self, s: str) -> int:
        """
        516. 最长回文子序列
        """
        n = len(s)
        f = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            f[i][i] = 1
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    f[i][j] = f[i + 1][j - 1] + 2
                else:
                    f[i][j] = max(f[i][j - 1], f[i + 1][j])
        return f[0][n - 1]

    # ==================== 编辑距离 ====================

    def minDistance(self, word1: str, word2: str) -> int:
        """
        72. 编辑距离
        """
        m, n = len(word1), len(word2)
        f = [[0] * (n + 1) for _ in range(m + 1)]
        f[0] = list(range(n + 1))

        for i, x in enumerate(word1):
            f[i + 1][0] = i + 1
            for j, y in enumerate(word2):
                if x == y:
                    f[i + 1][j + 1] = f[i][j]
                else:
                    f[i + 1][j + 1] = min(f[i][j + 1], f[i + 1][j], f[i][j]) + 1
        return f[m][n]

    def numDistinct(self, s: str, t: str) -> int:
        """
        115. 不同的子序列
        """
        m, n = len(s), len(t)
        if m < n:
            return 0
        f = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            f[i][n] = 1
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if s[i] == t[j]:
                    f[i][j] = f[i + 1][j + 1] + f[i + 1][j]
                else:
                    f[i][j] = f[i + 1][j]
        return f[0][0]

    # ==================== 背包问题 ====================

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """
        494. 目标和
        转化为背包问题
        """
        target += sum(nums)
        if target < 0 or target % 2:
            return 0
        target //= 2
        f = [0] * (target + 1)
        f[0] = 1
        for x in nums:
            for c in range(target, x - 1, -1):
                f[c] = f[c] + f[c - x]
        return f[target]

    def coinChange(self, coins: List[int], amount: int) -> int:
        """
        322. 零钱兑换
        """
        f = [math.inf] * (amount + 1)
        f[0] = 0

        for i, x in enumerate(coins):
            for c in range(x, amount + 1):
                f[c] = min(f[c], f[c - x] + 1)

        ans = f[amount]
        return ans if ans < math.inf else -1

    # ==================== 区间DP ====================

    def minScoreTriangulation(self, values: List[int]) -> int:
        """
        1039. 多边形三角剖分的最低得分
        """
        n = len(values)
        f = [[0] * n for _ in range(n)]
        for i in range(n - 3, -1, -1):
            for j in range(i + 2, n):
                res = math.inf
                for k in range(i + 1, j):
                    res = min(res, f[i][k] + f[k][j] + values[i] * values[j] * values[k])
                f[i][j] = res
        return f[0][n - 1]

    # ==================== 状态机DP ====================

    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        """
        3263. 能量饮料的最大提升
        """
        n = len(energyDrinkA)
        c = (energyDrinkA, energyDrinkB)

        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return 0
            return max(dfs(i - 1, j), dfs(i - 2, j ^ 1)) + c[j][i]

        return max(dfs(n - 1, 0), dfs(n - 1, 1))

    # ==================== 数位DP ====================

    def countSpecialNumbers(self, n: int) -> int:
        """
        2376. 统计特殊整数
        不含重复数字
        """
        s = str(n)

        @cache
        def dfs(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
            if i == len(s):
                return 1 if is_num else 0
            res = 0
            if not is_num:
                res += dfs(i + 1, mask, False, False)
            up = int(s[i]) if is_limit else 9
            for d in range(1 - int(is_num), up + 1):
                if mask >> d & 1 == 0:
                    res += dfs(i + 1, mask | (1 << d), is_limit and d == up, True)
            return res

        return dfs(0, 0, True, False)

    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
        """
        902. 最大为 N 的数字组合
        """
        s = str(n)

        @cache
        def dfs(i: int, is_limit: bool, is_num: bool) -> int:
            if i == len(s):
                return int(is_num)
            res = 0
            if not is_num:
                res += dfs(i + 1, False, False)
            up = s[i] if is_limit else '9'
            for d in digits:
                if d > up:
                    break
                res += dfs(i + 1, is_limit and d == up, True)
            return res

        return dfs(0, True, False)

    def countDigitOne(self, n: int) -> int:
        """
        233. 数字 1 的个数
        """
        s = str(n)

        @cache
        def dfs(i: int, ctn: int, is_limit: bool) -> int:
            if i == len(s):
                return ctn
            res = 0
            up = int(s[i]) if is_limit else 9
            for d in range(up + 1):
                res += dfs(i + 1, ctn + (d == 1), is_limit and d == up)
            return res

        return dfs(0, 0, True)

    # ==================== 记忆化搜索 ====================

    def maximumLength(self, nums: List[int], k: int) -> int:
        """
        3167. 字符串的最多k个不同相邻差值
        """
        n = len(nums)

        @cache
        def dfs(i: int, j: int) -> int:
            if i == 0:
                return 0
            mx = 0
            for p in range(i):
                if nums[p] == nums[i]:
                    mx = max(mx, dfs(p, j) + 1)
                elif p and nums[p] != nums[i]:
                    mx = max(mx, dfs(p, j - 1) + 1)
            return mx

        return max(dfs(i, k) for i in range(n - 1, -1, -1))

    def climbStairs(self, n: int, costs: List[int]) -> int:
        """
        爬楼梯的最小代价
        """
        @cache
        def dfs(i):
            if i == 0:
                return 0
            return min(dfs(j) + (i - j) * (i - j) for j in range(max(i - 3, 0), i)) + costs[i-1]

        return dfs(n)

    def superEggDrop(self, k: int, n: int) -> int:
        """
        887. 鸡蛋掉落
        """
        import itertools
        f = [0] * (k + 1)
        for i in itertools.count(1):
            for j in range(k, 0, -1):
                f[j] = f[j] + f[j - 1] + 1
                if f[k] >= n:
                    return i

    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        """
        3149. 找出分数最低的排列
        """
        MOD = 1_000_000_007
        req = [-1] * n
        for end, ctn in requirements:
            req[end] = ctn
        if req[0]:
            return 0

        @cache
        def dfs(i: int, j: int) -> int:
            if i == 0:
                return 1
            r = req[i - 1]
            if r >= 0:
                return dfs(i - 1, r) if r <= j <= i + r else 0
            return sum(dfs(i - 1, j - k) for k in range(min(i, j) + 1)) % MOD

        return dfs(n - 1, req[-1])

    # ==================== 最大点数 ====================

    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        """
        1458. 两个子序列的最大点积
        """
        m, n = len(nums1), len(nums2)
        f = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                xij = nums1[i] * nums2[j]
                f[i][j] = xij
                if i > 0:
                    f[i][j] = max(f[i][j], f[i - 1][j])
                if j > 0:
                    f[i][j] = max(f[i][j], f[i][j - 1])
                if i > 0 and j > 0:
                    f[i][j] = max(f[i][j], f[i - 1][j - 1] + xij)
        return f[-1][-1]

    def takeCharacters(self, s: str, k: int) -> int:
        """
        2516. 每种字符至少取k个
        滑动窗口 + DP
        """
        from collections import Counter
        ctn = Counter(s)
        if any(ctn[x] < k for x in 'abc'):
            return -1
        ans = left = 0

        for right, ch in enumerate(s):
            ctn[ch] -= 1
            while ctn[ch] < k:
                ctn[s[left]] += 1
                left += 1
            ans = max(ans, right - left + 1)
        return len(s) - ans


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 测试股票问题
    print("=== 买卖股票 ===")
    print(solution.maxProfit([7,1,5,3,6,4]))  # 5
    print(solution.maxProfit2([7,1,5,3,6,4]))  # 7

    # 测试打家劫舍
    print("\n=== 打家劫舍 ===")
    print(solution.rob([2,7,9,3,1]))  # 12

    # 测试最长递增子序列
    print("\n=== 最长递增子序列 ===")
    print(solution.lengthOfLIS([10,9,2,5,3,7,101,18]))  # 4
