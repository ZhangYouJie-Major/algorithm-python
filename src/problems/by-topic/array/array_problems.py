"""
数组 (Array) 题目集合

包含所有数组相关的题目
"""

from typing import List
from collections import Counter, defaultdict
from itertools import pairwise, accumulate
from bisect import bisect_left, bisect_right
import math


class Solution:
    """数组题目合集"""

    # ==================== 前缀和 ====================

    def subarraySum(self, nums: List[int], k: int) -> int:
        """
        560. 和为 K 的子数组
        前缀和 + 哈希表
        """
        pre_sum = 0
        ans = 0
        ctn = defaultdict(int)
        for x in nums:
            ctn[pre_sum] += 1
            pre_sum += x
            ans += ctn[k - pre_sum]
        return ans

    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        """
        930. 和相同的二元子数组
        前缀和
        """
        pre_sum = 0
        ctn = defaultdict(int)
        ans = 0
        for x in nums:
            ctn[pre_sum] += 1
            pre_sum += x
            ans += ctn[pre_sum - goal]
        return ans

    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        """
        974. 和可被 K 整除的子数组
        同余定理
        """
        ans = pre_sum = 0
        ctn = defaultdict(int)
        for x in nums:
            ctn[pre_sum % k] += 1
            pre_sum += x
            ans += ctn[pre_sum % k]
        return ans

    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        """
        523. 连续的子数组和
        """
        pre_sum = 0
        ctn = defaultdict(int)
        for i, x in enumerate(nums):
            if pre_sum % k not in ctn:
                ctn[pre_sum % k] = i
            pre_sum += x
            if pre_sum % k in ctn and i - ctn[pre_sum % k] + 1 >= 2:
                return True
        return False

    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        """
        2429. 数组的前缀异或查询
        前缀异或
        """
        n = len(arr)
        xors = [0] * (n + 1)
        for i, x in enumerate(arr):
            xors[i + 1] = xors[i] ^ x
        ans = []
        for x, y in queries:
            ans.append(xors[y + 1] ^ xors[x])
        return ans

    def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
        """
        2848. 与车相交的点
        前缀和 + 二分
        """
        s = [0] * (len(words) + 1)
        for i, val in enumerate(words):
            s[i + 1] = s[i] + (val[0] in 'aeiou' and val[-1] in 'aeiou')
        ans = []
        for q in queries:
            ans.append(s[q[1] + 1] - s[q[0]])
        return ans

    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        """
        2389. 和有限的最长子序列
        排序 + 前缀和 + 二分
        """
        nums.sort()
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        for i, val in enumerate(queries):
            queries[i] = bisect_right(nums, val)
        return queries

    # ==================== 差分数组 ====================

    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        """
        1094. 拼车
        差分数组
        """
        d = [0] * 1001
        for num, _from, _to in trips:
            d[_from] += num
            d[_to] -= num
        return all(s <= capacity for s in accumulate(d))

    # ==================== 模拟 ====================

    def rotate(self, nums: List[int], k: int) -> None:
        """
        189. 轮转数组
        三次翻转
        """
        n = len(nums)

        def reverser(i: int, j: int) -> None:
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1

        k %= n
        reverser(0, n - 1)
        reverser(0, k - 1)
        reverser(k, n - 1)

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """
        54. 螺旋矩阵
        模拟遍历
        """
        if not matrix:
            return []

        m, n = len(matrix), len(matrix[0])
        top, bottom, left, right = 0, m - 1, 0, n - 1
        result = []

        while top <= bottom and left <= right:
            # 从左到右
            for j in range(left, right + 1):
                result.append(matrix[top][j])
            top += 1

            # 从上到下
            for i in range(top, bottom + 1):
                result.append(matrix[i][right])
            right -= 1

            if top <= bottom:
                # 从右到左
                for j in range(right, left - 1, -1):
                    result.append(matrix[bottom][j])
                bottom -= 1

            if left <= right:
                # 从下到上
                for i in range(bottom, top - 1, -1):
                    result.append(matrix[i][left])
                left += 1

        return result

    def generateMatrix(self, n: int) -> List[List[int]]:
        """
        59. 螺旋矩阵 II
        """
        matrix = [[0] * n for _ in range(n)]
        top, bottom, left, right = 0, n - 1, 0, n - 1
        num = 1

        while top <= bottom and left <= right:
            for j in range(left, right + 1):
                matrix[top][j] = num
                num += 1
            top += 1

            for i in range(top, bottom + 1):
                matrix[i][right] = num
                num += 1
            right -= 1

            if top <= bottom:
                for j in range(right, left - 1, -1):
                    matrix[bottom][j] = num
                    num += 1
                bottom -= 1

            if left <= right:
                for i in range(bottom, top - 1, -1):
                    matrix[i][left] = num
                    num += 1
                left += 1

        return matrix

    # ==================== 计数问题 ====================

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        49. 字母异位词分组
        排序作为key
        """
        dic = defaultdict(list)
        for s in strs:
            dic[''.join(sorted(s))].append(s)
        return list(dic.values())

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        347. 前 K 个高频元素
        哈希表 + 堆
        """
        import heapq
        count = Counter(nums)
        return [item for item, _ in count.most_common(k)]

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        """
        238. 除自身以外数组的乘积
        左右乘积
        """
        n = len(nums)
        answer = [1] * n

        # 左侧乘积
        left_product = 1
        for i in range(n):
            answer[i] = left_product
            left_product *= nums[i]

        # 右侧乘积
        right_product = 1
        for i in range(n - 1, -1, -1):
            answer[i] *= right_product
            right_product *= nums[i]

        return answer

    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        215. 数组中的第K个最大元素
        堆排序
        """
        from queue import PriorityQueue
        q = PriorityQueue()
        for item in nums:
            q.put((-item, item))
        for index in range(k):
            q.get()
            if index == k - 1:
                return q.get()[1]
        return -1

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        239. 滑动窗口最大值
        单调队列
        """
        from collections import deque
        ans = []
        q = deque()
        for index, value in enumerate(nums):
            while q and nums[q[-1]] <= value:
                q.pop()
            q.append(index)

            if index - q[0] >= k:
                q.popleft()

            if index >= k - 1:
                ans.append(nums[q[0]])
        return ans

    # ==================== 子数组问题 ====================

    def maxSubArray(self, nums: List[int]) -> int:
        """
        53. 最大子数组和
        动态规划
        """
        import math
        ans = -math.inf
        pre_sum = pre_min_sum = 0
        for x in nums:
            pre_sum += x
            pre_min_sum = min(pre_min_sum, pre_sum)
            ans = max(ans, pre_sum - pre_min_sum)
        return ans

    def findMaxLength(self, nums: List[int]) -> int:
        """
        525. 连续数组
        前缀和 + 哈希表
        """
        prefix_sum = 0
        first_occurrence = {0: -1}
        max_length = 0

        for i, num in enumerate(nums):
            # 将0视为-1
            prefix_sum += 1 if num == 1 else -1

            # 如果此前出现过相同的前缀和，则中间的0和1数量相等
            if prefix_sum in first_occurrence:
                max_length = max(max_length, i - first_occurrence[prefix_sum])
            else:
                first_occurrence[prefix_sum] = i

        return max_length

    # ==================== 矩阵操作 ====================

    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        73. 矩阵置零
        原地修改
        """
        m, n = len(matrix), len(matrix[0])
        first_row_has_zero = any(matrix[0][j] == 0 for j in range(n))
        first_col_has_zero = any(matrix[i][0] == 0 for i in range(m))

        # 使用第一行和第一列标记
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0

        # 根据标记置零
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0

        # 处理第一行和第一列
        if first_row_has_zero:
            for j in range(n):
                matrix[0][j] = 0

        if first_col_has_zero:
            for i in range(m):
                matrix[i][0] = 0

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        74. 搜索二维矩阵
        从右上角或左下角开始搜索
        """
        if not matrix or not matrix[0]:
            return False

        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1

        while row < m and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1

        return False


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 测试和为K的子数组
    print("=== 和为K的子数组 ===")
    print(solution.subarraySum([1,1,1], 2))

    # 测试最大子数组和
    print("\n=== 最大子数组和 ===")
    print(solution.maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))

    # 测试轮转数组
    print("\n=== 轮转数组 ===")
    nums = [1,2,3,4,5,6,7]
    solution.rotate(nums, 3)
    print(nums)
