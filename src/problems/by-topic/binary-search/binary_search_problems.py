"""
二分查找 (Binary Search) 题目集合

包含所有使用二分查找解决的题目
"""

from typing import List
from bisect import bisect_left, bisect_right
import math


class Solution:
    """二分查找题目合集"""

    # ==================== 基础二分查找 ====================

    def search(self, nums: List[int], target: int) -> int:
        """
        704. 二分查找
        标准二分查找
        """
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return -1

    def searchInsert(self, nums: List[int], target: int) -> int:
        """
        35. 搜索插入位置
        """
        left, right = 0, len(nums)

        while left < right:
            mid = (left + right) // 2

            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid

        return left

    # ==================== 搜索旋转数组 ====================

    def searchRotated(self, nums: List[int], target: int) -> int:
        """
        33. 搜索旋转排序数组
        """
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                return mid

            # 判断哪一侧是有序的
            if nums[left] <= nums[mid]:
                # 左侧有序
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # 右侧有序
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1

    def findMin(self, nums: List[int]) -> int:
        """
        153. 寻找旋转排序数组中的最小值
        """
        left, right = 0, len(nums) - 1

        while left < right:
            mid = (left + right) // 2

            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid

        return nums[left]

    # ==================== 搜索范围 ====================

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        """
        34. 在排序数组中查找元素的第一个和最后一个位置
        """
        def findLeft():
            left, right = 0, len(nums)
            while left < right:
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid
            return left

        def findRight():
            left, right = 0, len(nums)
            while left < right:
                mid = (left + right) // 2
                if nums[mid] <= target:
                    left = mid + 1
                else:
                    right = mid
            return left

        left_idx = findLeft()
        right_idx = findRight()

        if left_idx < len(nums) and nums[left_idx] == target:
            return [left_idx, right_idx - 1]

        return [-1, -1]

    # ==================== 二分答案 ====================

    def mySqrt(self, x: int) -> int:
        """
        69. x 的平方根
        二分查找
        """
        if x < 2:
            return x

        left, right = 1, x // 2

        while left <= right:
            mid = (left + right) // 2
            sqrt = mid * mid

            if sqrt == x:
                return mid
            elif sqrt < x:
                left = mid + 1
            else:
                right = mid - 1

        return right

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """
        74. 搜索二维矩阵
        从右上角开始搜索（类似二分）
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

    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        """
        875. 爱吃香蕉的珂珂
        二分答案
        """
        def canEat(speed):
            hours = 0
            for pile in piles:
                hours += (pile + speed - 1) // speed
            return hours <= h

        left, right = 1, max(piles)

        while left < right:
            mid = (left + right) // 2
            if canEat(mid):
                right = mid
            else:
                left = mid + 1

        return left

    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        """
        1385. 两个数组间的距离值
        排序 + 二分查找
        """
        arr2.sort()
        ans = 0

        for x in arr1:
            i = bisect_left(arr2, x - d)
            if i == len(arr2) or arr2[i] > x + d:
                ans += 1

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

    def maximizeWin(self, prizePositions: List[int], k: int) -> int:
        """
        2555. 两个线段获得的最多奖品
        """
        from bisect import bisect_left
        n = len(prizePositions)
        dp = [0] * (n + 1)
        ans = 0

        for i in range(n):
            x = bisect_left(prizePositions, prizePositions[i] - k)
            ans = max(ans, dp[x] + i - x + 1)
            dp[i + 1] = max(dp[i], i - x + 1)

        return ans

    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        """
        2300. 咒语和药水的成功对数
        排序 + 二分
        """
        potions.sort()
        n = len(potions)
        result = []

        for spell in spells:
            # 找到第一个满足 spell * potion >= success 的位置
            min_potion = (success + spell - 1) // spell  # 向上取整
            idx = bisect_left(potions, min_potion)
            result.append(n - idx)

        return result

    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        """
        2187. 完成旅途的最少时间
        二分答案
        """
        def canComplete(t):
            trips = 0
            for tm in time:
                trips += t // tm
            return trips >= totalTrips

        left, right = 1, min(time) * totalTrips

        while left < right:
            mid = (left + right) // 2
            if canComplete(mid):
                right = mid
            else:
                left = mid + 1

        return left

    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        """
        475. 供暖器
        二分查找
        """
        heaters.sort()
        radius = 0

        for house in houses:
            # 找到最近的热水器
            idx = bisect_left(heaters, house)

            if idx == 0:
                dist = abs(heaters[0] - house)
            elif idx == len(heaters):
                dist = abs(heaters[-1] - house)
            else:
                dist = min(abs(heaters[idx] - house), abs(heaters[idx - 1] - house))

            radius = max(radius, dist)

        return radius

    def splitArray(self, nums: List[int], k: int) -> int:
        """
        410. 分割数组的最大值
        二分答案
        """
        def canSplit(max_sum):
            count = 1
            current_sum = 0

            for num in nums:
                current_sum += num
                if current_sum > max_sum:
                    count += 1
                    current_sum = num
                    if count > k:
                        return False

            return True

        left, right = max(nums), sum(nums)

        while left < right:
            mid = (left + right) // 2
            if canSplit(mid):
                right = mid
            else:
                left = mid + 1

        return left

    def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
        """
        1292. 元素和小于等于阈值的正方形的最大边长
        二分答案 + 前缀和
        """
        m, n = len(mat), len(mat[0])
        s = [[0] * (n + 1) for _ in range(m + 1)]

        # 计算前缀和
        for i in range(m):
            for j in range(n):
                s[i + 1][j + 1] = s[i][j + 1] + s[i + 1][j] - s[i][j] + mat[i][j]

        def query(r1, c1, r2, c2) -> int:
            return s[r2 + 1][c2 + 1] - s[r2 + 1][c1] - s[r1][c2 + 1] + s[r1][c1]

        ans = 0

        for i in range(m):
            for j in range(n):
                while ans + i < m and ans + j < n and query(i, j, ans + i, ans + j) <= threshold:
                    ans += 1

        return ans


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 测试二分查找
    print("=== 二分查找 ===")
    print(solution.search([-1,0,3,5,9,12], 9))

    # 测试搜索插入位置
    print("\n=== 搜索插入位置 ===")
    print(solution.searchInsert([1,3,5,6], 5))

    # 测试平方根
    print("\n=== x的平方根 ===")
    print(solution.mySqrt(8))
