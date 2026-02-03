"""
贪心 (Greedy) 题目集合

包含所有使用贪心算法解决的题目
"""

from typing import List
from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
import math


class Solution:
    """贪心题目合集"""

    def maxProfit(self, prices: List[int]) -> int:
        """
        121. 买卖股票的最佳时机
        贪心：记录最低价
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
        贪心：每涨就卖
        """
        ans = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                ans += prices[i] - prices[i - 1]
        return ans

    def jump(self, nums: List[int]) -> int:
        """
        45. 跳跃游戏 II
        贪心：每次跳到最远位置
        """
        n = len(nums)
        if n <= 1:
            return 0

        jumps = 0
        current_end = 0
        farthest = 0

        for i in range(n - 1):
            farthest = max(farthest, i + nums[i])

            if i == current_end:
                jumps += 1
                current_end = farthest

                if current_end >= n - 1:
                    break

        return jumps

    def canJump(self, nums: List[int]) -> bool:
        """
        55. 跳跃游戏
        贪心：维护最远可达位置
        """
        farthest = 0
        for i, num in enumerate(nums):
            if i > farthest:
                return False
            farthest = max(farthest, i + num)
        return True

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        56. 合并区间
        贪心：排序后合并
        """
        intervals.sort(key=lambda x: x[0])
        merged = []

        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged

    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        """
        435. 无重叠区间
        贪心：按结束时间排序
        """
        if not intervals:
            return 0

        intervals.sort(key=lambda x: x[1])
        count = 1
        end = intervals[0][1]

        for interval in intervals[1:]:
            if interval[0] >= end:
                count += 1
                end = interval[1]

        return len(intervals) - count

    def findMinArrowShots(self, points: List[List[int]]) -> int:
        """
        452. 用最少数量的箭引爆气球
        贪心：按结束位置排序
        """
        if not points:
            return 0

        points.sort(key=lambda x: x[1])
        arrows = 1
        end = points[0][1]

        for point in points[1:]:
            if point[0] > end:
                arrows += 1
                end = point[1]

        return arrows

    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        """
        406. 根据身高重建队列
        贪心：身高降序，k升序
        """
        people.sort(key=lambda x: (-x[0], x[1]))
        queue = []

        for p in people:
            queue.insert(p[1], p)

        return queue

    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        """
        605. 种花问题
        贪心：能种就种
        """
        count = 0
        length = len(flowerbed)

        for i in range(length):
            if flowerbed[i] == 0:
                empty_left = (i == 0) or (flowerbed[i - 1] == 0)
                empty_right = (i == length - 1) or (flowerbed[i + 1] == 0)

                if empty_left and empty_right:
                    flowerbed[i] = 1
                    count += 1

                    if count >= n:
                        return True

        return count >= n

    def isSubsequence(self, s: str, t: str) -> bool:
        """
        392. 判断子序列
        双指针/贪心
        """
        i, j = 0, 0

        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1

        return i == len(s)

    def partitionLabels(self, s: str) -> List[int]:
        """
        763. 划分字母区间
        贪心：记录每个字母最后出现位置
        """
        last_occurrence = {char: i for i, char in enumerate(s)}
        result = []
        start = end = 0

        for i, char in enumerate(s):
            end = max(end, last_occurrence[char])

            if i == end:
                result.append(end - start + 1)
                start = i + 1

        return result

    def maxOperations(self, nums: List[int], k: int) -> int:
        """
        1679. K和数对的最大数目
        贪心：排序 + 双指针
        """
        nums.sort()
        left, right = 0, len(nums) - 1
        operations = 0

        while left < right:
            current_sum = nums[left] + nums[right]

            if current_sum == k:
                operations += 1
                left += 1
                right -= 1
            elif current_sum < k:
                left += 1
            else:
                right -= 1

        return operations

    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        """
        1005. K次取反后最大化的数组和
        贪心：先翻转负数，再翻转最小正数
        """
        nums.sort()

        # 先翻转所有负数
        for i in range(len(nums)):
            if k > 0 and nums[i] < 0:
                nums[i] = -nums[i]
                k -= 1
            else:
                break

        # 如果k还有剩余，翻转最小的数
        if k > 0:
            nums.sort()
            if k % 2 == 1:
                nums[0] = -nums[0]

        return sum(nums)

    def balancedStringSplit(self, s: str) -> int:
        """
        1221. 分割平衡字符串
        贪心：计数
        """
        balance = 0
        count = 0

        for char in s:
            if char == 'L':
                balance += 1
            else:
                balance -= 1

            if balance == 0:
                count += 1

        return count

    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        """
        1710. 卡车上的最大单元数
        贪心：按单元数降序
        """
        boxTypes.sort(key=lambda x: -x[1])
        units = 0

        for boxes, units_per_box in boxTypes:
            if truckSize >= boxes:
                units += boxes * units_per_box
                truckSize -= boxes
            else:
                units += truckSize * units_per_box
                break

        return units

    def minOperations(self, nums: List[int]) -> int:
        """
        1827. 最少操作使数组递增
        贪心：逐个处理
        """
        operations = 0

        for i in range(1, len(nums)):
            if nums[i] <= nums[i - 1]:
                needed = nums[i - 1] + 1 - nums[i]
                operations += needed
                nums[i] = nums[i - 1] + 1

        return operations

    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        """
        3075. 幸福值最大化的选择
        贪心：排序后每次选最大的
        """
        happiness.sort(reverse=True)
        total = 0

        for i in range(k):
            value = max(0, happiness[i] - i)
            total += value
            if value == 0:
                break

        return total

    def latestTimeCatchTheBus(self, buses: List[int], passengers: List[int], capacity: int) -> int:
        """
        2332. 坐上公交的最晚时间
        贪心 + 模拟
        """
        buses.sort()
        passengers.sort()

        j = 0
        for b in buses:
            c = capacity
            while c and j < len(passengers) and passengers[j] <= b:
                j += 1
                c -= 1

        j -= 1
        ans = buses[-1] if c else passengers[j]

        while j >= 0 and ans == passengers[j]:
            j -= 1
            ans -= 1

        return ans


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 测试买卖股票
    print("=== 买卖股票的最佳时机 ===")
    print(solution.maxProfit([7,1,5,3,6,4]))

    # 测试跳跃游戏
    print("\n=== 跳跃游戏 ===")
    print(solution.jump([2,3,1,1,4]))

    # 测试合并区间
    print("\n=== 合并区间 ===")
    print(solution.merge([[1,3],[2,6],[8,10],[15,18]]))
