"""
双指针 (Two Pointers) 题目集合

包含所有使用双指针技巧解决的题目
"""

from typing import List
from collections import Counter


class Solution:
    """双指针题目合集"""

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        1. 两数之和
        哈希表解法
        """
        dic = {}
        for i, val in enumerate(nums):
            if target - val in dic:
                return [i, dic[target - val]]
            dic[val] = i

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        15. 三数之和
        排序 + 双指针
        """
        nums.sort()
        n = len(nums)
        ans = []
        for i in range(n - 2):
            x = nums[i]
            if i and nums[i - 1] == nums[i]:
                continue
            if x + nums[i + 1] + nums[i + 2] > 0:
                break
            if x + nums[-1] + nums[-2] < 0:
                continue

            j, k = i + 1, n - 1
            while j < k:
                s = nums[i] + nums[j] + nums[k]
                if s > 0:
                    k -= 1
                elif s < 0:
                    j += 1
                else:
                    ans.append([nums[i], nums[j], nums[k]])
                    j += 1
                    k -= 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
        return ans

    def maxArea(self, height: List[int]) -> int:
        """
        11. 盛最多水的容器
        左右双指针
        """
        left, right = 0, len(height) - 1
        ans = 0
        while left < right:
            area = min(height[left], height[right]) * (right - left)
            ans = max(ans, area)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return ans

    def trap(self, height: List[int]) -> int:
        """
        42. 接雨水
        左右最大值
        """
        if not height:
            return 0

        left, right = 0, len(height) - 1
        left_max, right_max = height[left], height[right]
        ans = 0

        while left < right:
            if left_max < right_max:
                left += 1
                left_max = max(left_max, height[left])
                ans += left_max - height[left]
            else:
                right -= 1
                right_max = max(right_max, height[right])
                ans += right_max - height[right]

        return ans

    def sortedSquares(self, nums: List[int]) -> List[int]:
        """
        977. 有序数组的平方
        双指针从两端向中间
        """
        n = len(nums)
        ans = [0] * n
        i, j = 0, n - 1
        for p in range(n - 1, -1, -1):
            x = nums[i] * nums[i]
            y = nums[j] * nums[j]
            if x > y:
                ans[p] = x
                i += 1
            else:
                ans[p] = y
                j -= 1
        return ans

    def moveZeroes(self, nums: List[int]) -> None:
        """
        283. 移动零
        快慢指针
        """
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1

    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        3. 无重复字符的最长子串
        滑动窗口 + 哈希表
        """
        from collections import Counter
        ans = 0
        ctn = Counter()
        left = 0
        for right, ch in enumerate(s):
            ctn[ch] += 1
            while ctn[ch] > 1 and left < right:
                ctn[s[left]] -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    def longestConsecutive(self, nums: List[int]) -> int:
        """
        128. 最长连续序列
        哈希表 + 去重
        """
        num_set = set(nums)
        ans = 0
        for num in num_set:
            if num - 1 not in num_set:
                cur_num = num
                ctn = 1
                while cur_num + 1 in num_set:
                    ctn += 1
                    cur_num = cur_num + 1
                ans = max(ans, ctn)
        return ans

    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        """
        1385. 两个数组间的距离值
        排序 + 二分查找
        """
        from bisect import bisect_left
        arr2.sort()
        ans = 0
        for x in arr1:
            i = bisect_left(arr2, x - d)
            if i == len(arr2) or arr2[i] > x + d:
                ans += 1
        return ans

    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        """
        1014. 最佳观光组合
        维护左侧最大值
        """
        ans = mx = 0
        for i, val in enumerate(values):
            ans = max(ans, mx + val - i)
            mx = max(mx, i + val)
        return ans

    def minimumDifference(self, nums: List[int], k: int) -> int:
        """
        1984. 学生分数的最小差值
        排序 + 滑动窗口
        """
        nums.sort()
        n = len(nums)
        return min(nums[i] - nums[i - k + 1] for i in range(k - 1, n))

    def smallestRangeII(self, nums: List[int], k: int) -> int:
        """
        910. 最小差值 II
        排序 + 枚举分界点
        """
        from itertools import pairwise
        nums.sort()
        ans = nums[-1] - nums[0]
        for x, y in pairwise(nums):
            mx = max(nums[-1] - k, x + k)
            mi = min(nums[0] + k, y - k)
            ans = min(ans, mx - mi)
        return ans

    def maxNumOfMarkedIndices(self, nums: List[int]) -> int:
        """
        2576. 求出最多标记的下标
        排序 + 双指针
        """
        nums.sort()
        i = 0
        for x in nums[(len(nums) + 1) // 2:]:
            if 2 * nums[i] <= x:
                i += 1
        return i * 2

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """
        56. 合并区间
        排序 + 遍历合并
        """
        intervals.sort(key=lambda p: p[0])
        merge = []
        for start, end in intervals:
            if merge and start <= merge[-1][1]:
                merge[-1][1] = max(end, merge[-1][1])
            else:
                merge.append([start, end])
        return merge


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 测试三数之和
    print("=== 三数之和 ===")
    print(solution.threeSum([-1,0,1,2,-1,-4]))

    # 测试盛水容器
    print("\n=== 盛最多水的容器 ===")
    print(solution.maxArea([1,8,6,2,5,4,8,3,7]))

    # 测试最长连续序列
    print("\n=== 最长连续序列 ===")
    print(solution.longestConsecutive([100,4,200,1,3,2]))
