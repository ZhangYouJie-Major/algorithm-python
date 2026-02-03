"""
滑动窗口 (Sliding Window) 题目集合

包含所有使用滑动窗口技巧解决的题目
"""

from typing import List
from collections import Counter, defaultdict
from bisect import bisect_left
from itertools import pairwise, accumulate
import math


class Solution:
    """滑动窗口题目合集"""

    # ==================== 固定窗口大小 ====================

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        239. 滑动窗口最大值
        单调队列 + 滑动窗口
        """
        ans = []
        q = []
        for index, value in enumerate(nums):
            while q and nums[q[-1]] <= value:
                q.pop()
            q.append(index)

            if index - q[0] >= k:
                q.pop(0)

            if index >= k - 1:
                ans.append(nums[q[0]])
        return ans

    def maxVowels(self, s: str, k: int) -> int:
        """
        1456. 定长子串中元音的最大数目
        固定窗口大小的滑动窗口
        """
        ans = windows = 0
        for i, val in enumerate(s):
            if val in 'aeiou':
                windows += 1
            if i < k - 1:
                continue
            ans = max(ans, windows)
            if s[i - k + 1] in 'aeiou':
                windows -= 1
        return ans

    def minimumRecolors(self, blocks: str, k: int) -> int:
        """
        2379. 得到 K 个黑块的最少涂色次数
        统计窗口中白色块的最小值
        """
        ans = math.inf
        ctn = 0
        for i, b in enumerate(blocks):
            if b == 'W':
                ctn += 1
            if i < k - 1:
                continue
            ans = min(ans, ctn)
            ctn -= blocks[i - k + 1] == 'W'
        return ans

    # ==================== 可变窗口大小 ====================

    def maximumLengthSubstring(self, s: str) -> int:
        """
        2730. 找到最长的半重复子字符串
        每个字符最多出现2次
        """
        left = ans = 0
        ctn = defaultdict(int)
        for right, val in enumerate(s):
            ctn[val] += 1
            while any(value > 2 for value in ctn.values()):
                ctn[s[left:left + 1]] -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    def longestSubarray(self, nums: List[int]) -> int:
        """
        1493. 删掉一个元素以后全为 1 的最长子数组
        最多一个0
        """
        left = ans = 0
        ctn = [0] * 2
        for right, val in enumerate(nums):
            ctn[val] += 1
            while ctn[0] > 1:
                ctn[nums[left]] -= 1
                left += 1
            ans = max(ans, right - left)
        return ans if ans != len(nums) else len(nums) - 1

    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        """
        1208. 尽可能使字符串相等
        窗口内差值和不超过maxCost
        """
        ans = left = 0
        diff = 0
        for right, (ch1, ch2) in enumerate(zip(s, t)):
            diff += abs(ord(ch1) - ord(ch2))
            while diff > maxCost:
                diff -= abs(ord(s[left]) - ord(t[left]))
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        """
        2730. 最长半重复子字符串
        最多一对相邻相同字符
        """
        ans, left, repeat = 1, 0, 0
        n = len(s)
        for right in range(1, n):
            repeat += s[right] == s[right - 1]
            while repeat > 1:
                if s[left] == s[left + 1]:
                    repeat -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    def totalFruit(self, fruits: List[int]) -> int:
        """
        904. 水果成篮
        最多两种类型
        """
        ans = left = 0
        ctn = defaultdict(int)
        for right, val in enumerate(fruits):
            ctn[val] += 1
            while len(ctn.keys()) > 2:
                ctn[fruits[left]] -= 1
                if ctn[fruits[left]] == 0:
                    del ctn[fruits[left]]
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        """
        1695. 最大子数组的唯一元素和
        所有元素不重复
        """
        mx = sum_val = left = 0
        ctn = defaultdict(int)
        for right, val in enumerate(nums):
            ctn[val] += 1
            sum_val += val
            while ctn[val] > 1:
                ctn[nums[left]] -= 1
                sum_val -= nums[left]
                left += 1
            mx = max(mx, sum_val)
        return mx

    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        """
        2958. 最多K个重复元素的最长子数组
        每个元素最多出现k次
        """
        ans = left = 0
        ctn = defaultdict(int)
        for right, val in enumerate(nums):
            ctn[val] += 1
            while ctn[val] > k:
                ctn[nums[left]] -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    def maximumBeauty(self, nums: List[int], k: int) -> int:
        """
        2779. 数组的最大美丽值
        排序后滑动窗口，最大最小值差不超过2k
        """
        nums.sort()
        ans = left = 0
        for right, val in enumerate(nums):
            while val - nums[left] > 2 * k:
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    def longestContinuousSubstring(self, s: str) -> int:
        """
        2414. 最长的连续字母字符串
        连续递增的字母
        """
        ans = ctn = 1
        for x, y in pairwise(map(ord, s)):
            ctn = ctn + 1 if x + 1 == y else 1
            ans = max(ans, ctn)
        return ans

    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        """
        2461. 长度为K子数组中的最大和
        恰好k个不同元素
        """
        ctn = Counter(nums[:k - 1])
        s = sum(nums[:k - 1])
        ans = 0
        for in_, out in zip(nums[k - 1:], nums):
            ctn[in_] += 1
            s += in_
            if len(ctn.keys()) == k:
                ans = max(ans, s)
            ctn[out] -= 1
            if ctn[out] == 0:
                del ctn[out]
            s -= out
        return ans

    def maxScore(self, cardPoints: List[int], k: int) -> int:
        """
        1423. 可获得的最大点数
        找n-k个连续的最小值
        """
        n = len(cardPoints)
        sum_ = sum(cardPoints)
        m = n - k
        ans = 0
        m_sum = sum(cardPoints[:m])
        ans = max(ans, sum_ - m_sum)
        for i in range(m, n):
            m_sum += cardPoints[i]
            m_sum -= cardPoints[i - m]
            ans = max(ans, sum_ - m_sum)
        return ans

    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        """
        2461. 长度为K子数组中的最大和（至少m个不同元素）
        """
        ctn = Counter()
        s = ans = 0
        for i, x in enumerate(nums):
            ctn[x] += 1
            s += x
            if i < k - 1:
                continue
            if len(ctn.keys()) >= m:
                ans = max(ans, s)
            remove_val = nums[i - k + 1]
            s -= remove_val
            ctn[remove_val] -= 1
            if ctn[remove_val] == 0:
                ctn.pop(remove_val)
        return ans

    # ==================== 计数类滑动窗口 ====================

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        """
        713. 乘积小于K的子数组数量
        """
        ans = left = 0
        s = 1
        for right, x in enumerate(nums):
            s *= x
            while left <= right and s >= k:
                s /= nums[left]
                left += 1
            ans += right - left + 1
        return ans

    def beautifulBouquet(self, flowers: List[int], ctn: int) -> int:
        """
        2264. 鲜花最多的花束
        每种花最多ctn朵
        """
        mod = 10 ** 9 + 7
        ans = left = 0
        c = defaultdict(int)
        for right, x in enumerate(flowers):
            c[x] += 1
            while c[x] > ctn:
                c[flowers[left]] -= 1
                left += 1
            ans += right - left + 1
        return ans % mod

    def numberOfSubstrings(self, s: str) -> int:
        """
        1358. 包含所有三种字符的子字符串数量
        """
        ctn = [0] * 3
        ans = left = 0
        for right, x in enumerate(s):
            ctn[ord(x) - ord('a')] += 1
            while left <= right and (ctn[0] >= 1 and ctn[1] >= 1 and ctn[2] >= 1):
                ctn[ord(s[left]) - ord('a')] -= 1
                left += 1
            ans += left
        return ans

    def countGood(self, nums: List[int], k: int) -> int:
        """
        2537. 统计好子数组的数量
        至少k对不同元素对
        """
        ans = left = 0
        ctn = defaultdict(int)
        s = 0
        for right, x in enumerate(nums):
            s += ctn[x]
            ctn[x] += 1
            while left < right and s >= k:
                y = nums[left]
                left += 1
                ctn[y] -= 1
                s -= ctn[y]
                if ctn[y] == 0:
                    del ctn[y]
            ans += left
        return ans

    def countCompleteSubarrays(self, nums: List[int]) -> int:
        """
        2808. 完整子数组的数量
        包含所有出现的不同元素
        """
        k = len(Counter(nums).keys())
        ans = left = 0
        ctn = Counter()
        for right, x in enumerate(nums):
            ctn[x] += 1
            while left <= right and len(ctn.keys()) == k:
                y = nums[left]
                left += 1
                ctn[y] -= 1
                if ctn[y] == 0:
                    del ctn[y]
            ans += left
        return ans

    def continuousSubarrays(self, nums: List[int]) -> int:
        """
        2762. 不间断子数组
        最大最小值差不超过2
        """
        ans = left = 0
        ctn = Counter()
        for right, x in enumerate(nums):
            ctn[x] += 1
            while max(ctn) - min(ctn) > 2:
                y = nums[left]
                left += 1
                ctn[y] -= 1
                if ctn[y] == 0:
                    del ctn[y]
            ans += right - left + 1
        return ans

    def maxOperations(self, nums: List[int], k: int) -> int:
        """
        1679. K和数对的最大数目
        """
        ctn = Counter()
        ans = 0
        for x in nums:
            if ctn[k - x]:
                ctn[k - x] -= 1
                ans += 1
            else:
                ctn[x] += 1
        return ans

    def minimumAverage(self, nums: List[int]) -> float:
        """
        3094. 最小平均值差
        """
        nums.sort()
        n = len(nums)
        i, j = 0, n - 1
        ans = math.inf
        while i < j:
            ans = min(ans, (nums[i] + nums[j]) / 2)
            j -= 1
            i += 1
        return ans

    def maximumTotalDamage(self, power: List[int]) -> int:
        """
        3186. 最大破坏力
        相邻差值>2的数字
        """
        ctn = Counter(power)
        a = sorted(ctn.keys())
        f = [0] * (len(a) + 1)
        j = 0
        for i, x in enumerate(a):
            while a[j] < x - 2:
                j += 1
            f[i + 1] = max(f[i], f[j] + x * ctn[x])
        return f[-1]

    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        """
        992. K个不同整数的子数组
        恰好k个 = 最多k个 - 最多k-1个
        """
        def cal(i: int) -> int:
            ctn = Counter()
            ans = left = 0
            for right, x in enumerate(nums):
                ctn[x] += 1
                while len(ctn.keys()) > i and left <= right:
                    c = nums[left]
                    left += 1
                    ctn[c] -= 1
                    if ctn[c] == 0:
                        del ctn[c]
                ans += right - left + 1
            return ans

        return cal(k) - cal(k - 1)

    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        """
        1248. 统计优美子数组
        恰好k个奇数
        """
        def check(i: int) -> int:
            ans = left = 0
            ctn = 0
            for right, x in enumerate(nums):
                ctn += (x % 2)
                while left <= right and ctn > i:
                    y = nums[left]
                    ctn -= (y % 2)
                    left += 1
                ans += right - left + 1
            return ans

        return check(k) - check(k - 1)

    def countOfSubstrings(self, word: str, k: int) -> int:
        """
        3389. 至少K个辅音的元音子字符串
        至少k个 = f(k) - f(k+1)
        """
        def f(s: str, k: int) -> int:
            ctn_1 = defaultdict(int)
            ctn_2 = left = ans = 0
            for ch in s:
                if ch in 'aeiou':
                    ctn_1[ch] += 1
                else:
                    ctn_2 += 1
                while len(ctn_1) == 5 and ctn_2 >= k:
                    out = s[left]
                    if out in 'aeiou':
                        ctn_1[out] -= 1
                        if ctn_1[out] == 0:
                            del ctn_1[out]
                    else:
                        ctn_2 -= 1
                    left += 1
                ans += left
            return ans

        return f(word, k) - f(word, k + 1)

    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        """
        2024. 考试的最大干扰问题
        最多改变k个
        """
        def maxConsecutiveChar(ch: str) -> int:
            ans, left, sum_val = 0, 0, 0
            for right in range(len(answerKey)):
                sum_val += answerKey[right] != ch
                while sum_val > k:
                    sum_val -= answerKey[left] != ch
                    left += 1
                ans = max(ans, right - left + 1)
            return ans

        return max(maxConsecutiveChar('T'), maxConsecutiveChar('F'))

    def maximumRobots(self, chargeTimes: List[int], runningCosts: List[int], budget: int) -> int:
        """
        2398. 预算内的最多机器人数目
        单调队列 + 滑动窗口
        """
        from collections import deque
        q = deque()
        ans = s = left = 0

        for right, (c, r) in enumerate(zip(chargeTimes, runningCosts)):
            while q and c >= chargeTimes[q[-1]]:
                q.pop()
            q.append(right)
            s += r

            while q and chargeTimes[q[0]] + s * (right - left + 1) > budget:
                if q[0] == left:
                    q.popleft()
                s -= runningCosts[left]
                left += 1
            ans = max(ans, right - left + 1)
        return ans


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 测试滑动窗口最大值
    print("=== 滑动窗口最大值 ===")
    print(solution.maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3))  # [3,3,5,5,6,7]

    # 测试元音最大数目
    print("\n=== 元音最大数目 ===")
    print(solution.maxVowels("abciiidef", 3))  # 3
