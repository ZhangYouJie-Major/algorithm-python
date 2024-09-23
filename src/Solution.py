import re
from bisect import bisect_right, bisect_left
from builtins import str
from collections import deque, Counter, defaultdict
from functools import cache
from queue import PriorityQueue
from typing import List, Dict, Optional
import math

from src.tool import Construct
from src.tool.Construct import ListNode
from itertools import pairwise, accumulate


class Solution:
    MOD = 1_000_000_007

    def wordcount(self, s: str) -> Dict[str, int]:
        word_dict = {}
        words = re.findall(r'\b\w+\b', s)
        for word in words:
            word_dict[word] = word_dict.get(word, 0) + 1
        return word_dict

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        ans = []
        q = deque()
        for index, value in enumerate(nums):
            while q and nums[q[-1]] <= value:
                q.pop()
            q.append(index)

            #  出
            if index - q[0] >= k:
                q.popleft()

            if index >= k - 1:
                ans.append(nums[q[0]])
        return ans

    def subarraySum(self, nums: List[int], k: int) -> int:
        left = -1
        sum = 0
        ans = 0
        for right, value in enumerate(nums):
            sum += value
            while sum >= k and left < right:
                if sum == k:
                    ans += 1
                left += 1
                sum -= nums[left]
        return ans

    def lengthOfLongestSubstring(self, s: str) -> int:

        ans = 0
        ctn = Counter()
        left = 0
        for right, ch in enumerate(s):
            ctn[ch] += 1
            if ch in ctn:
                while ctn[ch] > 1 and left < right:
                    ctn[s[left]] -= 1
                    left += 1
                ans = max(ans, right - left + 1)
        return ans

    def findKthLargest(self, nums: List[int], k: int) -> int:
        q = PriorityQueue()
        for item in nums:
            q.put((-item, item))
        for index in range(k):
            q.get()
            if index == k - 1:
                return q.get()[1]

        return -1

    def isArraySpecial(self, nums: List[int], queries: List[List[int]]) -> List[bool]:
        arr = list(accumulate((x % 2 == y % 2 for x, y in pairwise(nums)), initial=0))
        return [arr[from_] == arr[to] for from_, to in queries]

    def checkRecord(self, s: str) -> bool:
        return s.count('A') < 2 and 'LLL' not in s

    def checkRecord(self, n: int) -> int:
        return 1

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i, val in enumerate(nums):
            if target - val in dic:
                return [i, dic[target - val]]
            dic[val] = i

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic = defaultdict(list)
        for s in strs:
            dic[''.join(sorted(s))].append(s)
        return list(dic.values())

    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        ans = 0
        for num in num_set:
            if num - 1 not in num_set:
                # 判断是最小的值开始递增
                cur_num = num
                ctn = 1
                while cur_num + 1 in num_set:
                    ctn += 1
                    cur_num = cur_num + 1
                ans = max(ans, ctn)
        return ans

    def findPermutationDifference(self, s: str, t: str) -> int:
        pos = {c: i for i, c in enumerate(s)}
        return sum(abs(pos[c] - i) for i, c in enumerate(t))

    def satisfiesConditions(self, grid: List[List[int]]) -> bool:
        for i, row in enumerate(grid):
            for j, col in enumerate(row):
                if j and col == row[j - 1] or i and grid[i][j] == grid[i - 1][j]:
                    return False

    def canMakeSquare(self, grid: List[List[str]]) -> bool:

        def check(i: int, j: int) -> bool:
            ctn = defaultdict(int)
            ctn[grid[i][j]] += 1
            ctn[grid[i + 1][j]] += 1
            ctn[grid[i][j + 1]] += 1
            ctn[grid[i + 1][j + 1]] += 1
            return ctn['B'] >= 3 or ctn['W'] >= 3

        return check(0, 0) or check(0, 1) or check(1, 0) or check(1, 1)

    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:

        def maxConsecutiveChar(ch: str) -> int:
            ans, left, sum = 0, 0, 0
            # 输入ch 统计非ch的数量不超过K
            for right in range(len(answerKey)):
                sum += answerKey[right] != ch
                while sum > k:
                    sum -= answerKey[left] != ch
                    left += 1
                ans = max(ans, right - left + 1)
            return ans

        return max(maxConsecutiveChar('T'), maxConsecutiveChar('F'))

    def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
        return sum(s <= queryTime <= e for s, e in zip(startTime, endTime))

    def maxStrength(self, nums: List[int]) -> int:
        mn, mx = nums[0]
        # 不选，只选x 最小值 * x 最大值*x
        for x in nums[1:]:
            mn, mx = max(mn, x, mn * x, mx * x), min(mx, x, mn * x, mx * x)
        return max(mn, mx)

    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        ans = 0
        # 枚举选中了k个元素  前k要小于l 后n-k要大于k
        for k in range(0, n + 1):
            if 0 < k <= nums[k - 1]:
                continue
            if n > k >= nums[k]:
                continue
            ans += 1
        return ans

    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)

        # 以nums[i]结尾的 至多有j个的最大长度
        @cache
        def dfs(i: int, j: int) -> int:
            if i == 0:
                return 0
            mx = 0
            # 枚举j之前的每一个元素
            for p in range(i):
                if nums[p] == nums[i]:
                    mx = max(mx, dfs(p, j) + 1)
                elif p and nums[p] != nums[i]:
                    mx = max(mx, dfs(p, j - 1) + 1)
            return mx

        return max(dfs(i, k) for i in range(n - 1, -1, -1))

    def clearDigits(self, s: str) -> str:
        st = []
        for ch in s:
            # 如果遇到数组 上一个肯定是字母 直接弹出栈顶元素即可
            if str.isdigit(ch):
                st.pop()
            else:
                st.append(ch)
        return "".join(st)

    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # ans 1
        tail = head
        cur = head.next
        while cur.next:
            if cur.val:
                tail.val += cur.val
            else:
                # 重置头结点 sum
                tail = tail.next
                tail.val = 0
            cur = cur.next
        tail.next = None
        return head

    def sortedSquares(self, nums: List[int]) -> List[int]:
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

    def countQuadruplets(self, nums: List[int]) -> int:
        n = len(nums)
        great = [0] * n
        great[-1] = [0] * (n + 1)
        # great[k][num[j]] 在k的右侧比nums[j]大的元素
        # less[j][nums[k]] 在j的左侧 比nums[k]小的元素
        for k in range(n - 2, -1, -1):
            great[k] = great[k + 1].copy()
            for x in range(1, nums[k + 1]):
                great[k][x] += 1
        ans = 0
        less = [0] * n
        less[0] = [0] * (n + 1)
        for j in range(1, n - 1):
            for x in range(nums[j - 1] + 1, n + 1):
                less[x] += 1
            for k in range(j + 1, n - 1):
                if nums[j] > nums[k]:
                    ans += less[nums[k]] * great[nums[j]]
        return ans

    def maximizeWin(self, prizePositions: List[int], k: int) -> int:
        """
            假设第二段右端点为prizePositions[i] 则左端点为prizePositions[i]-k

            dp[i] 表示右端点不超过prizePositions[i] 一条线段可以
            1、 不选 prizePositions[i] dp[i] = dp[i-1]
            2、 选 dp[i] = i-j+1
            dp[i] = max(dp[i-1], i-j+1)

        """
        n = len(prizePositions)
        dp = [0] * (n + 1)
        ans = 0
        for i in range(n):
            x = bisect_left(prizePositions, prizePositions[i] - k)
            ans = max(ans, dp[x] + i - x + 1)
            dp[i + 1] = max(dp[i], i - x + 1)
        return ans

    def maxNumOfMarkedIndices(self, nums: List[int]) -> int:
        nums.sort()
        i = 0
        # 取另外一遍的数字 如果能匹配 两边做指针右移
        for x in nums[(len(nums) + 1) // 2:]:
            if 2 * nums[i] <= x:
                i += 1
        return i * 2

    def maximumRobots(self, chargeTimes: List[int], runningCosts: List[int], budget: int) -> int:
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

    def maxVowels(self, s: str, k: int) -> int:
        """
            1、 入i元素  如果i < k-1 则重复第一步
            2、更新元素
            3、出 i- k +1 元素
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

    def maximumLengthSubstring(self, s: str) -> int:
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

    def removeStars(self, s: str) -> str:
        # 栈模拟
        q = deque()
        for ch in s:
            if ch != '*':
                q.append(ch)
            else:
                q.pop()
        return "".join(q)

    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        mx = sum = left = 0
        ctn = defaultdict(int)
        for right, val in enumerate(nums):
            ctn[val] += 1
            sum += val
            while ctn[val] > 1:
                ctn[nums[left]] -= 1
                sum -= nums[left]
                left += 1
            mx = max(mx, sum)

        return mx

    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
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

        # 因为变化的上下区间可以浮动 -> 把所有数字排序后 交集不为空的最大长度
        # 我们保证最左侧和最右侧的交集不为空 则中间的交集不会为空
        # x + k >= y - k ->  y - x <= 2k

        nums.sort()
        ans = left = 0
        for right, val in enumerate(nums):
            while val - nums[left] > 2 * k:
                left += 1
            ans = max(ans, right - left + 1)
        return ans

    def latestTimeCatchTheBus(self, buses: List[int], passengers: List[int], capacity: int) -> int:
        # 先排序
        buses.sort()
        passengers.sort()

        # 模拟乘客上车
        j = 0
        for b in buses:
            c = capacity

            # passengers[j] <= b 下车时间在出发前的  上车
            while c and j < len(passengers) and passengers[j] <= b:
                j += 1
                c -= 1
        j -= 1
        # 插队  如果还有座位直接在末站下车
        ans = buses[-1] if c else passengers[j]

        # 插队 寻找j 上一个没下车的坐标
        while j >= 0 and ans == passengers[j]:
            j -= 1
            ans -= 1
        return ans

    def longestContinuousSubstring(self, s: str) -> int:
        ans = ctn = 1
        for x, y in pairwise(map(ord, s)):
            ctn = ctn + 1 if x + 1 == y else 1
            ans = max(ans, ctn)
        return ans

    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
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
        # k张牌的最大值 n-k的最小值
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

    def countSpecialNumbers(self, n: int) -> int:
        s = str(n)

        """
            is_limit 表示前面填的数字是否都是n对应位置的 如果为True 则当前位置最多为int(s[i]) 否则至多为9
            is_num 表示前面是否填了数字  如果为True 则当前可以从0开始 False则需要从1开始  保证不出现 010这样的数据
        """

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
        s = str(n)

        """
            is_limit 表示前面填的数字是否都是n对应位置的 如果为True 则当前位置最多为int(s[i]) 否则至多为9
            is_num 表示前面是否填了数字  如果为True 则当前可以从0开始 False则需要从1开始  保证不出现 010这样的数据
        """

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
        s = str(n)
        """
            dfs(i,ctn,is_limit) 表示前i位有ctn个1的情况下 我们可以构造的数组 包含1的总和
            is_limit 表示前面填的数字是否都是n对应位置的 如果为True 则当前位置最多为int(s[i]) 否则至多为9
        """

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

    def edgeScore(self, edges: List[int]) -> int:
        score = [0] * len(edges)
        ans = 0
        for i, to in enumerate(edges):
            score[to] += i
            if score[to] > score[ans] or score[to] == score[ans] and to < ans:
                ans = to
        return ans

    def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
        s = [0] * (len(words) + 1)
        for i, val in enumerate(words):
            s[i + 1] = s[i] + (val[0] in 'aeiou' and val[-1] in 'aeiou')
        ans = [] * (len(queries))
        for q in queries:
            ans.append(s[q[1] + 1] - s[q[0]])
        return ans

    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        nums.sort()
        for i in range(1, len(nums)):
            nums[i] += nums[i - 1]
        for i, val in enumerate(queries):
            queries[i] = bisect_right(nums, val)
        return queries

    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        in_ = Counter(y for _, y in trust)
        out_ = Counter(x for x, _ in trust)
        return next((i for i in range(1, n + 1) if in_[i] == n - 1 and out_[i] == 0), -1)

    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        ans = mx = 0
        for i, val in enumerate(values):
            ans = max(ans, mx + val - i)
            mx = max(mx, i + val)
        return ans

    def countBeautifulPairs(self, nums: List[int]) -> int:
        n = len(nums)
        ctn = 0
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(int(str(nums[i])[0]), int(str(nums[i])[-1])) == 1:
                    ctn += 1
        return ctn

    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        pre_max = [-1] * n
        ans = 0
        for i in range(1, n):
            pre_max[i] = max(pre_max[i - 1], nums[i - 1])
        suf_max = nums[-1]
        for i in range(n - 2, 0, -1):
            ans = max(ans, (pre_max[i] - nums[i]) * suf_max)
            suf_max = max(suf_max, nums[i])
        return ans


s = Solution
print(s.maximumTripletValue(s, [1000000, 1, 1000000]))
