import itertools
import math, heapq, re
from bisect import bisect_right, bisect_left
from builtins import str
from collections import deque, Counter, defaultdict
from functools import cache
from itertools import pairwise, accumulate, combinations, permutations
from queue import PriorityQueue
from typing import List, Dict, Optional

from src.tool.Construct import ListNode, TreeNode


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
        # left = -1
        # sum = 0
        # ans = 0
        # for right, value in enumerate(nums):
        #     sum += value
        #     while sum >= k and left < right:
        #         if sum == k:
        #             ans += 1
        #         left += 1
        #         sum -= nums[left]
        # return ans
        pre_sum = 0
        ans = 0
        ctn = defaultdict(int)
        for x in nums:
            ctn[pre_sum] += 1
            pre_sum += x
            ans += ctn[k - pre_sum]
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
        # (nums[i] - nums[j]) * nums[k]  枚举nums[j] 维护nums[j]左边的最大值和右侧最大值  保证(nums[i] - nums[j]) * nums[k]最大
        n = len(nums)
        suf_max = [0] * (n + 1)
        for i in range(n - 1, 0, -1):
            suf_max[i] = max(suf_max[i + 1], nums[i])
        ans = pre_max = 0
        for j, x in enumerate(nums):
            ans = max(ans, (pre_max - x) * suf_max[j + 1])
            pre_max = max(pre_max, x)
        return ans

    def maximumSubsequenceCount(self, text: str, pattern: str) -> int:
        # 先统计text本身有多少个 pattern子序列 然后贪心的把pattern加在text两侧最大
        a, b = pattern
        ans = ctn_a = ctn_b = 0
        for ch in text:
            if ch == b:
                ans += ctn_a
                ctn_b += 1
            if ch == a:
                ctn_a += 1
        return ans + max(ctn_a, ctn_b)

    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        min_price = prices[0]
        for p in prices:
            ans = max(ans, p - min_price)
            min_price = min(min_price, p)
        return ans

    def maxProfit2(self, prices: List[int]) -> int:
        """
         n = len(prices)
        @cache
        def dfs(i: int, hold: bool) -> int:
            if i < 0:
                return -math.inf if hold else 0
            if hold:
                return max(dfs(i - 1, True), dfs(i - 1, False) - prices[i])
            return max(dfs(i - 1, False), dfs(i - 1, True) + prices[i])

        return dfs(n - 1, False)
        """
        buy = -prices[0]
        sell = 0
        for p in prices:
            buy = max(buy, sell - p)
            sell = max(sell, buy + p)
        return sell

    def maxProfit3(self, prices: List[int]) -> int:
        n = len(prices)
        buy_1 = buy_2 = -prices[0]
        sell_1 = sell_2 = 0
        for i in range(1, n):
            buy_1 = max(buy_1, -prices[i])
            sell_1 = max(sell_1, buy_1 + prices[i])
            buy_2 = max(buy_2, sell_1 - prices[i])
            sell_2 = max(sell_2, buy_2 + prices[i])
        return sell_2

    def differenceOfSum(self, nums: List[int]) -> int:
        ans = 0
        for x in nums:
            ans += x
            while x:
                x -= x % 10
                x = x // 10
        return ans

    def distinctNames(self, ideas: List[str]) -> int:
        groups = defaultdict(set)
        for s in ideas:
            groups[s[0]].add(s[1:])
        ans = 0
        # 2 * ((a - (a & b)) * （b - (a & b)）
        for a, b in permutations(groups.values(), 2):
            # 枚举所有的value组对
            m = len(a & b)  # a和b的交集
            ans += (len(a) - m) * (len(b) - m)
        return ans * 2

    def maximumLength(self, nums: List[int], k: int) -> int:
        """
        f[x][j] 表示以x结尾的  有至多j对相邻不用元素的最长子序列的长度

        f[x][j] = max(f[x][j] +1 ,max(f[y][j-1] for y in set  y != x ) + 1)
        mx[j] 表示 max(f[y][j-1] for y in set  y != x )
        f[x][j] = max(f[x][j] + 1, mx[j-1] + 1)

        """

        fs = {}
        mx = [0] * (k + 2)
        for x in nums:
            if x not in fs:
                fs[x] = [0] * (k + 2)
            f = fs[x]
            for j in range(k, -1, -1):
                f[j] = max(f[j], mx[j]) + 1
                mx[j + 1] = max(mx[j + 1], f[j])
        return mx[-1]

    def takeCharacters(self, s: str, k: int) -> int:
        ctn = Counter(s)
        if any(ctn[x] < k for x in 'abc'):
            return -1
        # 滑动窗口中 abc的数量最优解为 满足 ctn[a] - k,ctn[b] - k,ctn[c] - k的最大长度
        ans = left = 0

        for right, ch in enumerate(s):
            # 窗口右移
            ctn[ch] -= 1
            # 如果取走后 ctn[ch] 小于k说明 两侧的已经取不到k了 需要减小窗口
            while ctn[ch] < k:
                ctn[s[left]] += 1
                left += 1
            ans = max(ans, right - left + 1)
        return len(s) - ans

    def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
        """
            tickets[k] 前面的减少的次数不超过 min(tickets[i],tickets[k])
            tickets[k] 后面的减少的次数不超过 min(tickets[i],tickets[k]- 1)
        """
        return sum(min(val, tickets[k] - (i > k)) for i, val in enumerate(tickets))

    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = left = 0
        ctn = 0
        mx = max(nums)
        for x in nums:
            if x == mx:
                ctn += 1
            while ctn == k:
                if mx == nums[left]:
                    ctn -= 1
                left += 1
            ans += left
        return ans

    def countOfSubstrings(self, word: str, k: int) -> int:

        """
        f(word,k) 元音至少出现一次，并且至少有k个辅音字母的子字符串的总数
        那么恰好k个就是  f(word,k) -  f(word,k+1)   k+1的长度比k大 总数自然比k小
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

    def destCity(self, paths: List[List[str]]) -> str:
        set_a = set(p[0] for p in paths)
        return next(p[1] for p in paths if p[1] not in set_a)

    def minCost(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        # f[t][i] 表示经过t分钟 到达城市i的最小花费
        n = len(passingFees)
        f = [[math.inf] * n for _ in range(maxTime + 1)]
        # 在城市0节点的花费
        f[0][0] = passingFees[0]
        for i in range(1, maxTime + 1):
            for start, end, time in edges:
                # i - time花费的最小值
                if i - time >= 0:
                    # i  - end - start
                    f[i][start] = min(f[i][start], f[i - time][end] + passingFees[start])
                    # i - start -end
                    f[i][end] = min(f[i][end], f[i - time][start] + passingFees[end])
        ans = min(f[i][n - 1] for i in range(maxTime + 1))
        return ans if ans < math.inf else -1

    def romanToInt(self, s: str) -> int:
        dic = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        ans = 0
        for x, y in pairwise(s):
            x, y = dic[x], dic[y]
            ans += x if x > y else -x
        return ans + dic[s[-1]]

    def myAtoi(self, s: str) -> int:
        s = s.strip()
        if not s:
            return 0
        ans, sign, index = 0, 1, 1
        int_max, int_min, overflow = 2 ** 31 - 1, -2 ** 30, 2 ** 31 // 10  # 2147483647
        c1 = s[0]
        if c1 == '-1':
            sign = -1
        elif c1 != '+':
            index = 0
        for c in s[index:]:
            if not '0' <= c <= '9':
                break
            #  判断是否溢出 10 *  ans > 2147483650  or res = 214748364 并且尾数大于7
            if ans > overflow or ans == overflow and c > '7':
                return int_max if sign == 1 else int_min
            ans = ans * 10 + (ord(c) - ord('0'))
        return sign * ans

    def minimumDifference(self, nums: List[int], k: int) -> int:

        ans = min(abs(x - k) for x in nums)
        for i, x in enumerate(nums):

            j = i - 1
            while j >= 0 and nums[j] | x != nums[j]:
                nums[j] |= x
                ans = min(ans, abs(nums[j] - k))
                j -= 1
        return ans

    def longestPalindrome(self, s: str) -> str:
        """
        f(i,j) = p(i+1,j-1) & si = sj
            dp[i][j] 表示s[i:j]是否是回文串
        """
        n = len(s)
        if n < 2:
            return s
        max_len, begin = 1, 0
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True

        # 枚举子串长度 2-n
        for L in range(2, n + 1):
            # 枚举左端点
            for i in range(n):
                # L = j- i + 1
                j = i + L - 1
                if j >= n:
                    break

                if s[i] != s[j]:
                    dp[i][j] = False
                else:
                    # 如果长度小于等于3 默认就是回文串
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                # 如果s[i,j]是回文串 且长度大于最大值
                if dp[i][j] and L > max_len:
                    max_len = L
                    begin = i
        return s[begin:begin + max_len]

    def strStr(self, haystack: str, needle: str) -> int:
        ans = -1
        for i in range(len(haystack) - len(needle)):
            if haystack[i:i + len(needle)] == needle:
                return i
        return ans

    def maxSubArray(self, nums: List[int]) -> int:
        ans = -math.inf
        pre_sum = pre_min_sum = 0
        for x in nums:
            pre_sum += x
            pre_min_sum = min(pre_min_sum, pre_sum)
            ans = max(ans, pre_sum - pre_min_sum)
        return ans

    def rotate(self, nums: List[int], k: int) -> None:

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

    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:

        # 找出x //= k的 所有因子 然后sum在nums2中ctn的个数
        ctn = defaultdict(int)
        for x in nums1:
            if x % k:
                continue
            x //= k
            for d in range(1, math.isqrt(x) + 1):
                # 找出x的因子
                if x % d:
                    continue
                ctn[d] += 1
                if d * d < x:
                    ctn[x // d] += 1
        return sum(ctn[p] for p in nums2)

    def intToRoman(self, num: int) -> str:
        R = [
            ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX'],
            ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LX', 'LXXX', 'XC'],
            ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM'],
            ['', 'M', 'MM', 'MMM']
        ]
        return R[3][num // 1000] + R[2][num // 100 % 10] + R[1][num // 10 % 10] + R[0][num % 10]

    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        # 不符合子串
        if m < n:
            return 0
        f = [[0] * (n + 1) for _ in range(m + 1)]
        # 如果j = n 则空串是任何s的子串
        for i in range(m + 1):
            f[i][n] = 1
        # 后向前遍历
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if s[i] == t[j]:
                    f[i][j] = f[i + 1][j + 1] + f[i + 1][j]
                else:
                    f[i][j] = f[i + 1][j]
        return f[0][0]

    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        ans = vis = 0
        for x in nums:
            # 判断x 是否在vis集合中
            if vis >> x & 1:
                ans ^= x
            else:
                # 添加元素
                vis |= (1 << x)
        return ans

    def superEggDrop(self, k: int, n: int) -> int:
        '''
            dfs(i,j)表示还剩i次操作机会 j个鸡蛋的情况下确认f的最大建筑层数
            dfs(i,j)  = dfs(i-1,j) + dfs(i-1,j-1)+1
            边界条件 dfs(i,0) = dfs(0,j) = 0

            转化为dp
            f[i][j] = f[i-1][j] + f[i-1][j-1] + 1

            i 只和i-1有关联
            f[j] = f[j] + f[j-1] + 1

        '''

        f = [0] * (k + 1)
        for i in itertools.count(1):
            for j in range(k, 0, -1):
                f[j] = f[j] + f[j - 1] + 1
                if f[k] >= n:
                    return i

    def evenOddBit(self, n: int) -> List[int]:
        mask = 0x55555
        return [(mask & n).bit_count(), (mask >> 1 & n).bit_count()]

    def isPowerOfFour(self, n: int) -> bool:
        mask = 0xaaaaaaa
        return n >= 0 and n & (n - 1) == 0 and (mask & n) == 0

    def minChanges(self, n: int, k: int) -> int:
        return -1 if n & k != k else n ^ k.bit_count()

    def findKOr(self, nums: List[int], k: int) -> int:
        ans = 0
        for i in range(max(nums).bit_length()):
            ctn = sum(x >> i & 1 for x in nums)
            if ctn >= k:
                ans |= 1 << i
        return ans

    def minOperations(self, nums: List[int], k: int) -> int:

        ans = 0
        for x in nums:
            ans ^= x
        return (ans ^ k).bit_count()

    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        ctn = [0, 0]
        for i in itertools.count(1):
            ctn[i % 2] += i
            if (ctn[0] > red or ctn[1] > blue) and (ctn[0] > blue or ctn[1] > red):
                return i - 1

    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        pre_sum = 0
        ctn = defaultdict(int)
        ans = 0
        for x in nums:
            ctn[pre_sum] += 1
            pre_sum += x
            ans += ctn[pre_sum - goal]
        return ans

    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
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

    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans = left = 0
        s = 0
        for right, x in enumerate(nums):
            s += x
            while left <= right and s * (right - left + 1) >= k:
                s -= nums[left]
                left += 1
            ans += right - left + 1
        return ans

    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        mod = 1_000_000_007
        dic = {}
        for i, x in enumerate(nums):
            if x in dic:
                dic[x][1] = i
            else:
                dic[x] = [i, i]
        a = sorted(dic.values(), key=lambda p: p[0])
        m = 0
        max_r = a[0][1]
        for start, end in a[1:]:
            if start > max_r:
                m += 1
            max_r = max(max_r, end)
        m %= mod
        return pow(2, m)

    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        n = len(nums)
        i, j = 0, n - 1
        ans = math.inf
        while i < j:
            ans = min(ans, (nums[i] + nums[j]) / 2)
            j -= 1
            i += 1
        return ans

    def continuousSubarrays(self, nums: List[int]) -> int:
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

    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
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
        ctn = [] * 3
        ans = left = 0
        for right, x in enumerate(s):
            ctn[ord(x) - ord('a')] += 1
            while left <= right and (ctn[0] >= 1 and ctn[1] >= 1 and ctn[2] >= 1):
                ctn[ord(s[left]) - ord('a')] -= 1
                left += 1
            ans += left
        return ans

    def countGood(self, nums: List[int], k: int) -> int:
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

    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 1_00_000_007
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

    def minOperations(self, nums: List[int]) -> int:
        ans = 0
        # 模拟每一次反转 如果num[i] = 0 反转当前三个 判断nums[-1]和nums[-2] 是否是1  不是1 则不能全部反转
        for i in range(len(nums) - 2):
            if nums[i] == 0:
                nums[i + 1] ^= 1
                nums[i + 2] ^= 1
                ans += 1
        return ans if nums[-1] and nums[-2] else -1

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda p: p[0])
        merge = []
        for start, end in intervals:
            if merge and start <= merge[-1][1]:
                merge[-1][1] = max(end, merge[-1][1])
            else:
                merge.append([start, end])
        return merge

    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        def check(i: int) -> int:
            ans = left = 0
            ctn = 0
            for right, x in enumerate(nums):
                ctn += (x % 2)
                while left <= right and ctn > i:
                    y = nums[left]
                    # xxx 根据题意改编条件
                    ctn -= (y % 2)
                    left += 1
                ans += right - left + 1
            return ans

        return check(k) - check(k - 1)

    def minOperations(self, nums: List[int]) -> int:
        """
            x = nums[i] 分类讨论
            x = 0 k为奇数  or x = 1 k为偶数时 经过反转已经变成1 不需要操作
            x = 0 k为偶数  or x = 1 k为奇数时 经过反转已经变成0 需要操作
            so x == k % 2 则当前需要反转

        """
        k = 0
        for x in nums:
            if x == k % 2:
                k += 1
        return k

    def numOfSubarrays(self, arr: List[int]) -> int:
        mod = 1_000_000_007
        ans = pre_sum = 0
        ctn = [0] * 2
        for x in arr:
            ctn[pre_sum % 2] += 1
            pre_sum += x
            ans += ctn[0] if pre_sum % 2 else ctn[1]
        return ans % mod

    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        # (pre_sum(right) - pre_sum(left)) % k == 0  --> 求同余
        ans = pre_sum = 0
        ctn = defaultdict(int)
        for x in nums:
            ctn[pre_sum % k] += 1
            pre_sum += x
            ans += ctn[pre_sum % k]
        return ans

    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        pre_sum = 0
        ctn = defaultdict(int)
        for i, x in enumerate(nums):
            if pre_sum % k not in ctn:
                ctn[pre_sum % k] = i
            pre_sum += x
            if pre_sum % k in ctn and i - ctn[pre_sum % k] + 1 >= 2:
                return True
        return False

    def smallestRangeII(self, nums: List[int], k: int) -> int:
        nums.sort()
        ans = nums[-1] - nums[0]
        # 把nums分成[nums[0],...,nums[i]] [nums[i+1],...,nums[n-1]]
        for x, y in pairwise(nums):
            mx = max(nums[-1] - k, x + k)
            mi = min(nums[0] + k, y - k)
            ans = min(ans, mx - mi)
        return ans

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        ans = 0
        ctn = defaultdict(int)
        ctn[0] = 1

        def dfs(node: Optional[TreeNode], s: int):
            if node is None:
                return
            nonlocal ans
            s += node.val
            ans += ctn[s - targetSum]
            ctn[s] += 1
            dfs(node.left, s)
            dfs(node.right, s)
            # 递归完成之后 要恢复现场
            ctn[s] -= 1

        dfs(root, 0)
        return ans

    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:

        n = len(arr)
        xors = [0] * (n + 1)
        for i, x in enumerate(arr):
            xors[i + 1] = xors[i] ^ x
        ans = []
        for x, y in queries:
            ans.append(xors[y + 1] ^ xors[x])
        return ans

    def countCompleteDayPairs(self, hours: List[int]) -> int:
        ans, H = 0, 24
        ctn = [0] * H
        for x in hours:
            ans += ctn[(H - x % H) % H]
            ctn[x % H] += 1
        return ans

    def maximumTotalDamage(self, power: List[int]) -> int:
        ctn = Counter(power)
        a = sorted(ctn.keys())
        f = [0] * (len(a) + 1)
        j = 0
        for i, x in enumerate(a):
            while a[j] < x - 2:
                j += 1
            f[i + 1] = max(f[i], f[j] + x * ctn[x])
        return f[-1]

    def rob(self, nums: List[int]) -> int:
        # def dfs(i: int) -> int:
        #     if i < 0:
        #         return 0
        #     return max(dfs(i - 1), dfs(i - 2) + nums[i])
        #
        # return dfs(len(nums) - 1)
        # dfs(i) = max(dfs(i-1),dfs(i-2)+nums[i])
        n = len(nums)
        if n <= 2:
            return max(nums[0], nums[-1])
        f = [0] * n
        f[0], f[1] = nums[0], max(nums[1], nums[0])
        for i in range(2, n):
            f[i] = max(f[i - 1], f[i - 2] + nums[i])
        return f[-1]

    def findWinningPlayer(self, skills: List[int], k: int) -> int:
        return 1

    def validStrings(self, n: int) -> List[str]:
        ans = []
        path = [''] * n

        def dfs(i: int) -> None:
            if i == n:
                ans.append(''.join(path))
                return
            path[i] = '1'

            dfs(i + 1)

            if i == 0 or path[i - 1] == '1':
                path[i] = '0'
                dfs(i + 1)

        dfs(0)
        return ans

    def getSmallestString(self, s: str) -> str:
        a = list(s)
        for i in range(len(a) - 1):
            if a[i] > a[i + 1] and ord(a[i]) % 2 == ord(a[i + 1]) % 2:
                a[i], a[i + 1] = a[i + 1], a[i]
                break
        return ''.join(a)

    def maxEnergyBoost(self, energyDrinkA: List[int], energyDrinkB: List[int]) -> int:
        n = len(energyDrinkA)
        c = (energyDrinkA, energyDrinkB)

        @cache
        def dfs(i: int, j: int) -> int:
            if i < 0:
                return 0
            return max(dfs(i - 1, j), dfs(i - 2, j ^ 1)) + c[j][i]

        return max(dfs(n - 1, 0), dfs(n - 1, 1))

    def maxOperations(self, nums: List[int], k: int) -> int:
        ctn = Counter()
        ans = 0
        for x in nums:
            if ctn[k - x]:
                ctn[k - x] -= 1
                ans += 1
            else:
                ctn[x] += 1
        return ans

    def losingPlayer(self, x: int, y: int) -> str:
        return 'Alice' if min(x, y // 4) % 2 else 'Bob'

    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        ans = [-1] * (n - k + 1)
        ctn = 0
        for i, x in enumerate(nums):
            ctn = ctn + 1 if i == 0 or nums[i] == nums[i - 1] + 1 else 1
            if ctn >= k:
                ans[i - k + 1] = nums[i]
        return ans


s = Solution
print(s.resultsArray(s, [1, 2, 3, 4, 3, 2, 5], 3))
