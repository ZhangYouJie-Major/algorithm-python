"""
字符串 (String) 题目集合

包含所有字符串相关的题目
"""

from typing import List
import re
import math


class Solution:
    """字符串题目合集"""

    # ==================== 基础操作 ====================

    def strStr(self, haystack: str, needle: str) -> int:
        """
        28. 找出字符串中第一个匹配项的下标
        暴力匹配
        """
        ans = -1
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i + len(needle)] == needle:
                return i
        return ans

    def longestCommonPrefix(self, strs: List[str]) -> str:
        """
        14. 最长公共前缀
        """
        s0 = strs[0]
        for j, c in enumerate(s0):
            for s in strs:
                if len(s) == j or s[j] != c:
                    return s0[0:j]
        return s0

    # ==================== 回文串 ====================

    def longestPalindrome(self, s: str) -> str:
        """
        5. 最长回文子串
        中心扩散或动态规划
        """
        n = len(s)
        if n < 2:
            return s
        max_len, begin = 1, 0
        dp = [[False] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = True

        for L in range(2, n + 1):
            for i in range(n):
                j = i + L - 1
                if j >= n:
                    break

                if s[i] != s[j]:
                    dp[i][j] = False
                else:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]
                if dp[i][j] and L > max_len:
                    max_len = L
                    begin = i
        return s[begin:begin + max_len]

    # ==================== 字符串转换 ====================

    def romanToInt(self, s: str) -> int:
        """
        13. 罗马数字转整数
        """
        from itertools import pairwise
        dic = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        ans = 0
        for x, y in pairwise(s):
            x, y = dic[x], dic[y]
            ans += x if x > y else -x
        return ans + dic[s[-1]]

    def intToRoman(self, num: int) -> int:
        """
        12. 整数转罗马数字
        """
        R = [
            ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX'],
            ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LX', 'LXXX', 'XC'],
            ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM'],
            ['', 'M', 'MM', 'MMM']
        ]
        return R[3][num // 1000] + R[2][num // 100 % 10] + R[1][num // 10 % 10] + R[0][num % 10]

    def myAtoi(self, s: str) -> int:
        """
        8. 字符串转换整数 (atoi)
        """
        s = s.strip()
        if not s:
            return 0
        ans, sign, index = 0, 1, 1
        int_max, int_min, overflow = 2 ** 31 - 1, -2 ** 31, 2 ** 31 // 10
        c1 = s[0]
        if c1 == '-':
            sign = -1
        elif c1 != '+':
            index = 0
        for c in s[index:]:
            if not '0' <= c <= '9':
                break
            if ans > overflow or ans == overflow and c > '7':
                return int_max if sign == 1 else int_min
            ans = ans * 10 + (ord(c) - ord('0'))
        return sign * ans

    # ==================== 字符串匹配 ====================

    def numDistinct(self, s: str, t: str) -> int:
        """
        115. 不同的子序列
        动态规划
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

    # ==================== 字符串处理 ====================

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        """
        139. 单词拆分
        动态规划
        """
        word_set = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True

        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break

        return dp[n]

    def wordcount(self, s: str) -> dict:
        """
        统计单词频率
        """
        word_dict = {}
        words = re.findall(r'\b\w+\b', s)
        for word in words:
            word_dict[word] = word_dict.get(word, 0) + 1
        return word_dict

    def removeStars(self, s: str) -> str:
        """
        2390. 删除星号
        栈模拟
        """
        from collections import deque
        q = deque()
        for ch in s:
            if ch != '*':
                q.append(ch)
            else:
                q.pop()
        return "".join(q)

    def clearDigits(self, s: str) -> str:
        """
        3170. 删除字符串中的数字
        栈模拟
        """
        st = []
        for ch in s:
            if ch.isdigit():
                st.pop()
            else:
                st.append(ch)
        return "".join(st)

    def getSmallestString(self, s: str) -> str:
        """
        3216. 交换后字典序最小的字符串
        """
        a = list(s)
        for i in range(len(a) - 1):
            if a[i] > a[i + 1] and ord(a[i]) % 2 == ord(a[i + 1]) % 2:
                a[i], a[i + 1] = a[i + 1], a[i]
                break
        return ''.join(a)

    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        3. 无重复字符的最长子串
        """
        from collections import Counter
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

    def distinctNames(self, ideas: List[str]) -> int:
        """
        2306. 公司命名
        """
        from collections import defaultdict
        from itertools import permutations
        groups = defaultdict(set)
        for s in ideas:
            groups[s[0]].add(s[1:])
        ans = 0
        for a, b in permutations(groups.values(), 2):
            m = len(a & b)
            ans += (len(a) - m) * (len(b) - m)
        return ans * 2

    def takeCharacters(self, s: str, k: int) -> int:
        """
        2516. 每种字符至少取k个
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

    def longestContinuousSubstring(self, s: str) -> int:
        """
        2414. 最长的连续字母字符串
        """
        from itertools import pairwise
        ans = ctn = 1
        for x, y in pairwise(map(ord, s)):
            ctn = ctn + 1 if x + 1 == y else 1
            ans = max(ans, ctn)
        return ans

    def checkRecord(self, s: str) -> bool:
        """
        551. 学生出勤记录 I
        """
        return s.count('A') < 2 and 'LLL' not in s

    def validStrings(self, n: int) -> List[str]:
        """
        3211. 生成不含相邻零的二进制字符串
        """
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


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 测试最长回文子串
    print("=== 最长回文子串 ===")
    print(solution.longestPalindrome("babad"))

    # 测试罗马数字
    print("\n=== 罗马数字转换 ===")
    print(solution.romanToInt("MCMXCIV"))

    # 测试无重复字符的最长子串
    print("\n=== 无重复字符的最长子串 ===")
    print(solution.lengthOfLongestSubstring("abcabcbb"))
