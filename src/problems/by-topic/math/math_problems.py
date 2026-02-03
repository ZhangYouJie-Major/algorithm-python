"""
数学 (Math) 题目集合

包含所有数学相关的题目
"""

from typing import List
import math


class Solution:
    """数学题目合集"""

    # ==================== 基础数学 ====================

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

    def isPalindrome(self, x: int) -> bool:
        """
        9. 回文数
        """
        if x < 0 or (x % 10 == 0 and x != 0):
            return False

        reverted = 0
        while x > reverted:
            reverted = reverted * 10 + x % 10
            x //= 10

        return x == reverted or x == reverted // 10

    def isPowerOfFour(self, n: int) -> bool:
        """
        342. 4的幂
        位运算
        """
        mask = 0xaaaaaaaa  # 排除2的幂中奇数位为1的情况
        return n >= 0 and n & (n - 1) == 0 and (mask & n) == 0

    def isPowerOfTwo(self, n: int) -> bool:
        """
        231. 2的幂
        位运算
        """
        return n > 0 and n & (n - 1) == 0

    # ==================== 位运算 ====================

    def evenOddBit(self, n: int) -> List[int]:
        """
        2595. 奇数位数和偶数位数
        """
        mask = 0x55555
        return [(mask & n).bit_count(), (mask >> 1 & n).bit_count()]

    def minChanges(self, n: int, k: int) -> int:
        """
        3222. 使 n 等于 k 的最少操作次数
        """
        return -1 if n & k != k else (n ^ k).bit_count()

    def findKOr(self, nums: List[int], k: int) -> int:
        """
        2917. 数组第K个异或值
        """
        ans = 0
        for i in range(max(nums).bit_length()):
            ctn = sum(x >> i & 1 for x in nums)
            if ctn >= k:
                ans |= 1 << i
        return ans

    def minOperations(self, nums: List[int], k: int) -> int:
        """
        3021. 将 Alice 和 Bob 的数字变为相等的最少操作次数
        """
        ans = 0
        for x in nums:
            ans ^= x
        return (ans ^ k).bit_count()

    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        """
        3158. 求出出现两次数字的 XOR 值
        """
        ans = vis = 0
        for x in nums:
            if vis >> x & 1:
                ans ^= x
            else:
                vis |= (1 << x)
        return ans

    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        """
        2429. 数组的前缀异或查询
        """
        n = len(arr)
        xors = [0] * (n + 1)
        for i, x in enumerate(arr):
            xors[i + 1] = xors[i] ^ x
        ans = []
        for x, y in queries:
            ans.append(xors[y + 1] ^ xors[x])
        return ans

    def countBeautifulPairs(self, nums: List[int]) -> int:
        """
        2748. 美丽下标对数目
        """
        n = len(nums)
        ctn = 0
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(int(str(nums[i])[0]), int(str(nums[i])[-1])) == 1:
                    ctn += 1
        return ctn

    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        """
        3164. 优质数对的总数 II
        """
        from collections import defaultdict
        ctn = defaultdict(int)
        for x in nums1:
            if x % k:
                continue
            x //= k
            for d in range(1, math.isqrt(x) + 1):
                if x % d:
                    continue
                ctn[d] += 1
                if d * d < x:
                    ctn[x // d] += 1
        return sum(ctn[p] for p in nums2)

    def differenceOfSum(self, nums: List[int]) -> int:
        """
        2535. 数组元素和与数字和的绝对差
        """
        ans = 0
        for x in nums:
            ans += x
            while x:
                ans -= x % 10
                x = x // 10
        return ans

    def minimumDifference(self, nums: List[int], k: int) -> int:
        """
        2953. 统计完全子数组的数目
        位运算 + 滑动窗口
        """
        ans = min(abs(x - k) for x in nums)
        for i, x in enumerate(nums):
            j = i - 1
            while j >= 0 and nums[j] | x != nums[j]:
                nums[j] |= x
                ans = min(ans, abs(nums[j] - k))
                j -= 1
        return ans

    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        """
        3243. 短路与和与
        """
        for i, x in enumerate(nums):
            if x == 2:
                nums[i] = -1
            else:
                t = -x
                nums[i] ^= (t & -t) >> 1
        return nums

    def countCompleteDayPairs(self, hours: List[int]) -> int:
        """
        3184. 构成整天的下标对数目 I
        """
        ans, H = 0, 24
        ctn = [0] * H
        for x in hours:
            ans += ctn[(H - x % H) % H]
            ctn[x % H] += 1
        return ans

    # ==================== 最大公约数 ====================

    def gcdOfStrings(self, str1: str, str2: str) -> str:
        """
        1071. 字符串的最大公因子
        """
        from math import gcd

        if str1 + str2 != str2 + str1:
            return ""

        return str1[:gcd(len(str1), len(str2))]

    # ==================== 组合数学 ====================

    def uniquePaths(self, m: int, n: int) -> int:
        """
        62. 不同路径
        组合数学
        """
        from math import comb
        return comb(m + n - 2, m - 1)

    def climbStairs(self, n: int) -> int:
        """
        70. 爬楼梯
        斐波那契数列
        """
        if n <= 2:
            return n

        a, b = 1, 2
        for _ in range(3, n + 1):
            a, b = b, a + b

        return b

    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        """
        3208. 交替组和 II
        """
        import itertools
        ctn = [0, 0]
        for i in itertools.count(1):
            ctn[i % 2] += i
            if (ctn[0] > red or ctn[1] > blue) and (ctn[0] > blue or ctn[1] > red):
                return i - 1

    # ==================== 进制转换 ====================

    def titleToNumber(self, columnTitle: str) -> int:
        """
        171. Excel 表列序号
        26进制转10进制
        """
        result = 0
        for char in columnTitle:
            result = result * 26 + (ord(char) - ord('A') + 1)
        return result

    def convertToTitle(self, columnNumber: int) -> str:
        """
        168. Excel表列名称
        10进制转26进制
        """
        result = []

        while columnNumber > 0:
            columnNumber -= 1
            result.append(chr(columnNumber % 26 + ord('A')))
            columnNumber //= 26

        return ''.join(reversed(result))


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 测试平方根
    print("=== x 的平方根 ===")
    print(solution.mySqrt(8))

    # 测试回文数
    print("\n=== 回文数 ===")
    print(solution.isPalindrome(121))

    # 测试2的幂
    print("\n=== 2的幂 ===")
    print(solution.isPowerOfTwo(16))
