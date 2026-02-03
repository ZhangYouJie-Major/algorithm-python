"""
回溯 (Backtracking) 题目集合

包含所有使用回溯算法解决的题目
"""

from typing import List
from collections import deque


class Solution:
    """回溯题目合集"""

    def generateParenthesis(self, n: int) -> List[str]:
        """
        22. 括号生成
        """
        ans = []
        path = []

        def dfs(i, balance):
            if len(path) == n:
                s = [')'] * (2 * n)
                for j in path:
                    s[j] = '('
                ans.append(''.join(s))
                return
            for right in range(balance + 1):
                path.append(i + right)
                dfs(i + right + 1, balance - right + 1)
                path.pop()

        dfs(0, 0)
        return ans

    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        46. 全排列
        """
        n = len(nums)
        ans = []
        path = [0] * n
        on_path = [False] * n

        def dfs(i):
            if i == n:
                ans.append(path[:])
                return
            for j, on in enumerate(on_path):
                if not on:
                    path[i] = nums[j]
                    on_path[j] = True
                    dfs(i + 1)
                    on_path[j] = False

        dfs(0)
        return ans

    def combine(self, n: int, k: int) -> List[List[int]]:
        """
        77. 组合
        """
        ans = []
        path = []

        def f(start):
            if len(path) == k:
                ans.append(path.copy())
                return

            for j in range(start, n + 1):
                path.append(j)
                f(j + 1)
                path.pop()

        f(1)
        return ans

    def solveNQueens(self, n: int) -> List[List[str]]:
        """
        51. N皇后
        """
        ans = []
        col = [0] * n

        def dfs(r, s):
            if r == n:
                ans.append(['.' * c + 'Q' + '.' * (n - 1 - c) for c in col])
                return
            for c in s:
                if all((r + c != R + col[R] and c - r != col[R] - R) for R in range(r)):
                    col[r] = c
                    dfs(r + 1, s - {c})

        dfs(0, set(range(n)))
        return ans

    def letterCombinations(self, digits: str) -> List[str]:
        """
        17. 电话号码的字母组合
        """
        if not digits:
            return []

        MAPPING = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
        n = len(digits)
        ans = []
        path = [""] * n

        def f(i):
            if i == n:
                ans.append("".join(path))
                return
            for c in MAPPING[int(digits[i])]:
                path[i] = c
                f(i + 1)

        f(0)
        return ans

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        78. 子集
        """
        ans = []
        path = []

        def dfs(i):
            if i == len(nums):
                ans.append(path[:])
                return
            # 不选
            dfs(i + 1)
            # 选
            path.append(nums[i])
            dfs(i + 1)
            path.pop()

        dfs(0)
        return ans

    def binaryTreePaths(self, root):
        """
        257. 二叉树的所有路径
        """
        ans = []

        def dfs(node, path):
            if node is None:
                return
            path += str(node.val)
            if not node.left and not node.right:
                ans.append(path)
            path += '->'
            dfs(node.left, path)
            dfs(node.right, path)

        dfs(root, '')
        return ans

    def pathSum(self, root, targetSum: int):
        """
        113. 路径总和 II
        """
        ans = []
        path = []

        def dfs(node, rem):
            if node is None:
                return
            rem -= node.val
            path.append(node.val)
            if not node.left and not node.right and rem == 0:
                ans.append(path[:])
            dfs(node.left, rem)
            dfs(node.right, rem)
            path.pop()

        dfs(root, targetSum)
        return ans

    def restoreIpAddresses(self, s: str) -> List[str]:
        """
        93. 复原IP地址
        """
        ans = []
        path = []

        def dfs(start, part):
            if part == 4:
                if start == len(s):
                    ans.append('.'.join(path))
                return

            for length in range(1, 4):
                if start + length > len(s):
                    break
                segment = s[start:start + length]

                # 验证段是否合法
                if (segment[0] == '0' and len(segment) > 1) or int(segment) > 255:
                    continue

                path.append(segment)
                dfs(start + length, part + 1)
                path.pop()

        dfs(0, 0)
        return ans


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 测试括号生成
    print("=== 括号生成 ===")
    print(solution.generateParenthesis(3))

    # 测试全排列
    print("\n=== 全排列 ===")
    print(solution.permute([1, 2, 3]))

    # 测试N皇后
    print("\n=== N皇后 ===")
    print(solution.solveNQueens(4))
