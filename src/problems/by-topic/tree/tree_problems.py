"""
树 (Tree) 题目集合

包含所有树相关的题目
"""

from typing import List, Optional
from collections import deque


class TreeNode:
    """二叉树节点定义"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class ListNode:
    """链表节点定义"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    """树题目合集"""

    # ==================== 二叉树遍历 ====================

    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        144. 二叉树的前序遍历
        """
        if not root:
            return []

        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        94. 二叉树的中序遍历
        """
        if not root:
            return []

        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)

    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        """
        145. 二叉树的后序遍历
        """
        if not root:
            return []

        return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        """
        102. 二叉树的层序遍历
        BFS
        """
        if not root:
            return []

        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            current_level = []

            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(current_level)

        return result

    # ==================== 二叉树属性 ====================

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        """
        104. 二叉树的最大深度
        """
        if not root:
            return 0

        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    def minDepth(self, root: Optional[TreeNode]) -> int:
        """
        111. 二叉树的最小深度
        """
        if not root:
            return 0

        if not root.left:
            return 1 + self.minDepth(root.right)
        if not root.right:
            return 1 + self.minDepth(root.left)

        return 1 + min(self.minDepth(root.left), self.minDepth(root.right))

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        """
        110. 平衡二叉树
        """
        def check(node):
            if not node:
                return 0

            left = check(node.left)
            if left == -1:
                return -1

            right = check(node.right)
            if right == -1:
                return -1

            if abs(left - right) > 1:
                return -1

            return max(left, right) + 1

        return check(root) != -1

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        """
        100. 相同的树
        """
        if not p and not q:
            return True

        if not p or not q:
            return False

        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        """
        101. 对称二叉树
        """
        def check(left, right):
            if not left and not right:
                return True

            if not left or not right:
                return False

            return left.val == right.val and check(left.left, right.right) and check(left.right, right.left)

        return check(root.left, root.right) if root else True

    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        """
        112. 路径总和
        """
        if not root:
            return False

        if not root.left and not root.right:
            return root.val == targetSum

        return self.hasPathSum(root.left, targetSum - root.val) or self.hasPathSum(root.right, targetSum - root.val)

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """
        113. 路径总和 II
        """
        result = []
        path = []

        def dfs(node, remaining):
            if not node:
                return

            path.append(node.val)
            remaining -= node.val

            if not node.left and not node.right and remaining == 0:
                result.append(path[:])

            dfs(node.left, remaining)
            dfs(node.right, remaining)

            path.pop()

        dfs(root, targetSum)
        return result

    # ==================== 二叉搜索树 ====================

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        """
        98. 验证二叉搜索树
        """
        def validate(node, min_val, max_val):
            if not node:
                return True

            if node.val <= min_val or node.val >= max_val:
                return False

            return validate(node.left, min_val, node.val) and validate(node.right, node.val, max_val)

        return validate(root, float('-inf'), float('inf'))

    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        """
        700. 二叉搜索树中的搜索
        """
        if not root:
            return None

        if root.val == val:
            return root

        return self.searchBST(root.left, val) if val < root.val else self.searchBST(root.right, val)

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        236. 最近公共祖先
        """
        if not root or root == p or root == q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root

        return left if left else right

    # ==================== 二叉树修改 ====================

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        226. 翻转二叉树
        """
        if not root:
            return None

        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)

        return root

    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """
        2181. 合并零之间的节点
        链表操作
        """
        tail = head
        cur = head.next

        while cur.next:
            if cur.val:
                tail.val += cur.val
            else:
                tail = tail.next
                tail.val = 0
            cur = cur.next

        tail.next = None
        return head

    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        143. 重排链表
        """
        # 利用快慢指针找到中间节点
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        mid = slow

        # 翻转后半部分
        pre = None
        cur = mid
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        head2 = pre

        # 合并两个链表
        while head2.next:
            nxt = head.next
            nxt2 = head2.next
            head.next = head2
            head2.next = nxt
            head = nxt
            head2 = nxt2

    def maxProduct(self, root: Optional[TreeNode]) -> int:
        """
        1339. 分裂二叉树的最大乘积
        """
        subtree_sums = []

        def dfs(node):
            if not node:
                return 0

            total = node.val + dfs(node.left) + dfs(node.right)
            subtree_sums.append(total)
            return total

        total_sum = dfs(root)
        max_prod = max(total_sum * (total_sum - s) for s in subtree_sums)

        return max_prod % 1_000_000_007

    def maxDotProduct(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> int:
        """
        """
        # 如果是树的最大点积，需要修改实现
        pass

    def subtreeWithAllDeepest(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        865. 具有所有最深节点的最小子树
        """
        max_depth = -1
        ans = None

        def dfs(node, depth):
            nonlocal ans, max_depth

            if node is None:
                max_depth = max(depth, max_depth)
                return depth

            left_depth = dfs(node.left, depth + 1)
            right_depth = dfs(node.right, depth + 1)

            if max_depth == left_depth == right_depth:
                ans = node

            return max(left_depth, right_depth)

        dfs(root, 0)
        return ans

    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        """
        968. 监控二叉树
        树形DP
        """
        def dfs(node):
            if node is None:
                return float('inf'), 0, 0

            l_choose, l_by_father, l_by_children = dfs(node.left)
            r_choose, r_by_father, r_by_children = dfs(node.right)

            choose = min(l_choose, l_by_father, l_by_children) + min(r_choose, r_by_father, r_by_children) + 1
            choose_by_father = min(l_choose, l_by_children) + min(r_choose, r_by_children)
            choose_by_children = min(l_choose + r_by_children, r_choose + l_by_children, l_choose + r_choose)

            return choose, choose_by_father, choose_by_children

        root_choose, _, root_children = dfs(root)
        return min(root_choose, root_children)

    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        """
        257. 二叉树的所有路径
        """
        result = []

        def dfs(node, path):
            if node is None:
                return

            path += str(node.val)

            if not node.left and not node.right:
                result.append(path)
            else:
                path += '->'
                dfs(node.left, path)
                dfs(node.right, path)

        dfs(root, '')
        return result


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 构建测试树
    #     3
    #    / \
    #   9  20
    #     /  \
    #    15   7
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(15)
    root.right.right = TreeNode(7)

    # 测试最大深度
    print("=== 最大深度 ===")
    print(solution.maxDepth(root))

    # 测试层序遍历
    print("\n=== 层序遍历 ===")
    print(solution.levelOrder(root))
