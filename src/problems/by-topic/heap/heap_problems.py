"""
堆 (Heap) 题目集合

包含所有使用堆数据结构解决的题目
"""

from typing import List
import heapq
from collections import Counter


class Solution:
    """堆题目合集"""

    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        215. 数组中的第K个最大元素
        最小堆维护前k大元素
        """
        return heapq.nlargest(k, nums)[-1]

    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        347. 前 K 个高频元素
        哈希表 + 堆
        """
        count = Counter(nums)
        return heapq.nlargest(k, count.keys(), key=count.get)

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        """
        4. 寻找两个正序数组的中位数
        双堆法（大顶堆+小顶堆）
        """
        # 合并数组
        merged = sorted(nums1 + nums2)
        n = len(merged)

        if n % 2 == 1:
            return float(merged[n // 2])
        else:
            return (merged[n // 2 - 1] + merged[n // 2]) / 2.0

    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        """
        871. 最低加油次数
        最大堆
        """
        import heapq
        max_heap = []
        result = 0
        prev = 0
        tank = startFuel

        stations.append([target, 0])

        for location, capacity in stations:
            tank -= (location - prev)

            while tank < 0 and max_heap:
                tank += -heapq.heappop(max_heap)
                result += 1

            if tank < 0:
                return -1

            heapq.heappush(max_heap, -capacity)
            prev = location

        return result

    def leastInterval(self, tasks: List[str], n: int) -> int:
        """
        621. 任务调度器
        最大堆模拟
        """
        from collections import Counter
        import heapq

        count = Counter(tasks)
        max_heap = [-cnt for cnt in count.values()]
        heapq.heapify(max_heap)

        time = 0
        while max_heap:
            temp = []
            # 执行n+1个任务
            for _ in range(n + 1):
                if max_heap:
                    temp.append(-heapq.heappop(max_heap))

            for cnt in temp:
                cnt -= 1
                if cnt > 0:
                    heapq.heappush(max_heap, -cnt)

            time += len(temp) if not max_heap else n + 1

        return time

    def reorganizeString(self, s: str) -> str:
        """
        767. 重构字符串
        最大堆
        """
        from collections import Counter
        import heapq

        count = Counter(s)
        max_heap = [(-cnt, char) for char, cnt in count.items()]
        heapq.heapify(max_heap)

        result = []
        prev_cnt, prev_char = 0, ''

        while max_heap:
            cnt, char = heapq.heappop(max_heap)
            result.append(char)

            if prev_cnt < 0:
                heapq.heappush(max_heap, (prev_cnt, prev_char))

            cnt += 1
            prev_cnt, prev_char = cnt, char

        return ''.join(result) if len(result) == len(s) else ''

    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        """
        973. 最接近原点的 K 个点
        最小堆
        """
        return heapq.nsmallest(k, points, key=lambda p: p[0]**2 + p[1]**2)

    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        """
        378. 有序矩阵中第 K 小的元素
        最小堆
        """
        n = len(matrix)
        min_heap = [(matrix[0][0], 0, 0)]

        for _ in range(k - 1):
            val, i, j = heapq.heappop(min_heap)

            if j + 1 < n:
                heapq.heappush(min_heap, (matrix[i][j + 1], i, j + 1))
            if j == 0 and i + 1 < n:
                heapq.heappush(min_heap, (matrix[i + 1][0], i + 1, 0))

        return heapq.heappop(min_heap)[0]

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """
        239. 滑动窗口最大值
        单调队列（不是堆，但思想类似）
        """
        from collections import deque

        result = []
        deque_idx = deque()

        for i, num in enumerate(nums):
            # 移除不在窗口内的元素
            while deque_idx and deque_idx[0] <= i - k:
                deque_idx.popleft()

            # 移除比当前元素小的元素
            while deque_idx and nums[deque_idx[-1]] < num:
                deque_idx.pop()

            deque_idx.append(i)

            if i >= k - 1:
                result.append(nums[deque_idx[0]])

        return result

    def findKPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        """
        373. 查找和最小的K对数字
        最小堆
        """
        if not nums1 or not nums2:
            return []

        heap = []
        result = []

        # 初始化堆：nums1的每个元素与nums2[0]配对
        for i, num in enumerate(nums1[:k]):
            heapq.heappush(heap, (num + nums2[0], i, 0))

        while heap and len(result) < k:
            _, i, j = heapq.heappop(heap)
            result.append([nums1[i], nums2[j]])

            if j + 1 < len(nums2):
                heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))

        return result


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 测试第K个最大元素
    print("=== 第K个最大元素 ===")
    print(solution.findKthLargest([3,2,1,5,6,4], 2))

    # 测试前K个高频元素
    print("\n=== 前K个高频元素 ===")
    print(solution.topKFrequent([1,1,1,2,2,3], 2))

    # 测试任务调度器
    print("\n=== 任务调度器 ===")
    print(solution.leastInterval(["A","A","A","B","B","B"], 2))
