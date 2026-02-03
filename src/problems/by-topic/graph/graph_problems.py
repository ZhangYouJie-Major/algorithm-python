"""
图论 (Graph) 题目集合

包含所有图论相关的题目
"""

from typing import List
import heapq
from collections import deque


class Solution:
    """图论题目合集"""

    # ==================== 并查集 ====================

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        """
        547. 省份数量
        并查集/DFS
        """
        n = len(isConnected)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if isConnected[i][j]:
                    union(i, j)

        return sum(1 for i in range(n) if find(i) == i)

    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        """
        1971. 寻找图中是否存在路径
        BFS/DFS/并查集
        """
        # 构建邻接表
        graph = [[] for _ in range(n)]
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        # BFS
        visited = [False] * n
        queue = deque([source])
        visited[source] = True

        while queue:
            node = queue.popleft()
            if node == destination:
                return True

            for neighbor in graph[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        return False

    # ==================== 拓扑排序 ====================

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        """
        207. 课程表
        拓扑排序（检测环）
        """
        # 构建图和入度数组
        graph = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses

        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1

        # BFS拓扑排序
        queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
        completed = 0

        while queue:
            course = queue.popleft()
            completed += 1

            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)

        return completed == numCourses

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """
        210. 课程表 II
        返回拓扑排序结果
        """
        # 构建图和入度数组
        graph = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses

        for course, prereq in prerequisites:
            graph[prereq].append(course)
            in_degree[course] += 1

        # BFS拓扑排序
        queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
        result = []

        while queue:
            course = queue.popleft()
            result.append(course)

            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)

        return result if len(result) == numCourses else []

    # ==================== 最短路径 ====================

    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        """
        743. 网络延迟时间
        Dijkstra算法
        """
        # 构建邻接表
        graph = [[] for _ in range(n + 1)]
        for u, v, w in times:
            graph[u].append((v, w))

        # Dijkstra
        dist = [float('inf')] * (n + 1)
        dist[k] = 0
        heap = [(0, k)]

        while heap:
            d, node = heapq.heappop(heap)

            if d > dist[node]:
                continue

            for neighbor, weight in graph[node]:
                new_dist = d + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))

        max_dist = max(dist[1:])
        return max_dist if max_dist < float('inf') else -1

    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        """
        787. K站中转内最便宜的航班
        Bellman-Ford（限制边数）
        """
        INF = float('inf')
        dist = [INF] * n
        dist[src] = 0

        # 最多k+1条边（k个中转）
        for _ in range(k + 1):
            new_dist = dist[:]
            updated = False

            for u, v, price in flights:
                if dist[u] + price < new_dist[v]:
                    new_dist[v] = dist[u] + price
                    updated = True

            dist = new_dist
            if not updated:
                break

        return dist[dst] if dist[dst] != INF else -1

    # ==================== 最小生成树 ====================

    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        """
        1584. 连接所有点的最小费用
        Prim算法/Kruskal算法
        """
        n = len(points)

        # 计算曼哈顿距离
        def distance(i, j):
            return abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])

        # Prim算法
        visited = [False] * n
        min_dist = [float('inf')] * n
        min_dist[0] = 0

        for _ in range(n):
            # 找未访问的最小距离点
            u = -1
            for i in range(n):
                if not visited[i] and (u == -1 or min_dist[i] < min_dist[u]):
                    u = i

            visited[u] = True

            # 更新距离
            for v in range(n):
                if not visited[v]:
                    d = distance(u, v)
                    if d < min_dist[v]:
                        min_dist[v] = d

        return sum(min_dist)

    # ==================== DFS/BFS ====================

    def numIslands(self, grid: List[List[str]]) -> int:
        """
        200. 岛屿数量
        DFS/BFS
        """
        if not grid or not grid[0]:
            return 0

        m, n = len(grid), len(grid[0])
        count = 0

        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
                return

            grid[i][j] = '0'
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    count += 1
                    dfs(i, j)

        return count

    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        """
        695. 岛屿的最大面积
        DFS
        """
        if not grid or not grid[0]:
            return 0

        m, n = len(grid), len(grid[0])
        max_area = 0

        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != 1:
                return 0

            grid[i][j] = 0
            area = 1
            area += dfs(i + 1, j)
            area += dfs(i - 1, j)
            area += dfs(i, j + 1)
            area += dfs(i, j - 1)

            return area

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    max_area = max(max_area, dfs(i, j))

        return max_area

    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        """
        1334. 阈值距离内邻居最少的城市
        Floyd算法
        """
        import math

        # 初始化距离矩阵
        dist = [[math.inf] * n for _ in range(n)]

        for i in range(n):
            dist[i][i] = 0

        for x, y, edge in edges:
            dist[x][y] = edge
            dist[y][x] = edge

        # Floyd算法
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

        # 找邻居最少的城市
        min_cities = n
        result = -1

        for i in range(n):
            cities_within_threshold = sum(1 for j in range(n) if i != j and dist[i][j] <= distanceThreshold)

            if cities_within_threshold <= min_cities:
                min_cities = cities_within_threshold
                result = i

        return result

    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        """
        2976. 转换字符串的最小成本
        Floyd算法
        """
        import math

        # 初始化距离矩阵
        dis = [[math.inf] * 26 for _ in range(26)]
        for i in range(26):
            dis[i][i] = 0

        for x, y, c in zip(original, changed, cost):
            i = ord(x) - ord('a')
            j = ord(y) - ord('a')
            dis[i][j] = min(dis[i][j], c)

        # Floyd算法
        for k in range(26):
            for i in range(26):
                if dis[i][k] == math.inf:
                    continue
                for j in range(26):
                    dis[i][j] = min(dis[i][j], dis[i][k] + dis[k][j])

        # 计算总成本
        ans = sum(dis[ord(x) - ord('a')][ord(y) - ord('a')] for x, y in zip(source, target))

        return ans if ans < math.inf else -1

    def minCost(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        """
        2512. 奖励最顶尖的 K 名学生
        DP + 限制时间
        """
        import math
        n = len(passingFees)
        f = [[math.inf] * n for _ in range(maxTime + 1)]
        f[0][0] = passingFees[0]

        for i in range(1, maxTime + 1):
            for start, end, time in edges:
                if i - time >= 0:
                    f[i][start] = min(f[i][start], f[i - time][end] + passingFees[start])
                    f[i][end] = min(f[i][end], f[i - time][start] + passingFees[end])

        ans = min(f[i][n - 1] for i in range(maxTime + 1))

        return ans if ans < math.inf else -1


# 测试代码
if __name__ == "__main__":
    solution = Solution()

    # 测试省份数量
    print("=== 省份数量 ===")
    print(solution.findCircleNum([[1,1,0],[1,1,0],[0,0,1]]))

    # 测试岛屿数量
    print("\n=== 岛屿数量 ===")
    grid = [
        ["1","1","0","0","0"],
        ["1","1","0","0","0"],
        ["0","0","1","0","0"],
        ["0","0","0","1","1"]
    ]
    print(solution.numIslands(grid))
