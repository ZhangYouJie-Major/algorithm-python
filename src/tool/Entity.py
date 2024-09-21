from typing import List


class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        s = [[0] * (n + 1) for _ in range(m + 1)]
        for i, row in enumerate(matrix):
            for j, col in enumerate(row):
                s[i + 1][j + 1] = s[i + 1][j] + s[i][j + 1] - s[i][j] + col
        self.s = s

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.s[row2 + 1][col2 + 1] - self.s[row1][col2 + 1] - self.s[row2 + 1][col1] + self.s[row1][col1]


class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates


class MagicDictionary:

    def __init__(self):
        self.dictionary = set()

    def buildDict(self, dictionary: List[str]) -> None:
        self.dictionary = set(dictionary)

    def search(self, searchWord: str) -> bool:

        for value in self.dictionary:
            ctn = 0
            if len(value) == len(searchWord):
                for index, ch in enumerate(searchWord):
                    if value[index] != ch:
                        ctn += 1
                if ctn == 1:
                    return True
        return False

    sum = 0

    def getImportance(self, employees: List['Employee'], id: int) -> int:
        employees = {e.id: e for e in employees}

        def dfs(id: int) -> int:
            employee = employees[id]
            return employee.importance + sum(dfs(sub) for sub in employee.subordinates)

        return dfs(id)


class NumArray:

    def __init__(self, nums: List[int]):
        s = [0] * (len(nums) + 1)
        for i, val in enumerate(nums):
            s[i + 1] = s[i] + val
        self.s = s

    def sumRange(self, left: int, right: int) -> int:
        return self.s[right + 1] - self.s[left]
