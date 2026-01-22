from itertools import pairwise

# 1. pairwise          相邻差分 / 坡度
h = [10, 12, 9, 14, 13]
print([y - x for x, y in pairwise(h)])  # [2, -3, 5, -1]

# 2. accumulate        前缀和 / 前缀最值
from itertools import accumulate
a = [3, -4, 2, -1]
print(list(accumulate(a)))            # [3, -1, 1, 0]
print(list(accumulate(a, min)))       # [3, -4, -4, -4]

# 3. combinations      无序选 k
from itertools import combinations
nums = [-1, 0, 1, 2, -1, -4]
print({tuple(sorted([i, j, k])) for i, j, k in combinations(nums, 3) if i + j + k == 0})
# {(-1, -1, 2), (-1, 0, 1)}

# 4. permutations      全排列暴力（n≤7）
from itertools import permutations
print(max(sum(i * v for i, v in enumerate(p, 1)) for p in permutations([3, 7, 2, 9])))  # 130

# 5. product           笛卡儿积
from itertools import product
keys = [1, 4, 7]
boxes = [2, 5]
print([(k, b) for k, b in product(keys, boxes) if k < b])  # [(1, 2), (1, 5), (4, 5)]

# 6. groupby           连续相同段
from itertools import groupby
s = "aaabbcbbaa"
print(max((len(list(g)), k) for k, g in groupby(s)))  # (3, 'a')

# 7. islice            滑窗 / 取前 k
from itertools import islice, count
print(list(islice(count(10, 3), 5)))  # [10, 13, 16, 19, 22]

# 8. takewhile         条件前缀
from itertools import takewhile
a = [2, 3, 5, 4, 6]
print(list(takewhile(lambda x: x > 0, (y - x for x, y in pairwise(a)))))  # [1, 2, -1]

# 9. dropwhile         跳过前缀
from itertools import dropwhile
s = "0001230045"
print(''.join(dropwhile(lambda c: c == '0', s)))  # 1230045

# 10. chain            拍平嵌套列表
from itertools import chain
adj = [[1, 2], [], [3, 4]]
print(list(chain.from_iterable(adj)))  # [1, 2, 3, 4]

# 11. count            无限计数器
from itertools import islice, count
print([i * i for i in islice(count(1), 5)])  # [1, 4, 9, 16, 25]

# 12. repeat           固定值填充
from itertools import repeat
names = ['a', 'b']
scores = [90, 95, 88]
print(list(zip(names, scores, repeat(0))))  # [('a', 90, 0), ('b', 95, 0)]

# 13. compress         布尔掩码过滤
from itertools import compress
data = [10, 20, 30, 40, 50]
mask = [1, 0, 1, 0, 1]
print(list(compress(data, mask)))  # [10, 30, 50]

# 14. filterfalse     反向过滤
from itertools import filterfalse
s = "a1b2c3"
print(''.join(filterfalse(str.isdigit, s)))  # abc

# 15. starmap         参数解包运算
from itertools import starmap
import operator
points = [(2, 3), (4, 5)]
print(list(starmap(operator.mul, points)))  # [6, 20]