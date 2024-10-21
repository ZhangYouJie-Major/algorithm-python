from itertools import pairwise, count, permutations, combinations
from collections import Counter, defaultdict

'''
    1、无限迭代器
        itertools.count(start, step)
        itertools.count(start, step)函数用于生成一个无限迭代的整数序列。它从 start 开始，并以 step 作为步长不断生成整数。

     2、组合生成器   
    
    itertools.permutations(iterable, r)
    itertools.permutations(iterable, r) 函数用于生成可迭代对象中所有长度为 r 的排列。


    itertools.combinations(iterable, r)
    itertools.combinations(iterable, r) 函数用于生成可迭代对象中所有长度为 r 的组合，不考虑元素顺序。
'''

'''
    python


'''

# 从 5 开始，步长为 2，生成整数序列
for i in count(5, 2):
    if i > 20:
        break
    print(i, end=' ')

# 定义一个列表
letters = ['a', 'b', 'c']

# 使用 permutations 生成排列
for perm in permutations(letters, 3):
    print(perm)

print('---------------')

# 使用 permutations 生成排列
for perm in combinations(letters, 3):
    print(perm)

a = [1, 2, 3, 4, 5]
for x, y in pairwise(a):
    print(x)
    print(y)
