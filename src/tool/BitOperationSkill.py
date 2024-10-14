"""
    https://leetcode.cn/circle/discuss/CaOJ45/

    1、集合的子交并补
        A ∩ B = a & b
        A ∪ B = a | b
        A \ B = a & ∼b
        A ⊆ B = a & b = a 且 a | b = b

    2、元素和集合
         假设我们需要处理的元素是 i  目标集合是s
        全集    (1 << n) - 1
        补集   ((1 << n) - 1) ^ s
        i ∈ S  (s << i) & 1 = 1
        i ∈ S  (s << 1) & 1 = 0
        添加元素i到集合中  s | (1 << i)
        删除元素  s & ~ (1 << i)
        删除最小元素 s & (s - 1)
        统计二进制1的个数 bin(s).count('1')
"""

s = [1, 2, 4, 6]

t = 0
for i in s:
    t |= (1 << i)
t = t & (t - 1)
print(bin(t)[2:])
