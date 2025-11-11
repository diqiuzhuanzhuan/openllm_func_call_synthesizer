=== 评测结果简表 ===
function_same | 总数: 913 | 正确: 890 | 错误: 23 | 准确率: 97.4808%
argument_same | 总数: 913 | 正确: 822 | 错误: 91 | 准确率: 90.0329%


=== argument评测结果分布: === 
argument_evaluation_result
exact_match         822
subset               52
no_match             18
has_extra_fields     13
partial_match         8
Name: count, dtype: int64

========== argument具体计算过程如下 ==========
数据总数 total = 913
完全一致 exact_match 数量 = 822
部分匹配 partial_match 数量 = 8
子集关系 subset 数量 = 52

完全一致 + 部分匹配 都算对：
(exact_match + partial_match) / total = 830 / 913 = 90.9091%

完全一致 + 部分匹配 + 子集关系 都算对：
(exact_match + partial_match + subset) / total = 882 / 913 = 96.6046%


