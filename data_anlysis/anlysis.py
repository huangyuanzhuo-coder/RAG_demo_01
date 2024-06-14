import json

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 加载数据
with open("./files/bbb.json", "r", encoding="utf-8") as f:
    json_data = json.loads(f.read())
    df = pd.json_normalize(json_data)
df['Date'] = pd.to_datetime(df['Year'])  # 将日期列转换为datetime类型

# 设置日期为索引
df.set_index('Date', inplace=True)

# 准备线性回归分析
X = np.array(df.index.map(lambda x: x.toordinal())).reshape(-1, 1)
y = df['Total_revenue_from_U.S._Government_in_million_U.S._dollars'].values

# 创建线性回归模型
model = LinearRegression()
model.fit(X, y)

# 预测趋势
trend = model.predict(X)

# 生成趋势分析总结
trend_summary = f"""
趋势分析总结：

- 总体趋势：过去的数据显示，随着时间的推移，'Value'的值呈现出{'上升趋势' if model.coef_ > 0 else '下降趋势'}。
- 回归方程：线性回归方程为 y = {model.intercept_:.2f} + {model.coef_:.2f} * x，其中 x 是日期的序号。
- 趋势强度：斜率为 {model.coef_:.2f}，表示每天'Value'的平均变化量。
- 预测：根据当前趋势，未来'Value'的值可能会继续{'上升' if model.coef_ > 0 else '下降'}。

请注意，这只是一个简单的线性趋势分析，实际情况可能更加复杂，需要考虑季节性、周期性和其他外部因素。
"""

print(trend_summary)
