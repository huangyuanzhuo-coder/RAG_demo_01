# 原始JSON数据
import json

data = {'Characteristic': {0: '2026*', 1: '2025*', 2: '2024*', 3: '2023*', 4: '2022*', 5: '2021*', 6: '2020*', 7: '2019', 8: '2018', 9: '2017', 10: '2016'}, 'Budget_balance_to_GDP_ratio': {0: '-1.53%', 1: '-1.69%', 2: '-1.79%', 3: '-1.99%', 4: '-2.62%', 5: '-8.87%', 6: '-9.86%', 7: '0.61%', 8: '0.9%', 9: '1.06%', 10: '0.61%'}}

# 转换后的数据列表
converted_data = []

# 遍历Characteristic键值对，创建新的字典并添加到列表中
for key, year in data['Characteristic'].items():
    revenue = data['Budget_balance_to_GDP_ratio'][key]
    converted_data.append({
        "id": key + 1,  # 假设id从1开始递增
        "Characteristic": year,
        "Budget_balance_to_GDP_ratio": revenue
    })

with open("./files/ccc.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(converted_data, ensure_ascii=False, indent=4))
# 输出转换后的数据
print(converted_data)