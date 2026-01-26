import json

path = "../data/json/test.json"  # 你的这段文件
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(json.dumps(data[:3], ensure_ascii=False, indent=2))   # 打印前三条数据
print(type(data), len(data), type(data[0]), len(data[0]))