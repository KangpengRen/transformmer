import json  # 导入json模块，用于处理JSON格式的数据

if __name__ == "__main__":  # 主程序入口，当脚本被直接执行时运行
    files = ["dev.json", "test.json", "train.json"]
    # 中文和英文文本制作对应的分词器和词表
    en_path = "corpus.en"
    ch_path = "corpus.ch"
    en_lines = []
    ch_lines = []

    for file in files:
        corpus = json.load(open("./json/" + file, "r", encoding="utf-8"))
        for item in corpus:
            en_lines.append(item[0] + "\n")
            ch_lines.append(item[1] + "\n")

    # 将英文句子写入文件
    with open(en_path, "w", encoding="utf-8") as f:
        f.writelines(en_lines)

    # 将中文句子写入文件
    with open(ch_path, "w", encoding="utf-8") as f:
        f.writelines(ch_lines)

    # 输出中文句子行数：
    print("lines of Chinese: ", len(ch_lines))
    # 输出英文句子行数：
    print("lines of English: ", len(en_lines))
    # 输出完成提示信息
    print("-------- Get Corpus ! --------")