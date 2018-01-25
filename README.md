# Stock-Text-Analysis

# 更新紀錄(1/10)
1. 雖然ASR和一般收益率有細微差距（在figs中可以看到分佈），但是如果根據幅度劃分為五類的話，兩者的分佈是一樣的
2. 因為這幾家公司的種類不同，所以沒有共用tfidf的，而是針對每家公司都從頭進行操作（統計高頻詞->訓練模型）
3. 這次的訓練數據都採用了16，17年的研報，總數量也可以在figs中的分佈圖看到
4. 訓練結果可以在figs中看到，成果仍然會overfit

## code

### jiebacut.py

**输入**
- `xq_yanbao.json`：原始数据集
- `stop_words.txt`：停用词词典

**输出**
- `split_True.json`：分词结果（默认使用停用词）
- `frequency.txt`：高频词表

### tfidf.py

**输入**
- `split_True.json`：分好词的数据集
- `frequency.txt`：高频词表

**输出**
- `tfidf.txt`：300维向量数组
- `words.txt`：向量的维度对应的词

### 2fc_keras.py
- 2層神經網路

**輸入**
- 單篇研報對應的300維tfidf數組

**輸出**
- 根據指定的SAR累加範圍與閾值生成的5類：大跌／小跌／持平／小漲／大漲

### ioutils.py
- `get_training_data_json(filename)`: 从json文件中获取二元组 (x_train, y_train)
- `shuffle_data(x_train,y_train)`: 数据shuffle

### data_dump_preprocess.py
原始数据预处理并转化为json格式

### yanbao_selet.py
用于筛选近一年的研报数据

# rawdata

研报原始数据及预处理结果

- 原始数据（及筛选时间后）的json文件
- 分词结果
- 高频词
- tfidf向量
- 股票走势

# xueqiuYanbao

爬取的研报数据和分析文件

# figs

训练结果