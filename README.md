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
研報raw data預處理：
1. 統一數字表示，中文或數字
2. 斷詞
3. 只选择特定年份，预设2016/17

輸入： 
1. 預處理後的研報raw data（實際不使用，作用為推導2和3的變量）
2. 研報的數字變量
- 研報的結論，[]內的關鍵字：增持，買入。。。等的布林變量
3. 研報的文字變量
- 關鍵字的出現數量，篩選方法
	- 斷詞後進行n gram排列(n=2,3,4,5)---不做，已經去掉停用詞，斷詞某種程度上取代ngram
	- 對每個n gram進行出現頻率統計，若在少於5%的文章中出現則刪除
	- 用wcp／gini coefficient進行feature selection
4. 客觀的數字變量（不一定有幫助）
- 當日的最高最低開盤收盤價
- 當日的chg值
- 當日的RSI,KDJ值等等

輸出：
參考重慶大學論文 3.2 預測目標定義 ，令k為10，也就是說每篇研報對應一個算數收益，而算數收益的幅度我們劃分為5大類，大漲(>10%)，小漲(>5%)，持平(<5%,>-5%)，小跌，大跌，以此類推。

工作流程：
1. 預處理rawdata --OK
2. 計算研報對應漲跌類別並與研報配對成tuple --OK
3. 計算研報數字變量，加入tuple --OK
4. 對所有（不分公司）的研報進行關鍵字篩選，假設股票預測關鍵字與公司類別無關-OK
5. 計算研報的文字變量，加入tuple-ok
6. 計算客觀數字變量，加入tuple --OK
7. 對feature進行一般化 
8. 對數據集進行訓練，模型包括svm，神經網路，bayes
9. 驗證分類效果by F1 measure(for binary), micro-average for multiclass
