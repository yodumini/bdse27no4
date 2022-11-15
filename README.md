## 資展國際巨量資料分析班 BDSE27-4
### [unicoin虛擬貨幣分析平台](https://unicoin.ga)

[詳細介紹](https://github.com/yodumini/bdse27no4/blob/main/doc/%E7%AC%AC%E5%9B%9B%E7%B5%84%20-%20%E6%9C%9F%E6%9C%AB%E5%B0%88%E9%A1%8C%E5%A0%B1%E5%91%8A_%E6%9C%80%E7%B5%82%E7%89%88.pptx.pdf)

### 動機
近年來虛擬貨幣市場掀起投資熱潮，自比特幣推出以來話題與爭議不斷，還有劇烈的價格波動，更是讓許多投資者既期待又怕受傷害。
為了更瞭解這項新崛起的金融商品，我們想利用專題的機會，研究虛擬貨幣的投資趨勢及風險

### 研究流程
- 環境建置：linux gcp建立Hadoop spark叢集計算效能、mysql
- 網路資料爬蟲：python資料收集、鉅亨網新聞爬蟲
- 風險分類模型：sklearn,kmeans
- 幣價預測模型：tensorflowm,lstm,gru
- 視覺化平台：flask,hmtl,css,js,bootstrap,highcharts

### 本專題視覺化平台用python flask框架
```
├── app.py # 主要執行程式
├── code # 專題研究過程程式碼
│   ├── deepLearningPractice
│   ├── newscrawl
│   └── risk-kmeans
├── data # 即時幣價kline圖數據源
│   ├── 1INCHUSDT.csv # 交易對的歷史幣價數據
│   └── 聚類结果.csv # 交易對風險分類結果
├── doc
│   ├── 第四組 - 期末專題報告_最終版.pptx.pdf
│   └── gcp部署網站流程.txt
├── predict # 預測幣價頁面所使用的數據源與模型程式碼
│   ├── bdse27_GRU_model_cci30.h5
│   ├── bdse27_lstm_model_cci30.h5
│   ├── data
│   ├── individual_stocks_5yr
│   ├── model.py
│   ├── scaler_cci30_5.save
│   ├── train_models.py
│   └── utils.py
├── requirements.txt # 所使用的packages
├── runtime.txt # 執行環境
├── static # 網頁模板使用素材
│   ├── css
│   ├── dbtheme
│   ├── img
│   ├── js
│   ├── mono-main
│   ├── scss
│   └── vendor
└── templates # html版面
    ├── _demo.html
    ├── _temp.html
    ├── index.html
    ├── news.html
    ├── news_stat.html
    ├── parts # 基礎模板被繼承用
    ├── predict.html
    ├── risk.html
    └── team.html
```

> 本研究成果僅供資展國際專題報告使用，如有侵權請來信告知
> sherwin8671@gmail.com
