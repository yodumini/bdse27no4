## 資展國際巨量資料分析班 BDSE27-4
### unicoin虛擬貨幣分析平台
本專題視覺化平台用python flask框架
```
├── app.py # 主要執行程式
├── code # 專題研究過程程式碼
│   ├── deepLearningPractice
│   └── newscrawl
├── data # 即時幣價kline圖數據源
│   ├── 1INCHUSDT.csv # 交易對的歷史幣價數據
│   └── 聚類结果.csv # 交易對風險分類結果
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
