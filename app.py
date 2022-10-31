from flask import Flask, render_template, request
from predict import utils
from predict.model import perform_training
import pandas as pd
import requests
# from urllib.parse import quote
# import feedparser
import os
import json
import multiprocessing as mp
from sqlalchemy import create_engine

app = Flask(__name__)

#連接資料庫
# username = 'sherwin'     # 資料庫帳號
# password = '123456'     # 資料庫密碼
# host = '172.22.35.211'    # 資料庫位址
username = 'bbdbe8ea88eccc'     # 資料庫帳號
password = '40cda128'     # 資料庫密碼
host = 'us-cdbr-east-06.cleardb.net'    # 資料庫位址
port = '3306'         # 資料庫埠號
# database = 'news'   # 資料庫名稱
database = 'heroku_052dd73d4bca4f8'   # 資料庫名稱
table = 'news1'   # 表格名稱

# 建立資料庫引擎
engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}')
# 建立資料庫連線
connection  = engine.connect()


# @app.route('/')
# def found():
#     title = "即時幣價"
#     return render_template('index.html', title=title)
#
# @app.route('/index')
# def index():
#     title = "即時幣價"
#     return render_template('index.html', title=title)

columns = ['s', 'o', 'h', 'l', 'v', 'qv']


def job(x):
    query_url = f'https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-product-by-symbol?symbol={x}'
    response = requests.get(query_url)
    data = json.loads(response.text)['data']
    df = pd.DataFrame(data, columns=columns, index=[0])
    return df

@app.route('/')
@app.route('/index')
def found():
    title = "即時幣價"
    stock_list = [file[:-4] for file in os.listdir("predict/data")]
    stock_list.sort()

    # df_list = []
    # for i in stock_list:
    #     query_url = f'https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-product-by-symbol?symbol={i}'
    #     response = requests.get(query_url)
    #     data = json.loads(response.text)
    #     if data['data']:
    #         df = pd.DataFrame(data['data'], columns=columns, index=[0])
    #         df_list.append(df)

    pool = mp.Pool(processes=3)
    df_list = pool.map(job, stock_list)
    df_list = pd.concat(df_list)
    df_list = df_list.dropna()
    df_list = df_list.sort_values('s')

    df_list2 = []
    for i in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT']:
        query_url = f'https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-product-by-symbol?symbol={i}'
        response = requests.get(query_url)
        data = json.loads(response.text)['data']
        if data:
            df_list2.append({
                'name': data['an'],
                'price': data['o'],
            })

    return render_template('index.html', title=title, crypto_name=stock_list, output=df_list, df_list=df_list2)

# @app.route('/index')
# def index():
#     title = "即時幣價"
#     stock_list = []
#     for file in dirs:
#         stock_list.append((file)[:-4])
#     stock_list.sort()
#
#     df_list = []
#     for i in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT']:
#         query_url = f'https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-product-by-symbol?symbol={i}'
#         response = requests.get(query_url)
#         data = json.loads(response.text)['data']
#         if data:
#             df_list.append({
#                 'name': data['an'],
#                 'price': data['o'],
#             })
#     return render_template('index.html', title=title, crypto_name=stock_list, df_list=df_list)

@app.route('/news', methods=['GET', 'POST'])
def analytics():
    title = "幣圈新聞"
    num = ['06','07','08','09','10']
    opt=len(num)
    if request.method == "GET":
        sql=' SELECT date, label, title, content, url FROM news1 '
        df = pd.read_sql_query(sql, engine, coerce_float=True)
        connection.close()
    else:
        num = request.form[ 'send' ]
        sql = f"SELECT date, label, title, content, url FROM news1 WHERE MONTH(date) = '{num}'"
        df = pd.read_sql_query(sql, engine, coerce_float=True)
        connection.close()
        num = ['06','07','08','09','10']
    return render_template( 'news.html',title=title,outputs=df,num=num,opt=opt )


# @app.route('/newslist', methods=['POST'])
# def newslist():
#     title = "幣圈新聞"
#     url = "https://tw.stock.yahoo.com/rss?q="
#     text = request.form['url']
#     url = url+quote(text)
#     r = feedparser.parse(url)['entries']
#     return render_template('news.html', res=r, title=title)


# @app.route('/predict')
# def charts():
#   title = "預測模型"
#   return render_template('predict.html', title=title)

@app.route('/test_post/nn') #路由
def test_post():
    csv_read_file = pd.read_csv("data/聚類结果.csv", encoding="utf-8")
    csv_read_data = csv_read_file.values.tolist()
    return {"imgdata": csv_read_data}


@app.route('/risk')
def risk():
    title = "風險分類"
    csv_read_file = pd.read_csv("data/聚類结果.csv", encoding="utf-8")
    csv_read_data = csv_read_file.values.tolist()
    extent = len(csv_read_data)
    return render_template('risk.html', df=csv_read_data, extent=extent, title=title)

@app.route('/team')
def team():
    title = "團隊介紹"
    return render_template('team.html', title=title)

@app.route('/_demo')
def demo():
    title = "測試頁面"
    return render_template('_demo.html', title=title)

all_files = utils.read_all_stock_files('predict/individual_stocks_5yr')



@app.route('/predict')
def landing_function():
    title = "預測模型"
    stock_files = list(all_files.keys())
    stock_files.sort()

    return render_template('predict.html', show_results="false", title=title,
                           stocklen=len(stock_files), stock_files=stock_files, len2=len([]),
                           all_prediction_data=[],
                           prediction_date="", dates=[], all_data=[], len=len([]))


@app.route('/process', methods=['POST'])
def process():
    title="模型預測結果"
    stock_file_name = request.form['stockfile']
    ml_algoritms = request.form.getlist('mlalgos')
    df = all_files[str(stock_file_name)]
    stockname = str(stock_file_name)
    all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations = \
        perform_training(str(stock_file_name), df, ml_algoritms)
    stock_files = list(all_files.keys())
    stock_files.sort()

    return render_template('predict.html', all_test_evaluations=all_test_evaluations, show_results="true",title=title,
                           stocklen=len(stock_files), stock_files=stock_files,
                           len2=len(all_prediction_data),
                           all_prediction_data=all_prediction_data,
                           prediction_date=prediction_date, dates=dates, all_data=all_data, len=len(all_data),
                           stockname=stockname)

# @app.route('/stockplot')
# def stockplot():
#     df = pd.read_csv("./data/BTCUSDT.csv")
#     df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
#     r = df.values.tolist()
#     return {"res": r}

@app.route('/stockplot',methods=["GET","POST"])
def stockplot():
    crypto = request.values
    if not crypto:
        crypto = "1INCHUSDT"
    else:
        crypto = request.values["crypto"]
    df = pd.read_csv(f"data/{crypto}.csv")
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    r = df.values.tolist()
    return {"res": r}

@app.route('/coinprice',methods=["GET"])
def coinprice():
    df_list = []
    for i in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT']:
        query_url = f'https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-product-by-symbol?symbol={i}'
        response = requests.get(query_url)
        data = json.loads(response.text)['data']
        if data:
            df_list.append({
                'name': data['an'],
                'price': data['o'],
            })
    return {"res": df_list}

if __name__=="__main__":
    # app.run(debug=True, port=5001)

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)