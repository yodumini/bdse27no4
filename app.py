from flask import Flask, render_template, request
from predict import utils
from predict.model import perform_training
import pandas as pd
import requests

app = Flask(__name__)

@app.route('/')
def found():
	title = "即時幣價"
	return render_template('index.html', title=title)
						   
@app.route('/index')
def index():
	title = "即時幣價"
	return render_template('index.html', title=title)

@app.route('/news')
def analytics():
	title = "幣圈新聞"
	return render_template('news.html', title=title)

# @app.route('/predict')
# def charts():
# 	title = "預測模型"
# 	return render_template('predict.html', title=title)

@app.route('/risk')
def risk():
	title = "風險分類"
	return render_template('risk.html', title=title)

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
    stock_files = list(all_files.keys())

    return render_template('predict.html', show_results="false",
                           stocklen=len(stock_files), stock_files=stock_files, len2=len([]),
                           all_prediction_data=[],
                           prediction_date="", dates=[], all_data=[], len=len([]))


@app.route('/process', methods=['POST'])
def process():
    stock_file_name = request.form['stockfile']
    ml_algoritms = request.form.getlist('mlalgos')
    df = all_files[str(stock_file_name)]
    stockname = str(stock_file_name)
    all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations = \
        perform_training(str(stock_file_name), df, ml_algoritms)
    stock_files = list(all_files.keys())

    return render_template('predict.html', all_test_evaluations=all_test_evaluations, show_results="true",
                           stocklen=len(stock_files), stock_files=stock_files,
                           len2=len(all_prediction_data),
                           all_prediction_data=all_prediction_data,
                           prediction_date=prediction_date, dates=dates, all_data=all_data, len=len(all_data),
                           stockname=stockname)

@app.route('/stockplot')
def stockplot():
    df = pd.read_csv("BNB-UST.csv")
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    r = df.values.tolist()
    return {"res":r}

@app.route('/newslist')
def newslist():
    url = "https://www.toptal.com/developers/feed2json/convert?url=https://tw.stock.yahoo.com/rss?category=intl-markets"
    r = requests.get(url)
    r = r.json()
    return r


if __name__=="__main__":
	app.run(debug=True, port=5001)