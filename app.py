from flask import Flask,render_template

app = Flask(__name__)

@app.route('/')
def found():
	title = "即時幣價"
	return render_template('index.html', title=title)
						   
@app.route('/index.html')
def index():
	title = "即時幣價"
	return render_template('index.html', title=title)

@app.route('/news.html')
def analytics():
	title = "幣圈新聞"
	return render_template('news.html', title=title)

@app.route('/predict.html')
def charts():
	title = "預測模型"
	return render_template('predict.html', title=title)

@app.route('/risk.html')
def risk():
	title = "風險分類"
	return render_template('risk.html', title=title)

@app.route('/team.html')
def team():
	title = "團隊介紹"
	return render_template('team.html', title=title)

@app.route('/_demo.html')
def demo():
	title = "測試頁面"
	return render_template('_demo.html', title=title)

if __name__=="__main__":
	app.run(debug=True, port=5001)