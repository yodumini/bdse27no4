# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request
import utils
import train_models as tm

# import os
# import pandas as pd

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


all_files = utils.read_all_stock_files('individual_stocks_5yr')


# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.read_all_stock_files
@app.route('/')
def landing_function():
    stock_files = list(all_files.keys())

    return render_template('index.html', show_results="false",
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

    return render_template('index.html', all_test_evaluations=all_test_evaluations, show_results="true",
                           stocklen=len(stock_files), stock_files=stock_files,
                           len2=len(all_prediction_data),
                           all_prediction_data=all_prediction_data,
                           prediction_date=prediction_date, dates=dates, all_data=all_data, len=len(all_data),
                           stockname=stockname)


if __name__ == '__main__':
    app.run(debug=True)
