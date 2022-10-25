import predict.train_models as tm

def perform_training(stock_name, df, models_list):
    all_colors = {
                #   'SVR_linear': '#FF9EDD',
                #   'SVR_poly': '#FFFD7F',
                #   'SVR_rbf': '#FFA646',
                #   'linear_regression': '#CC2A1E',
                #   'random_forests': '#8F0099',
                #   'KNN': '#CCAB43',
                #   'elastic_net': '#CFAC43',
                #   'DT': '#85CC43',
                  'LSTM_model': '#CC7674',
                  'GRU_model': '#85CC46',
                  "LSTM_cci30_model":'#FF9EDD',
                  "GRU_cci30_model":'#FFFD7F'}

    print(df.head())
    dates, prices, ml_models_outputs, prediction_date, test_price,look_back = tm.train_predict_plot(stock_name, df, models_list)
    # 沒仔細想，但是這樣才能讓資料數量相同且對齊
    dates = dates[look_back:-look_back]
    prices = prices[look_back:-look_back]

    origdates = dates
    tolerance=200
    if len(dates) > tolerance:
        dates = dates[-tolerance:]
        prices = prices[-tolerance:]

    all_data = [(prices, 'false', stock_name, '#000000')]
    for model_output in ml_models_outputs:
        if len(origdates) > tolerance:
            all_data.append(
                (((ml_models_outputs[model_output])[0])[-tolerance:], "true", model_output, all_colors[model_output]))
        else:
            all_data.append(
                (((ml_models_outputs[model_output])[0]), "true", model_output, all_colors[model_output]))

    all_prediction_data = []
    all_test_evaluations = []
    all_prediction_data.append(("Original", test_price))
    for model_output in ml_models_outputs:
        all_prediction_data.append((model_output, (ml_models_outputs[model_output])[1]))
        all_test_evaluations.append((model_output, (ml_models_outputs[model_output])[2]))

    return all_prediction_data, all_prediction_data, prediction_date, dates, all_data, all_data, all_test_evaluations
