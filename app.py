import joblib
import pandas as pd
from scipy.stats import zscore
from flask import Flask, jsonify, request
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

app = Flask(__name__)

# Global variable to store the uploaded DataFrame
df_train = None
df_forecast = None

# Route to handle the index page
@app.route("/", methods=["GET"])
def index():
    return jsonify(message="Welcome to the Forecasting API! Use /forecast to get predictions. Upload a CSV file with /upload.")

# Route to handle CSV file upload
@app.route('/upload', methods=['POST'])
def upload_csvs():
    global df_train, df_forecast

    # Check for both files
    if 'train_file' not in request.files or 'forecast_file' not in request.files:
        return jsonify(error="Both 'train_file' and 'forecast_file' are required."), 400

    train_file = request.files['train_file']
    forecast_file = request.files['forecast_file']

    try:
        # Read both CSVs into pandas DataFrames
        df_train = pd.read_csv(train_file)
        df_forecast = pd.read_csv(forecast_file)

        return jsonify(
            message="Files uploaded successfully.",
            train_preview=df_train.head(3).to_dict(orient='records'),
            forecast_exog_preview=df_forecast.head(3).to_dict(orient='records')
        )
    except Exception as e:
        return jsonify(error=str(e)), 500

# Route to training and testing the model
@app.route("/train", methods=["GET"])
def forecast():
    # Load the dataset
    if df_train is None:
        return jsonify(error="No data uploaded yet"), 400
    df = df_train

    # Data preprocessing
    df['dteday'] = pd.to_datetime(df['dteday'])
    df.set_index('dteday', inplace=True)
    
    # outliners detection and correction 
    z_scores = zscore(df['cnt'])
    outliers = abs(z_scores) > 2
    df['cnt'][outliers] = df['cnt'].rolling(window=30, center=True).mean()[outliers]

    # train-test split
    test_days = 30 # Number of days to reserve for testing
    train,test = df.iloc[:-test_days], df.iloc[-test_days:]

    # Create the exogenous variables
    exog_train, exog_test = df.iloc[:-test_days,1:12], df.iloc[-test_days:,1:12]
    
    # prepare the data
    y = train['cnt']

    # model training
    # SARIMAX model configuration
    best_cfg = ((1, 0, 2), (0, 0, 0, 12))
    model_SARIMAX_best = SARIMAX(y,
                exog=exog_train,
                order=best_cfg[0],
                seasonal_order=best_cfg[1])
    model_SARIMAX_best = model_SARIMAX_best.fit(disp=False)

    # model testing
    X_test = exog_test
    y_predict = model_SARIMAX_best.forecast(steps=len(X_test), exog=X_test)
    mape = mean_absolute_percentage_error(test['cnt'], y_predict)

    # save the model
    joblib.dump(model_SARIMAX_best, 'model_SARIMAX_best.pkl')
    print(f"Modle saved, MAPE: {mape:.2f}%")

    return jsonify({
        "message": "Forecasting completed successfully.",
        "mape": mape,
        "predictions": y_predict.tolist(),
        "test_data": test['cnt'].tolist()
    })

# Route to handle forecasting
@app.route("/forecast", methods=["GET"])
def api_forecast():
    global df_forecast
    if df_forecast is None:
        return jsonify(error="No data uploaded yet"), 400
    
    # Data preprocessing
    df_forecast['dteday'] = pd.to_datetime(df_forecast['dteday'])
    df_forecast.set_index('dteday', inplace=True)
    
    # Create the exogenous variables
    exog_forecast = df_forecast.iloc[:,1:12]
    
    # Check the exogenous variables contain nan
    if exog_forecast.isnull().values.any():
        return jsonify(error="Exogenous variables contain NaN values. Please clean the data."), 400
    
    # Load the saved model
    model_SARIMAX_best = joblib.load("model_SARIMAX_best.pkl")
    print("Model loaded successfully.")

    # Forecasting
    print("Starting forecasting...")
    if model_SARIMAX_best is None:
        return jsonify(error="Model not trained yet. Please train the model first."), 400
    
    y_forecast = model_SARIMAX_best.forecast(steps=len(exog_forecast), exog=exog_forecast)
    print("Forecasting completed.")

    # Save the forecasted values
    y_forecast.index = df_forecast.index
    df_forecast['cnt'] = y_forecast
    df_all = pd.concat([df_train, df_forecast])
    # add label 'forecast' to the forecasted data
    df_all['label'] = 'historical'
    df_all.loc[df_forecast.index, 'label'] = 'forecast'
    df_all.to_csv('forecasted_data.csv', index=True)
    print("Forecasted data saved to 'forecasted_data.csv'.")

    return jsonify({
        "message": "Forecasting completed successfully, Forecasted data saved to 'forecasted_data.csv..",
        "predictions": y_forecast.tolist()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
