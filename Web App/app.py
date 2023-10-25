from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)


model = joblib.load('../first_model.joblib')

['tenure', 'InternetService_Fiber optic', 'InternetService_No',
       'OnlineSecurity_No internet service',
       'OnlineBackup_No internet service',
       'DeviceProtection_No internet service',
       'TechSupport_No internet service', 'StreamingTV_No internet service',
       'StreamingMovies_No internet service', 'Contract_Two year',
       'PaymentMethod_Electronic check']

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST' :
        tenure = request.form['tenure']
        InternetService_Fiber = request.form['InternetService_Fiber optic']
        InternetService_No = request.form['InternetService_No']
        OnlineSecurity = request.form['OnlineSecurity_No internet service']
        OnlineBackup_No = request.form['OnlineBackup_No internet service']
        DeviceProtection_No = request.form['DeviceProtection_No internet service']
        TechSupport_No = request.form['TechSupport_No internet service']
        StreamTV = request.form['StreamingTV_No internet service']
        StreamMovies = request.form['StreamingMovies_No internet service']
        contrat = request.form['Contract_Two year']
        pay = request.form['PaymentMethod_Electronic check']
        
        data_input = [[tenure,InternetService_Fiber,InternetService_No,OnlineSecurity,OnlineBackup_No,DeviceProtection_No,
                      TechSupport_No,StreamTV, StreamMovies, contrat, pay]]
        #data_input = pd.to_numeric(data_input)
       

        
        scaler = MinMaxScaler()
        X = scaler.fit_transform(data_input)

        
        
        
        
        prediction = model.predict(X)
    
    return render_template('result.html', result = prediction)



if __name__ == '__main__':
    app.run(debug=True)