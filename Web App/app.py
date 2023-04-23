from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)


model = joblib.load('../my_model.joblib')



@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST' :
        tenure = request.form['tenure']
        MonthlyCharges = request.form['MonthlyCharges']
        TotalCharges = request.form['TotalCharges']
        Fiber_optic = request.form['Fiber optic']
        StreamingTV = request.form['StreamingTV internet service']
        StreamingMovies = request.form['StreamingMovies internet service']
        Contract_Two_year = request.form['Contract_Two year']
        PaymentMethod_Electronic_check = request.form['PaymentMethod_Electronic_check']
        data_input = [tenure,MonthlyCharges,TotalCharges,Fiber_optic,StreamingTV,StreamingMovies,Contract_Two_year,PaymentMethod_Electronic_check]
        data_input = pd.to_numeric(data_input)

        prediction = model.predict([data_input])
    
    return render_template('result.html', result = prediction)



print("hello")

if __name__ == '__main__':
    app.run(debug=True)