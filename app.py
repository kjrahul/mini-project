from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np
app=Flask(__name__)
model = pickle.load(open("RandomForestModel.pkl",'rb'))
car=pd.read_csv("Cleaned_car_data.csv")

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel_type = car['fuel_type'].unique()
    return render_template('index.html',companies = companies ,car_models = car_models, years = year,fuel_type = fuel_type) 

@app.route('/prediction_result', methods=['POST'])
def predict():
    
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kilo_driven = int(request.form.get('kilo_driven'))
    
    #print(company,car_model,year,fuel_type,kms_driven)
    if year <= 2009:
        alert_message = "Alert: The car is before 2009 and is re-registered."
    else:
        alert_message = None
        
    prediction = model.predict(pd.DataFrame([[car_model ,company ,year ,kilo_driven ,fuel_type]] , columns=['name','company','year','kms_driven','fuel_type']))
    
    predictedval = str(np.round(prediction[0],2))
    return render_template('result.html',result=predictedval, alert_message=alert_message)


    
if __name__=="__main__":
    app.run(debug=True)