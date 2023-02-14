import pickle

import jsonify
import numpy as np
import requests
import sklearn
from flask import Flask, jsonify, render_template, request

model_file='model-C=10.bin'
app = Flask(__name__)
dv,model = pickle.load(open(model_file, 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Gender_=request.form['Gender']
        seniorcitizen_=request.form['seniorcitizenship']
        partners_=request.form['partners']
        dependents_=request.form['dependents']
        phoneservice_=request.form['phoneservice']
        multipleslines_=request.form['multiplelines']
        internetservice_=request.form['internetservice']
        onlinesecurity_=request.form['onlinesecurity']
        onlinebackup_=request.form['onlinebackup']
        deviceprotection_=request.form['deviceprotection']
        techsupport_=request.form['ts']
        streamingtv_=request.form['streamingtv']
        streamingmovies_=request.form["s_movies"]
        contract_=request.form['contract']
        paperlessbilling_=request.form['paperlessbilling']
        paymentmethod_=request.form["paymentmethod"]
        tenure_=request.form['tenure']
        monthlycharges_=request.form["monthlycharges" ]
        totalcharges_=request.form['totalcharges']
        customer={
            'gender': Gender_,
        'seniorcitizen':seniorcitizen_,
        'partner': partners_,
        'dependents': dependents_,
        'phoneservice': phoneservice_,
        'multiplelines': multipleslines_,
        'internetservice': internetservice_,
        'onlinesecurity': onlinesecurity_,
        'onlinebackup': onlinebackup_,
        'deviceprotection': deviceprotection_,
        'techsupport': techsupport_,
        'streamingtv': streamingtv_,
        'streamingmovies': streamingmovies_,
        'contract': contract_,
        'paperlessbilling': paperlessbilling_,
        'paymentmethod': paymentmethod_,
        'tenure': tenure_,
        'monthlycharges': monthlycharges_,
        'totalcharges': totalcharges_}
        X=dv.transform([customer])
        output=model.predict_proba(X)[0,1]
        if output<0:
            return render_template('index.html',prediction_texts="unvalid data")
        else:
            return render_template('index.html',prediction_text="Churn Probability : {}".format(round(output*100,2)))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)