#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('autosave', '0')



import requests



url='http://localhost:9696/predict'


customer_id='xyz'
customer={
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "yes",
    "phoneservice": "yes",
    "multiplelines": "yes",
    "internetservice": "fiber_optic",
    "onlinebackup": "no",
    "deviceprotection": "yes",
    "techsupport": "no",
    "streamingtv": "yes",
    "streamingmovies": "yes",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 104.2,
    "totalcharges": 1743.5  
 }


response=requests.post(url,json=customer).json()


if response['churn']==True:
    print('sending promo')
else:
    print('Not')





