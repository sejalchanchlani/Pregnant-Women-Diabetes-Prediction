# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 01:53:56 2020

@author: asus
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Pregnancies':1, 'Glucose':2, 'BloodPressure':4, 'Insulin':5,'BMI':6,'DiabetesPedigreeFunction':7, 'Age':8 })

print(r.json())