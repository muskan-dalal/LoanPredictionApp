# !/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
model_ROI = pickle.load(open('model_ROI.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict_logic():
    if request.method == 'POST':
        LP_CustomerPrincipalPayments = float(request.form.get('LP_CustomerPrincipalPayments'))
        LP_CustomerPayments = float(request.form.get('LP_CustomerPayments'))
        DebtToIncomeRatio = float(request.form.get('DebtToIncomeRatio'))
        StatedMonthlyIncome = float(request.form.get('StatedMonthlyIncome'))
        LP_GrossPrincipalLoss = float(request.form.get('LP_GrossPrincipalLoss'))

        LoanOriginalAmount = float(request.form.get('LoanOriginalAmount'))
        MonthlyLoanPayment = float(request.form.get('MonthlyLoanPayment'))
    pred_name = model.predict([[LP_CustomerPrincipalPayments, LP_CustomerPayments, DebtToIncomeRatio,
                                StatedMonthlyIncome, LP_GrossPrincipalLoss, LoanOriginalAmount,
                                MonthlyLoanPayment]]).tolist()[0]
    pred_ROI = model_ROI.predict([[LP_CustomerPrincipalPayments, LP_CustomerPayments, DebtToIncomeRatio,
                                StatedMonthlyIncome, LP_GrossPrincipalLoss, LoanOriginalAmount,
                                MonthlyLoanPayment]]).tolist()[0]
    approved = "Congratulations! Your loan has been approved."
    not_approved = "Sorry, you cannot get a loan."
    result = ''
    roi_rate = 'Rate of Interest Not valid'
    if pred_name == '0':
        result = approved
        roi_rate = pred_ROI
    else:
        result = not_approved
    return render_template('predict.html', pred_name=pred_name, prediction=result, rate=roi_rate)




if __name__ == '__main__':
    app.run(debug=True)

# In[ ]:




