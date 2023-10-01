#!/usr/bin/env python
# coding: utf-8

from src.flight_fare_prediction.logger import logging
from src.flight_fare_prediction.exception import CustomException
from src.flight_fare_prediction.components.data_ingestion import DataIngestion
from src.flight_fare_prediction.components.data_ingestion import DataIngestionConfig
from src.flight_fare_prediction.components.data_transformation import DataTransformationConfig
from src.flight_fare_prediction.components.data_transformation import DataTransformation
from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import sys

app = Flask(__name__)
model = pickle.load(open('model/flight_rf.pkl', 'rb'))

@app.route('/')
@cross_origin()
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method=='POST':
        
        # Date Of Journey
        date_dep = request.form['Dep_Time']
        Journey_day = pd.to_datetime(date_dep, format='%Y-%m-%dT%H:%M').day
        Journey_month = pd.to_datetime(date_dep, format='%Y-%m-%dT%H:%M').month
        
        # Departure
        Dep_hour = pd.to_datetime(date_dep, format='%Y-%m-%dT%H:%M').hour
        Dep_min = pd.to_datetime(date_dep, format='%Y-%m-%dT%H:%M').minute
        
        # Arrival
        date_arr = request.form['Arrival_Time']
        Arrival_hour = pd.to_datetime(date_arr, format='%Y-%m-%dT%H:%M').hour
        Arrival_min = pd.to_datetime(date_arr, format='%Y-%m-%dT%H:%M').minute
        
        # Duration
        dur_hour = abs(Arrival_hour-Dep_hour)
        dur_min = abs(Arrival_min-Dep_min)

        Arrival_hour = "{}".format(Arrival_hour)
        Arrival_min = "{}".format(Arrival_min)
        
        # Total Stops
        Total_Stops = request.form['stops']
        
        # Airline
        airline = request.form['airline']
        if (airline=='Jet_Airways'):
            Jet_Airways = 1
            Indigo = 0
            Air_India = 0
            GoAir = 0
            Jet_Airways_Business = 0
            Multiple_carriers = 0
            Multiple_carriers_Premium_economy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            Vistara_Premium_economy = 0
            
        elif (airline=='Indigo'):
            Jet_Airways = 0
            Indigo = 1
            Air_India = 0
            GoAir = 0
            Jet_Airways_Business = 0
            Multiple_carriers = 0
            Multiple_carriers_Premium_economy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            Vistara_Premium_economy = 0
            
        elif (airline=='Air_India'):
            Jet_Airways = 0
            Indigo = 0
            Air_India = 1
            GoAir = 0
            Jet_Airways_Business = 0
            Multiple_carriers = 0
            Multiple_carriers_Premium_economy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            Vistara_Premium_economy = 0
            
        elif (airline=='GoAir'):
            Jet_Airways = 0
            Indigo = 0
            Air_India = 0
            GoAir = 1 
            Jet_Airways_Business = 0
            Multiple_carriers = 0
            Multiple_carriers_Premium_economy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            Vistara_Premium_economy = 0
            
        elif (airline=='Jet_Airways_Business'):
            Jet_Airways = 0
            Indigo = 0
            Air_India = 0
            GoAir = 0
            Jet_Airways_Business = 1
            Multiple_carriers = 0
            Multiple_carriers_Premium_economy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            Vistara_Premium_economy = 0
            
        elif (airline=='Multiple_carriers'):
            Jet_Airways = 0
            Indigo = 0
            Air_India = 0
            GoAir = 0
            Jet_Airways_Business = 0
            Multiple_carriers = 1
            Multiple_carriers_Premium_economy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            Vistara_Premium_economy = 0
            
        elif (airline=='Multiple_carriers_Premium_economy'):
            Jet_Airways = 0
            Indigo = 0
            Air_India = 0
            GoAir = 0
            Jet_Airways_Business = 0
            Multiple_carriers = 0
            Multiple_carriers_Premium_economy = 1
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            Vistara_Premium_economy = 0
            
        elif (airline=='SpiceJet'):
            Jet_Airways = 0
            Indigo = 0
            Air_India = 0
            GoAir = 0
            Jet_Airways_Business = 0
            Multiple_carriers = 0
            Multiple_carriers_Premium_economy = 0
            SpiceJet = 1
            Trujet = 0
            Vistara = 0
            Vistara_Premium_economy = 0
            
        elif (airline=='Trujet'):
            Jet_Airways = 0
            Indigo = 0
            Air_India = 0
            GoAir = 0
            Jet_Airways_Business = 0
            Multiple_carriers = 0
            Multiple_carriers_Premium_economy = 0
            SpiceJet = 0
            Trujet = 1
            Vistara = 0
            Vistara_Premium_economy = 0
            
        elif (airline=='Vistara'):
            Jet_Airways = 0
            Indigo = 0
            Air_India = 0
            GoAir = 0
            Jet_Airways_Business = 0
            Multiple_carriers = 0
            Multiple_carriers_Premium_economy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 1
            Vistara_Premium_economy = 0
            
        elif (airline=='Vistara_Premium_economy'):
            Jet_Airways = 0
            Indigo = 0
            Air_India = 0
            GoAir = 0
            Jet_Airways_Business = 0
            Multiple_carriers = 0
            Multiple_carriers_Premium_economy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            Vistara_Premium_economy = 1
            
        else:
            Jet_Airways = 0
            Indigo = 0
            Air_India = 0
            GoAir = 0
            Jet_Airways_Business = 0
            Multiple_carriers = 0
            Multiple_carriers_Premium_economy = 0
            SpiceJet = 0
            Trujet = 0
            Vistara = 0
            Vistara_Premium_economy = 0
            
        # Source
        Source = request.form['Source']
        if (Source == 'Delhi'):
            s_Delhi = 1
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 0
            
        elif (Source == 'Kolkata'):
            s_Delhi = 0
            s_Kolkata = 1
            s_Mumbai = 0
            s_Chennai = 0
            
        elif (Source == 'Mumbai'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 1
            s_Chennai = 0
            
        elif (Source == 'Chennai'):
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 1
            
        else: 
            s_Delhi = 0
            s_Kolkata = 0
            s_Mumbai = 0
            s_Chennai = 0
            
        # Destination
        Destination = request.form['Destination']
        if (Destination == 'Cochin'):
            d_Cochin = 1
            d_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
            d_New_Delhi = 0
            
        elif (Destination == 'Delhi'):
            d_Cochin = 0
            d_Delhi = 1
            d_Hyderabad = 0
            d_Kolkata = 0
            d_New_Delhi = 0
            
        elif (Destination == 'Hyderabad'):
            d_Cochin = 0
            d_Delhi = 0
            d_Hyderabad = 1
            d_Kolkata = 0
            d_New_Delhi = 0
            
        elif (Destination == 'Kolkata'):
            d_Cochin = 0
            d_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 1
            d_New_Delhi = 0
            
        elif (Destination == 'New_Delhi'):
            d_Cochin = 0
            d_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
            d_New_Delhi = 1
            
        else: 
            d_Cochin = 0
            d_Delhi = 0
            d_Hyderabad = 0
            d_Kolkata = 0
            d_New_Delhi = 0
  
        prediction = model.predict([[Total_Stops, Journey_day, Journey_month, Dep_hour, Dep_min, Arrival_hour, Arrival_min, dur_hour, dur_min, Air_India, GoAir, Indigo, Jet_Airways, Jet_Airways_Business, Multiple_carriers, Multiple_carriers_Premium_economy, SpiceJet, Trujet, Vistara, Vistara_Premium_economy, s_Chennai, s_Delhi, s_Kolkata, s_Mumbai, d_Cochin, d_Delhi, d_Hyderabad, d_Kolkata, d_New_Delhi]])
        logging.info("Prediction is now completed...")
        output = round(prediction[0], 2)
        
        return render_template('home.html', prediction_text='Your flight price is {}'.format(output))
    
    return render_template('home.html')



if __name__ == "__main__":
    logging.info("The execution has started")
    
    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        #data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_path, test_path)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
    
    #app.run(host='0.0.0.0', port=8000)