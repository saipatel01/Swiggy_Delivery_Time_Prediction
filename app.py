
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import gdown

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        file_id = "18DUwaTypnR462Bva_f6i230Er4CMANuf"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "model.pkl", quiet=False)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    return model

model = load_model()



st.title("Swiggy Delivery Time Prediction")
df = pd.read_csv(r'swiggy_cleaned.csv')
df.dropna(inplace=True)

age = st.number_input("Enter the Rider Age :",min_value=20,max_value=50)
ratings = st.number_input("Enter the Rider Rating",min_value=0,max_value=5)
weather = st.selectbox('Enter the weather condition',df['weather'].unique())
traffic = st.selectbox('Enter the traffic condition',df['traffic'].unique())
vehicle_condition = st.selectbox('Enter the Vehicle condition',df['vehicle_condition'].unique())
type_of_vehicle = st.selectbox('Enter the Type of vehicle',df[ 'type_of_vehicle'].unique())
multiple_deliveries = st.selectbox('Enter the multiple_deliveries',df['multiple_deliveries'].unique())
festival = st.selectbox('Is it festival',df['festival'].unique())
city_name  = st.selectbox('Enter city_name ',df['city_name'].unique())
is_weekend = st.selectbox('Is it weekend',df['is_weekend'].unique())
pickup_time_minutes = st.number_input("Enter the Pick up Time")
order_time_hour = st.number_input("Enter the Order hour",min_value=0,max_value=24)
distance = st.number_input("Enter the Distance between restaurent and delivery location")

columns_to_drop = ['rider_id','restaurant_latitude',
                   'restaurant_longitude','delivery_latitude',
                   'delivery_longitude','order_date','type_of_order','city_type','order_day','order_month',
                   'order_day_of_week','order_time_of_day']
df.drop(columns=columns_to_drop,inplace=True)

X = df.drop(columns='time_taken')
data = pd.DataFrame([[age,ratings,weather,traffic,
               vehicle_condition,type_of_vehicle,multiple_deliveries,
               festival,city_name,is_weekend,pickup_time_minutes,
               order_time_hour,distance]],columns=X.columns)

if st.button("Predict"):
    time = model.predict(data)[0]
    st.write(f"The Delivery Time is {time} mins")
    st.snow()
