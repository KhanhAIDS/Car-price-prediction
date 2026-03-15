import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

@st.cache_resource
def load_pipeline():
    return joblib.load('best_car_price_model.pkl')

pipeline_data = load_pipeline()
preprocessor = pipeline_data['preprocessor']
model = pipeline_data['model']
features = pipeline_data['features']

with open('./data.json', 'r') as json_file:
    data = json.load(json_file)

st.title("Car Price Prediction")

brand = st.selectbox('Car brand:', sorted(data.keys()))
model_name = st.selectbox('Model:', sorted(data[brand].keys()))

seats = data[brand][model_name].get('Số chỗ ngồi', 'N/A')
doors = data[brand][model_name].get('Số cửa', 'N/A')
st.write("Number of seats:", seats, "Number of doors:", doors)

status = st.selectbox('Status:', ['New', 'Used'])
origin = st.selectbox('Origin:', sorted(data[brand][model_name]['Xuất xứ']))
style = st.selectbox('Style:', sorted(data[brand][model_name]['Kiểu dáng']))
gear = st.selectbox('Gear:', sorted(data[brand][model_name]['Hộp số']))
transmission = st.selectbox('Transmission:', sorted(data[brand][model_name]['Dẫn động']))
fuel = st.selectbox('Fuel type:', sorted(data[brand][model_name]['Loại nhiên liệu']))
cylinder = st.selectbox('Cylinder capacity:', sorted(data[brand][model_name]['Dung tích động cơ']))

if status == 'New':
    km = 0
    st.write('Traveled Km:', km)
else:
    km = st.slider('Traveled Km:', 0, 150000, 0)
    
age = st.slider('Age:', 0, 50, 0)

submit_button = st.button('Predict')

if submit_button:
    user_input = {
        "Số Km đã đi": km,
        "Xuất xứ": origin,
        "Kiểu dáng": style,
        "Hộp số": gear,
        "Dẫn động": transmission,
        "Loại nhiên liệu": fuel,
        "Dung tích động cơ": float(cylinder),
        "Tuổi": age,
        "Brand_Model": f"{brand}_{model_name}"
    }

    user_df = pd.DataFrame([user_input])

    try:
        user_encoded = preprocessor.transform(user_df)
        user_encoded_df = pd.DataFrame(user_encoded, columns=features)

        log_prediction = model.predict(user_encoded_df)
        real_price = np.expm1(log_prediction)[0]
        
        st.success(f"Predicted price: {real_price:,.0f}")
    except Exception as e:
        st.error(f"Lỗi xử lý: {e}")
