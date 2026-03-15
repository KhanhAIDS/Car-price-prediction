import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

# 1. Tải Pipeline (Preprocessor + Model + Features) và cache để tăng tốc tải trang
@st.cache_resource
def load_pipeline():
    return joblib.load('best_car_price_model.pkl')

pipeline_data = load_pipeline()
preprocessor = pipeline_data['preprocessor']
model = pipeline_data['model']
features = pipeline_data['features']

# 2. Tải cấu trúc dữ liệu JSON để hiển thị UI
with open('./data.json', 'r') as json_file:
    data = json.load(json_file)

st.title("Car Price Prediction")

# 3. Giao diện nhập liệu
brand = st.selectbox('Car brand:', sorted(data.keys()))
model_name = st.selectbox('Model:', sorted(data[brand].keys()))

# Thông tin chỉ để hiển thị (không đưa vào model vì đã drop)
seats = data[brand][model_name].get('Số chỗ ngồi', 'N/A')
doors = data[brand][model_name].get('Số cửa', 'N/A')
st.write("Number of seats:", seats, "Number of doors:", doors)

# Tình trạng không đưa vào model, chỉ dùng để tự động set Km
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

# 4. Xử lý dự đoán
submit_button = st.button('Predict')

if submit_button:
    # Cấu trúc Dataframe đúng với format 9 cột mà preprocessor yêu cầu
    user_input = {
        "Số Km đã đi": km,
        "Xuất xứ": origin,
        "Kiểu dáng": style,
        "Hộp số": gear,
        "Dẫn động": transmission,
        "Loại nhiên liệu": fuel,
        "Dung tích động cơ": float(cylinder),
        "Tuổi": age,
        "Brand_Model": f"{brand}_{model_name}" # Gộp tên thương hiệu và model
    }

    user_df = pd.DataFrame([user_input])

    try:
        # Tự động thực hiện mã hóa (OneHot & LeaveOneOut)
        user_encoded = preprocessor.transform(user_df)
        user_encoded_df = pd.DataFrame(user_encoded, columns=features)
        
        # Dự đoán logarit và giải mã (expm1)
        log_prediction = model.predict(user_encoded_df)
        real_price = np.expm1(log_prediction)[0]
        
        st.success(f"Predicted price: {real_price:,.0f}")
    except Exception as e:
        st.error(f"Lỗi xử lý: {e}")