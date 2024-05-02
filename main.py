import numpy as np
import streamlit as st
import pandas as pd
import pickle
import datetime as dt
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model (Ensure the model file path is correct)
model = pickle.load(open('RandomForestRegressor.pkl', 'rb'))


def handle_missing_values(df):
    df['Levy'] = df['Levy'].replace('-', np.nan)
    df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')
    df['Levy'].fillna(df['Levy'].median(), inplace=True)  # Replace missing Levy with the median

    df['Engine volume'] = df['Engine volume'].replace('Turbo', '', regex=True)
    df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce')

    df['Mileage'] = df['Mileage'].str.replace(' km', '').str.replace(',', '').astype('Int64')
    df['Mileage'].fillna(df['Mileage'].median(), inplace=True)  # Replace missing Mileage with the median

    # For categorical data, consider filling missing values with the mode
    for column in df.select_dtypes(include=['object']).columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    return df


def remove_outliers(df):
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def encode_categorical(df):
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure all data is string type
        label_encoders[col] = le
    return df, label_encoders


def scale_features(df, numeric_columns):
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df, scaler


def preprocess_data(df):
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df, label_encoders = encode_categorical(df)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df, scaler = scale_features(df, numeric_cols)
    return df, label_encoders, scaler


# Usage
# df, label_encoders, scaler = preprocess_data(training_data)
# For prediction: Apply label_encoders and scaler as stored from training preprocessing

def predict_price(input_data):
    # Load the data
    df = pd.DataFrame([input_data])

    # Preprocess the data
    df, label_encoders, scaler = preprocess_data(df)

    # Ensure the columns are in the same order as when the model was trained
    model_columns = ['Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type', "Gear box type",
                     "Drive wheels", 'Wheel', "Color", 'Levy', 'Engine volume',
                     'Mileage', 'Cylinders', 'Airbags', 'Age']
    df = df[model_columns]

    # Make predictions
    prediction = model.predict(df)
    return prediction[0]


def app():
    # Setting up the Streamlit interface
    st.title('Car Price Prediction App')

    # Creating user input fields
    with st.form(key='prediction_form'):
        manufacturer = st.selectbox('Select the Manufacturer', options=[
            'LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ', 'OPEL', 'PORSCHE', 'BMW',
            'JEEP', 'VOLKSWAGEN', 'AUDI', 'RENAULT', 'NISSAN', 'SUBARU', 'DAEWOO', 'KIA', 'MITSUBISHI', 'SSANGYONG',
            'MAZDA', 'GMC', 'FIAT', 'INFINITI', 'ALFA ROMEO', 'SUZUKI', 'ACURA', 'LINCOLN', 'VAZ', 'GAZ', 'CITROEN',
            'LAND ROVER', 'MINI', 'DODGE', 'CHRYSLER', 'JAGUAR', 'ISUZU', 'SKODA', 'DAIHATSU', 'BUICK', 'TESLA',
            'CADILLAC', 'PEUGEOT', 'BENTLEY', 'VOLVO', 'სხვა', 'HAVAL', 'HUMMER', 'SCION', 'UAZ', 'MERCURY', 'ZAZ',
            'ROVER', 'SEAT', 'LANCIA', 'MOSKVICH', 'MASERATI', 'FERRARI', 'SAAB', 'LAMBORGHINI', 'ROLLS-ROYCE',
            'PONTIAC', 'SATURN', 'ASTON MARTIN', 'GREATWALL'])
        model = st.text_input('Enter the Model of the Car', value='RX 450')
        Levy = st.text_input('Levy', value='0')
        type = st.selectbox('Type',
                            options=['Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon', 'Universal', 'Coupe',
                                     'Minivan', 'Cabriolet', 'Limousine', 'Pickup'])
        prod_year = st.number_input('Production Year', min_value=1939, max_value=2020, value=2010)
        mileage = st.text_input('Mileage in km', value='50000 km')
        fuel_type = st.selectbox('Fuel Type',
                                 options=['Hybrid', 'Petrol', 'Diesel', 'CNG', 'Plug-in Hybrid', 'LPG', 'Hydrogen'])
        volume = st.selectbox('Engine Volume', options=[
            '3.5', '3', '1.3', '2.5', '2', '1.8', '2.4', '4', '1.6', '3.3', '2.0 Turbo',
            '2.2 Turbo', '4.7', '1.5', '4.4', '3.0 Turbo', '1.4 Turbo', '3.6', '2.3',
            '1.5 Turbo', '1.6 Turbo', '2.2', '2.3 Turbo', '1.4', '5.5', '2.8 Turbo'
        ])
        color = st.selectbox('Color', options=[
            'Silver', 'Black', 'White', 'Grey', 'Blue', 'Green', 'Red', 'Sky blue', 'Orange',
            'Yellow', 'Brown', 'Golden', 'Beige', 'Carnelian red', 'Purple', 'Pink'
        ])
        gear_box_type = st.selectbox('Gear Box Type', options=['Automatic', 'Tiptronic', 'Variator', 'Manual'])
        drive_wheels = st.selectbox('Drive Wheels', options=['4x4', 'Front', 'Rear'])
        Cylinders = st.selectbox('Cylinders', options=[6, 4, 8, 1, 12, 3, 2, 16, 5, 7, 9, 10, 14])
        airBag = st.selectbox('Air Bag', options=[12, 8, 2, 0, 4, 6, 10, 3, 1, 16, 5, 7, 9, 11, 14, 15, 13])
        leather_interior = st.selectbox('Leather Interior', options=['Yes', 'No'])
        Wheel = st.selectbox('Wheel', options=['Left wheel', 'Right-hand drive'])
        submit_button = st.form_submit_button(label='Predict Price')

    # When the submit button is clicked
    dtime = dt.datetime.now()
    if submit_button:
        input_data = {
            'Manufacturer': manufacturer,
            'Model': model,
            'Age': dtime.year - prod_year,
            "Category": type,
            'Mileage': mileage,
            'Fuel type': fuel_type,
            'Engine volume': volume,
            'Color': color,
            "Gear box type": gear_box_type,
            "Leather interior": leather_interior,
            "Drive wheels": drive_wheels,
            "Airbags": airBag,
            "Cylinders": Cylinders,
            "Wheel": Wheel,
            'Levy': Levy
        }
        prediction = predict_price(input_data)
        st.success(f'The estimated price of the car is {prediction:.2f} $')


# Run this in your command line: streamlit run app.py

if __name__ == '__main__':
    app()