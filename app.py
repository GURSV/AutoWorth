import streamlit as st
import joblib
import numpy as np

model = joblib.load('random_forest_model.pkl')

company_encoder = joblib.load('company_encoder.pkl')
fuel_type_encoder = joblib.load('fueltype_encoder.pkl')
color_encoder = joblib.load('colour_encoder.pkl')
bodystyle_encoder = joblib.load('bodystyle_encoder.pkl')
dealer_state_encoder = joblib.load('dealerstate_encoder.pkl')
city_encoder = joblib.load('city_encoder.pkl')

st.title("AutoWorth: Indian Used Car Price Predictor")
st.subheader("Enter the car details:")

company_options = company_encoder.classes_.tolist()
fuel_type_options = fuel_type_encoder.classes_.tolist()
bodystyle_options = bodystyle_encoder.classes_.tolist()
dealer_state_options = dealer_state_encoder.classes_.tolist()
city_options = city_encoder.classes_.tolist()

company = st.selectbox("Company", company_options)
fuel_type = st.selectbox("Fuel Type", fuel_type_options)
color = st.text_input("Color", placeholder="Enter the color of the car")
kilometer = st.number_input("Kilometer Driven", min_value=0, max_value=1000000, step=1)
bodystyle = st.selectbox("Body Style", bodystyle_options)
age = st.number_input("Car Age (years)", min_value=0, max_value=100, step=1)
owner = st.selectbox("Owner", ['1st Owner', '2nd Owner', '3rd Owner', '4th Owner'])
dealer_state = st.selectbox("Dealer State", dealer_state_options)
dealer_name = st.text_input("Dealer Name", placeholder="Enter the name of the dealer")
city = st.selectbox("City", city_options)
warranty = st.number_input("Warranty (years)", min_value=0, max_value=10, step=1)
quality_score = st.number_input("Quality Score", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict Price"):
    try:
        company_encoded = company_encoder.transform([company])[0]
        fuel_type_encoded = fuel_type_encoder.transform([fuel_type])[0]
        
        try:
            color_encoded = color_encoder.transform([color])[0]
        except ValueError:
            st.error(f"Invalid color entered. Valid colors are: {', '.join(color_encoder.classes_)}")
            st.stop()
            
        bodystyle_encoded = bodystyle_encoder.transform([bodystyle])[0]
        dealer_state_encoded = dealer_state_encoder.transform([dealer_state])[0]
        city_encoded = city_encoder.transform([city])[0]

        owner_mapping = {'1st Owner': 0, '2nd Owner': 1, '3rd Owner': 2, '4th Owner': 3}
        owner_encoded = owner_mapping[owner]

        input_data = np.array([
            company_encoded,
            fuel_type_encoded,
            color_encoded,
            kilometer,
            bodystyle_encoded,
            age,
            owner_encoded,
            dealer_state_encoded,
            len(dealer_name),  # Using dealer name length as a feature...
            city_encoded,
            warranty,
            quality_score
        ]).reshape(1, -1)

        predicted_price = model.predict(input_data)[0]
        st.success(f"The predicted price of the car is ₹{predicted_price:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check all your inputs and try again.")