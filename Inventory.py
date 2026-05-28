import streamlit as st
import numpy as np
import joblib 

model = joblib.load('inventory_model.pkl')
location_encoder = joblib.load('location_encoder.pkl')
product_encoder = joblib.load('product_encoder.pkl')
label_encoder = joblib.load('label_encoder.pkl')

st.set_page_config(page_title='Inventory Management System',page_icon='📦',layout='centered')
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}
div.stButton > button {
    width: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title('ML Smart Inventory Optimization System')
st.write('Predict whether the stock is overstocked, understocked or safe')
st.divider()

product_type = st.selectbox('Product Type',['haircare','skincare','cosmetic'])
price = st.slider('Price(In Rs)',min_value=1,max_value=100)
revenue = st.slider('Revenue Generated(In Rs)',min_value=1061,max_value=9867)
stock = st.slider('Stock levels',min_value=0,max_value=100)
order = st.slider('Order quantities',min_value=1,max_value=96)
shipping = st.slider('Shipping times(In Days)',min_value=1,max_value=10)
location = st.selectbox('Location',['Delhi','Mumbai','Kolkata','Banglore','Chennai'])

location_encoded = location_encoder.transform([location])[0]
product_encoded = product_encoder.transform([product_type])[0]

st.divider()

if st.button('📦Predict Inventory Status'):

    input_data = np.array([[
        price,
        revenue,
        stock,
        order,
        shipping,
        location_encoded,
        product_encoded
    ]])

    prediction = model.predict(input_data)

    result = label_encoder.inverse_transform(prediction)

    st.subheader('📍Prediction Result')

    if result[0] == 'Overstocked':

        st.error('⚠️ Overstocked')

        st.write(
            f'''{product_type} in location {location} may remain unsold'''
        )

    elif result[0] == 'Understocked':

        st.warning('⚠️ Understocked')

        st.write(
            f'''{product_type} in location {location} may run out soon.'''
        )

    else:

        st.success('✅ Balanced')

        st.write(
            f'''{product_type} in location {location} seems balanced'''
        )

st.divider()
