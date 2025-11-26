import streamlit as st
import pickle
import os
import pandas as pd

st.title("Marketing Campaign Classification")
st.write("This model predicts customer response upon marketing campaign based on the information you provide.")

st.subheader("Input Customer Information")
st.write("Fill out the fields below and click Predict.")

def load_model():
    with open("marketing_campaign_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

logo_path = "images/parami.jpg"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown("**Student Name:** Thant Sin Tun")
st.sidebar.markdown("**Student ID:** PIUS20230003")
st.sidebar.markdown("(Class of 2027)")
st.sidebar.markdown("(Intro to Machine Learning)")


education = st.selectbox(
    "Enter customer education",
    ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
)
marital_status = st.selectbox(
    "Enter marital status",
    ["Single", "Together", "Married", "Divorced", "Widow", "Alone", "Absurd", "YOLO"]
)

income = st.slider("Enter Income",min_value=0,max_value=700000,value=50000, step=1000)


# income = st.number_input("Enter customer income", min_value=0.0, max_value=9999999.0)
# p_l = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
# p_w = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)


response_images = {
    'No': 'images/setosa.jpg',
    'Yes': 'images/versicolor.jpg'
}

kid_home = st.radio("Kid at Home?", [0, 1])
teen_home = st.radio("Teen at Home?", [0, 1])
recency = st.number_input("Recency", min_value=0)

complain = st.radio("Complain?", [0, 1])

# ----- SPENDING -----
st.subheader("Monthly Spending")

wine = st.number_input("MntWines", min_value=0)
fruit = st.number_input("MntFruits", min_value=0)
meat = st.number_input("MntMeatProducts", min_value=0)
fish = st.number_input("MntFishProducts", min_value=0)
sweet = st.number_input("MntSweetProducts", min_value=0)
gold = st.number_input("MntGoldProds", min_value=0)

# ----- PURCHASES -----
st.subheader("Purchases Info")

numdealp = st.number_input("NumDealsPurchases", min_value=0)
numwebp = st.number_input("NumWebPurchases", min_value=0)
numcatp = st.number_input("NumCatalogPurchases", min_value=0)
numsp = st.number_input("NumStorePurchases", min_value=0)
numwebvm = st.number_input("NumWebVisitsMonth", min_value=0)

# ----- CAMPAIGNS -----
st.subheader("Campaign Acceptance")

acmp1 = st.radio("AcceptedCmp1", [0, 1])
acmp2 = st.radio("AcceptedCmp2", [0, 1])
acmp3 = st.radio("AcceptedCmp3", [0, 1])
acmp4 = st.radio("AcceptedCmp4", [0, 1])
acmp5 = st.radio("AcceptedCmp5", [0, 1])

# -------- BUTTON --------
if st.button("Predict"):
    user_data = pd.DataFrame([{
        'Education': education,
        'Marital_Status': marital_status,
        'Kidhome': kid_home,
        'Teenhome': teen_home,
        'Recency': recency,
        'MntWines': wine,
        'MntFruits': fruit,
        'MntMeatProducts': meat,
        'MntFishProducts': fish,
        'MntSweetProducts': sweet,
        'MntGoldProds': gold,
        'NumDealsPurchases': numdealp,
        'NumWebPurchases': numwebp,
        'NumCatalogPurchases': numcatp,
        'NumStorePurchases': numsp,
        'NumWebVisitsMonth': numwebvm,
        'AcceptedCmp1': acmp1,
        'AcceptedCmp2': acmp2,
        'AcceptedCmp3': acmp3,
        'AcceptedCmp4': acmp4,
        'AcceptedCmp5': acmp5,
        'Complain': complain,
        'Income': income
    }])
    
    model = load_model()
    result = model.predict(user_data)
    if result[0] == 1:
        st.success('Customer will respond the campaign!!')
    else:
        st.warning('Customer will not respond the campaign!!')

 
