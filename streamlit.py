import streamlit as st
import pandas as pd
import pickle
import os

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("xgb_smote_model.pkl", "rb"))   # <-- change filename if different

model = load_model()

# -------------------------
# UI Layout
# -------------------------
st.title("📊 Customer Campaign Response Prediction App")
st.write("Fill in customer details below and click **Predict** to see the result.")

logo_path = "images/parami.jpg"

st.sidebar.markdown("👩‍🎓 Student Info")

st.sidebar.markdown("---")  

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown("---")  


st.sidebar.markdown("**Name:** Thant Sin Tun")
st.sidebar.markdown("**Student ID:** PIUS20230003")
st.sidebar.markdown("**Class:** 2027")
st.sidebar.markdown("Intro to Machine Learning")

st.sidebar.markdown("---")  

# -------------------------
# Input Form Section
# -------------------------
with st.form("user_input_form"):

    st.subheader("🧍 Customer Profile")
    age = st.number_input("Age", min_value=0, value=25, max_value=100)
    family_size = st.number_input("Family Size", min_value=1)
    marital = st.number_input("Marital Status (Married/Together = 1, else 0)", min_value=0, max_value=1)

    st.subheader("💰 Income Information")
    income = st.slider("Enter Customer's Income $",min_value=0,max_value=120000,value=50000, step=1000)
    
    st.subheader("🎓 Education Level")
    education = st.radio(
    "Select the customer's highest education level:",
    ["Undergraduate", "Graduate", "Postgraduate"],
    horizontal=True)
    # Convert to one-hot encoded values
    edu_undergrad = 1 if education == "Undergraduate" else 0
    edu_grad = 1 if education == "Graduate" else 0
    edu_postgrad = 1 if education == "Postgraduate" else 0


    st.subheader("📅 Member Enrollment Date")
    st.write("Please provide the member's enrollment year and month to the company. ")
    year = st.number_input("Enrollment Year (YYYY)", min_value=2012, max_value=2014)
    month = st.number_input("Enrollment Month (1-12)", min_value=1, max_value=12)
    

    st.subheader("🛒 Purchase & Spending History")
    total_mnt = st.slider("Total Amount Spent", min_value=0,max_value=3000,value=500)
    recency = st.slider("Recency (Days since last purchase)", min_value=0, value=5, max_value=100)
    numdp = st.slider("NumDealsPurchases", min_value=0, max_value=30)
    numwebp = st.slider("NumWebPurchases", min_value=0, max_value=30)
    numcatp = st.slider("NumCatalogPurchases", min_value=0, max_value=30)
    numsp = st.slider("NumStorePurchases", min_value=0, max_value=30)
    numwebvm = st.slider("NumWebVisitsMonth", min_value=0, max_value=30)

    st.subheader("📢 Previous Campaign Response")
    acmp1 = st.radio("Respond to Campaign 1 ?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    acmp2 = st.radio("Respond to Campaign 2 ?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    acmp3 = st.radio("Respond to Campaign 3 ?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    acmp4 = st.radio("Respond to Campaign 4 ?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    acmp5 = st.radio("Respond to Campaign 5 ?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")



    submitted = st.form_submit_button("Predict")

# -------------------------
# Prepare Data + Predict
# -------------------------
if submitted:
    user_data = pd.DataFrame([{
        'Income': income,
        'Recency': recency,
        'NumDealsPurchases': numdp,
        'NumWebPurchases': numwebp,
        'NumCatalogPurchases': numcatp,
        'NumStorePurchases': numsp,
        'NumWebVisitsMonth': numwebvm,
        'AcceptedCmp3': acmp3,
        'AcceptedCmp4': acmp4,
        'AcceptedCmp5': acmp5,
        'AcceptedCmp1': acmp1,
        'AcceptedCmp2': acmp2,
        'Education_Graduate': edu_grad,
        'Education_Postgraduate': edu_postgrad,
        'Education_Undergraduate': edu_undergrad,
        'NewMaritalStatus': marital,
        'Age': age,
        'Year': year,
        'Month': month,
        'TotalMntSpent': total_mnt,
        'FamilySize': family_size
    }])

    # st.write("### 📥 Your Input Data")
    # st.dataframe(user_data)

    pred = model.predict(user_data)[0]

    if pred == 1:
        st.success("💡 Customer is likely to respond to the campaign!")
    else:
        st.warning("⚠ Customer is unlikely to respond.")






