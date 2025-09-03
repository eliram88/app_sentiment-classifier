import streamlit as st
import joblib
import numpy as np

# بارگذاری مدل
model = joblib.load('xgb_sentiment_model.pkl')

st.set_page_config(page_title="App Review Sentiment Predictor", layout="centered")

# عنوان
st.title("📱 Predicting user sentiment towards apps")

st.markdown("Predict the overall sentiment of users by entering app features!")

# ورودی‌ها
size_mb = st.slider("App Size (MB):", min_value=1.0, max_value=100.0, step=0.5)
rating = st.slider("User rating:", min_value=1.0, max_value=5.0, step=0.1)

category_labels = {
    0: "🎮 Games",
    1: "📚 Educational",
    2: "💰 Financial",
    3: "💬 Social",
    4: "🛠 Tools",
    5: "🧭 Travel"
}

category_code = st.selectbox(
    "App Category:",
    options=list(category_labels.keys()),
    format_func=lambda x: category_labels[x]
)

type_label = st.radio(
    "App Type:",
    options=["Free", "Paid"]
)

# تبدیل انتخاب کاربر به کد عددی
type_code = 0 if type_label == "Free" else 1

# پیش‌بینی
if st.button("Predict Sentiment"):
    features = np.array([[size_mb, rating, category_code, type_code]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("✅ The overall sentiment of users is **positive**.")
    else:
        st.error("❌ The overall user sentiment is **negative**.")
