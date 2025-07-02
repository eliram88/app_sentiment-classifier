import streamlit as st
import joblib
import numpy as np

# بارگذاری مدل
model = joblib.load('xgb_sentiment_model.pkl')

st.set_page_config(page_title="App Review Sentiment Predictor", layout="centered")

# عنوان
st.title("📱 پیش‌بینی احساس کاربران نسبت به اپ‌ها")

st.markdown("با وارد کردن ویژگی‌های اپ، احساس کلی کاربران را پیش‌بینی کن!")

# ورودی‌ها
size_mb = st.slider("حجم اپ (MB):", min_value=1.0, max_value=100.0, step=0.5)
rating = st.slider("امتیاز کاربران:", min_value=1.0, max_value=5.0, step=0.1)

category_labels = {
    0: "🎮 بازی‌ها",
    1: "📚 آموزشی",
    2: "💰 مالی",
    3: "💬 اجتماعی",
    4: "🛠 ابزارها",
    5: "🧭 سفر و گردشگری"
}

category_code = st.selectbox(
    "دسته‌بندی اپ:",
    options=list(category_labels.keys()),
    format_func=lambda x: category_labels[x]
)

type_label = st.radio(
    "نوع اپ:",
    options=["رایگان", "پولی"]
)

# تبدیل انتخاب کاربر به کد عددی
type_code = 0 if type_label == "رایگان" else 1

# پیش‌بینی
if st.button("پیش‌بینی احساس"):
    features = np.array([[size_mb, rating, category_code, type_code]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("✅ احساس کلی کاربران **مثبت** است.")
    else:
        st.error("❌ احساس کلی کاربران **منفی** است.")
