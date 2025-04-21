import streamlit as st
import joblib

# 加载模型和TF-IDF向量器
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# 网页标题
st.title("🛡️ 恶意网址检测系统")
st.write("请输入一个网址，系统将预测它是否为恶意网址。")

# 用户输入框
user_input = st.text_input("🌐 输入网址：")

# 检测按钮
if st.button("立即检测"):
    if user_input:
        # 向量化输入
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]

        if prediction == 0:
            st.success("✅ 安全网址")
        else:
            st.error("⚠️ 警告：该网址可能是恶意的！")
    else:
        st.warning("请输入网址！")
