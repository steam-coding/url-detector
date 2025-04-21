# ========== 🔧 导入必要的库 ==========
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========== 📂 第一步：加载数据 ==========
# 从本地读取CSV文件
df = pd.read_csv('malicious_phish.csv')

# 打印前几行，确认数据结构（可选）
# print("【原始数据预览】")
# print(df.head())

# ========== 🧹 第二步：数据预处理 ==========

# 将多类别的"type"标签（benign, phishing等）转换为二分类标签
# benign → 0（正常网址），其余如 phishing/malware/defacement → 1（恶意网址）
df['label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)

# 输出处理后的标签分布
print("\n【二分类标签分布】")
print(df['label'].value_counts())

# ========== 🔡 第三步：特征提取（文本向量化） ==========
# 使用 TF-IDF 算法将网址字符串转换为稀疏矩阵特征（向量形式）
vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(df['url'])  # 特征矩阵
# y = df['label']                          # 标签列

# ========== 📦 第四步：划分训练集和测试集 ==========
# 将数据按 8:2 划分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# ========== 🧠 第五步：模型训练 ==========
# 使用朴素贝叶斯分类器进行训练（适合文本分类任务）
# model = MultinomialNB()
# model.fit(X_train, y_train)

# ========== 🔍 第六步：模型预测与评估 ==========
# 使用训练好的模型对测试集进行预测
# y_pred = model.predict(X_test)

# 计算模型在测试集上的准确率
# acc = accuracy_score(y_test, y_pred)
# print(f"\n✅ 模型准确率：{acc:.4f}")

# 打印更详细的分类报告（包含精确率、召回率、F1分数）
# print("\n📋 分类报告（Classification Report）:")
# print(classification_report(y_test, y_pred))

# 打印混淆矩阵（便于了解分类错误情况）
# print("\n📊 混淆矩阵（Confusion Matrix）:")
# print(confusion_matrix(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 抽样
df_sampled = df.sample(n=80000, random_state=42)

# 特征处理
X = vectorizer.fit_transform(df_sampled['url'])
y = df_sampled['label']

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 模型列表（减小随机森林树数）
models = {
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=20, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear')
}

for name, model in models.items():
    print(f"\n🚀 正在训练模型：{name}")
    model.fit(X_train, y_train)  # 训练模型
    y_pred = model.predict(X_test)  # 预测结果

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 模型准确率：{acc:.4f}")
    print("📋 分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("📊 混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))


import joblib

# 假设随机森林表现最好，保存它
joblib.dump(models["Random Forest"], "rf_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
