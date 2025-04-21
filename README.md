# 🛡️ 恶意网址检测系统（AI + 网络安全）

一个基于机器学习的恶意网址识别系统，能够自动判断用户输入的网址是否安全，适用于网络安全预警、浏览器插件、防钓鱼工具等场景。

## 🌐 在线体验地址

👉 [点击访问项目演示](https://akiko.cloud)

> 已绑定独立域名 `akiko.cloud`，通过 Streamlit Cloud 部署，支持任意设备访问体验。

---

## 🚀 项目亮点

- ✅ 使用真实世界恶意网站数据集（65万条网址）
- ✅ 二分类模型识别恶意网址（钓鱼/伪装/恶意软件）
- ✅ 支持输入任意网址，自动判断其是否安全
- ✅ 使用 Streamlit 打造交互式网页应用
- ✅ 已部署上线 + 绑定独立域名（akiko.cloud）

---

## 💡 使用的技术栈

| 技术 | 用途 |
|------|------|
| Python | 开发主语言 |
| Pandas | 数据处理 |
| Scikit-learn | 机器学习模型 |
| TF-IDF | 文本特征提取 |
| Streamlit | 网页界面开发 |
| Cloudflare + 阿里云 | 域名绑定与转发部署 |
| Git + GitHub | 版本管理和代码托管 |

---

## 📦 项目结构

```plaintext
url_classifier/
├── url_app.py                # Streamlit 网页主程序
├── malicious_phish.csv      # 数据集（未上传）
├── rf_model.pkl              # 训练好的模型
├── tfidf_vectorizer.pkl      # TF-IDF 向量器
├── requirements.txt          # 项目依赖列表
└── README.md                 # 本文件
