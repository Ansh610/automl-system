# 🚀 AutoML Intelligence Platform

A full-stack **AutoML web application** that automatically trains multiple machine learning models, evaluates their performance, detects bias, generates insights, and provides an interactive dashboard for analysis and prediction.

The platform allows users to upload datasets, run automated model training, compare results visually, and make real-time predictions through a modern React dashboard.

---

# 🌐 Live Demo

🔗 **Live App:**
https://automl-system-1.onrender.com

📂 **GitHub Repository:**
https://github.com/Ansh610/automl-system

---

# 📊 Features

### 🤖 Automated Machine Learning

Automatically trains and evaluates multiple ML models including:

* Logistic Regression
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)
* XGBoost

Performs hyperparameter tuning using **GridSearchCV** and selects the best model automatically.

---

### 📈 Model Performance Dashboard

Interactive charts built with **Recharts**:

* Model accuracy comparison
* Classification performance metrics
* ROC curve with AUC
* Feature importance visualization
* Confusion matrix visualization

---

### ⚖️ Bias Detection

Detects potential bias in model predictions across sensitive attributes such as:

* Gender
* City

Helps identify fairness issues in machine learning models.

---

### 🔎 Data Profiling

Automatically generates a dataset report including:

* Number of rows
* Number of columns
* Missing values
* Dataset preview

---

### 💡 AI Business Insights

Generates automated insights from the dataset to help understand patterns and potential business outcomes.

---

### 🔮 Real-Time Prediction

Users can manually input customer data to predict conversion likelihood.

Inputs include:

* Age
* Income
* City
* Gender
* Website visits
* Time spent

The model returns:

* Prediction (Convert / Not Convert)
* Probability score

---

### 📂 Dataset Generator

Built-in endpoint to generate synthetic lead datasets for testing.

---

# 🧠 System Architecture

Frontend and backend are separated for scalability.

```
React Dashboard
        │
        │ API Requests
        ▼
FastAPI Backend
        │
        ├── AutoML Training
        ├── Bias Detection
        ├── Explainability
        ├── Dataset Insights
        ▼
Machine Learning Models
```

---

# 🛠 Tech Stack

### Frontend

* React
* Material UI
* Recharts
* Axios
* React Dropzone

### Backend

* FastAPI
* Uvicorn

### Machine Learning

* Scikit-learn
* XGBoost
* Pandas
* NumPy
* Joblib

### Deployment

* Render

# 👨‍💻 Author

**Ansh**


