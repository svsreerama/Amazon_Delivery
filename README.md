# 🚚 Amazon Delivery Time Prediction

This project predicts delivery times for Amazon-like e-commerce orders based on real-world features such as distance, traffic, weather, agent rating, and vehicle type.

---

## 🧠 Project Overview

- **Domain**: E-Commerce, Logistics
- **Objective**: Predict delivery time (in minutes) using regression models
- **Dataset**: `amazon_delivery.csv` (includes agent, order, and route details)

---

## 📁 Project Structure

```
amazon_delivery_project/
├── data/                   # Raw and processed data
│   ├── amazon_delivery.csv
│   ├── amazon_delivery_cleaned.csv
│   └── amazon_delivery_features.csv
├── scripts/                # Core Python scripts
│   ├── data_preparation.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_development.py
│   └── mlflow_tracking.py
├── models/                 # Saved trained models
│   └── xgboost_model.pkl
├── eda_plots/              # Auto-generated EDA images
├── streamlit_app.py        # Streamlit frontend
├── README.md               # 📄 You're reading it!
```

---

## 🔧 Setup Instructions

1. **Clone / Extract the Project**
   ```bash
   cd amazon_delivery_project
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Requirements**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost mlflow streamlit joblib
   ```

---

## 🚀 How to Run

### 1. Prepare the Data
```bash
python3 scripts/data_preparation.py
```

### 2. Run EDA and Save Charts
```bash
python3 scripts/eda.py
```

### 3. Generate Features
```bash
python3 scripts/feature_engineering.py
```

### 4. Train Models
```bash
python3 scripts/model_development.py
```

### 5. Log Experiments with MLflow
```bash
python3 scripts/mlflow_tracking.py
mlflow ui  # Then open http://localhost:5000
```

### 6. Launch Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## 🧪 Models Trained

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor ✅ (best model)

| Model             | RMSE   | MAE    | R²     |
|------------------|--------|--------|--------|
| Linear Regression| 33.15  | 26.22  | 0.582  |
| Random Forest    | 22.82  | 17.47  | 0.802  |
| XGBoost          | 22.58  | 17.54  | 0.806  |

---

## 📊 Key Features

- Agent Age, Rating
- Distance between store & drop
- Time features: Hour, Day of Week
- Weather, Traffic, Area
- Vehicle Type, Product Category

---

## 📦 Deliverables

- ✅ Cleaned dataset
- ✅ Feature-engineered dataset
- ✅ EDA plots
- ✅ Trained regression models
- ✅ Streamlit app for prediction
- ✅ MLflow tracking setup
- ✅ Documentation (`README.md`)

---

