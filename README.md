
# 🛢️ Sand Production Prediction Using Machine Learning

A petroleum engineering + machine learning project that predicts **sand production in oil wells** using **synthetic data generation**, **multiple ML algorithms**, and **feature importance analysis**.

---

## 📌 Project Overview
Sand production is a major challenge in petroleum engineering that can damage equipment, reduce production, and increase costs.  
This project demonstrates how **machine learning** can be applied to **predict sand production** based on realistic geological, rock mechanics, and operational parameters.

---

## 🎯 Objectives
- Generate realistic petroleum engineering dataset (synthetic but physics-based).
- Train and evaluate multiple machine learning models.
- Compare model performance using **R², RMSE, MAE**, and cross-validation.
- Interpret feature importance for engineering insights.
- Visualize results with professional plots.



## 🏗️ Project Workflow

The workflow of this project is divided into four main stages:

---

### 1️⃣ Data Generation
- Synthetic dataset is generated using petroleum engineering principles.
- Includes features like **Permeability, Cohesion, Grain Size, Density, Flow Rate, Drawdown, Friction Angle, Cement Quality, Clay Content, Reservoir Pressure, and Completion Type**.
- Target variable: **Sand_Production** (calculated from physical + operational parameters with random noise for realism).
- Dataset exported as `sand_data.csv`.

---

### 2️⃣ Model Training & Evaluation
- Input features are preprocessed:
  - Encode categorical variable (**Completion Type**).
  - Standardize features using `StandardScaler`.
- Split into **train (80%)** and **test (20%)** sets.
- Train three models:
  - **Linear Regression**
  - **Random Forest Regressor**
  - **Gradient Boosting Regressor**
- Evaluate models on:
  - R² (coefficient of determination)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Cross-validation error (CV)
- Best model selected based on **highest test R²**.

---

### 3️⃣ Feature Importance Analysis
- Use **Random Forest Regressor** for feature importance.
- Rank input features by contribution to sand production prediction.
- Export results as `feature_importance.csv`.

---

### 4️⃣ Visualization & Reporting
- Generate plots for:
  - Model comparison (R², RMSE)
  - Actual vs Predicted values
  - Top feature importance (bar chart)
  - Residual analysis
  - Distribution of actual vs predicted values
- Print summary table of model metrics in console.
- Save processed data and feature importance files for further use.

---

✅ This workflow ensures a **complete pipeline**: from data generation → model training → evaluation → feature interpretation → visualization.


## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/sand-production-ml.git
cd sand-production-ml
```

### 2️⃣ Install Dependencies

Make sure you have Python 3.8+ installed. Then run:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## ▶️ Usage

Run the main script:

```bash
python sand_production.py
```

This will:

* Generate synthetic dataset.
* Train & evaluate all models.
* Print performance summary.
* Save dataset and feature importance to CSV.
* Display visualization plots.

---

## 📊 Example Output

**Model Performance Table:**

```
MODEL SUMMARY
============================================================
Model          TrainR² TestR²  RMSE    MAE     CV
Linear         0.812   0.795   32.54   21.87   30.12
RandomForest   0.978   0.912   18.34   12.11   19.02
GradientBoost  0.856   0.823   28.75   18.56   26.44
```

**Feature Importance (Random Forest):**

```
Top Features (RandomForest):
   1. Flow_Rate
   2. Permeability
   3. Drawdown
   4. Reservoir_Pressure
   5. Cohesion
```

**Plots Generated:**

* R² and RMSE bar charts
* Actual vs Predicted scatter plot
* Feature importance (top 10)
* Residuals plot
* Distribution of actual vs predicted sand production

---

## 📂 Output Files

* `sand_data.csv` → full synthetic dataset
* `feature_importance.csv` → top features ranked by RandomForest

---

## 🧠 Tech Stack

* **Python** (3.8+)
* **NumPy** & **Pandas** → data handling
* **Matplotlib** → visualization
* **Scikit-learn** → ML models & metrics

---

## 🚀 Future Enhancements

* Add more advanced ML models (XGBoost, Neural Networks).
* Hyperparameter tuning for better accuracy.
* Real dataset integration from petroleum engineering case studies.
* Deploy as a web app (Flask/Streamlit).

---


