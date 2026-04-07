# Stroke Prediction System

##  Overview

Stroke is a leading cause of death and long-term disability worldwide. Early identification of at-risk individuals can save lives through timely preventive interventions. This project provides an **intelligent, permanent solution** for stroke risk assessment using machine learning.

The system consists of:
- A **Flask backend** with a trained **Linear Discriminant Analysis (LDA)** model.
- An **interactive dashboard** for real-time visualisation of risk distributions and patient history.
- A **SQLite database** that stores every prediction permanently for longitudinal tracking.
- A **REST API** for integration with electronic health records (EHRs).

The application is deployed live at:  
 [https://stroke-prediction-cjdn.onrender.com](https://stroke-prediction-cjdn.onrender.com)

---

##  Features

-  **Real‑time stroke risk prediction** based on 11 clinical/demographic inputs.
-  **Personalised recommendations** (e.g., weight management, smoking cessation, diabetes screening).
-  **Risk level classification** (Very Low, Low, Medium, High, Very High).
-  **Persistent storage** of all predictions with timestamps and patient IDs.
-  **Interactive dashboard** with:
  - Risk distribution over time
  - Gender analysis (female/male risk comparison)
  - Age, BMI, and glucose level distributions
  - Recent predictions table
-  **REST API** for programmatic access.
-  **Handles missing values** (BMI imputed with median) and class imbalance (SMOTE).

---

##  Dataset

The model is trained on the **Stroke Prediction Dataset** from Kaggle:  
[https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

- **Total records:** 5,110
- **Stroke cases:** 249 (4.87%) – imbalanced
- **Features:** gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke (target)

---

##  Model Performance

After preprocessing (median imputation, Yeo‑Johnson scaling, one‑hot encoding) and SMOTE for class balancing, several classifiers were evaluated. **Linear Discriminant Analysis (LDA)** achieved the best cross‑validated performance:

| Model | Mean Accuracy |
|-------|----------------|
| LDA   | 83.74% ± 3.01% |
| Logistic Regression | 83.55% ± 2.90% |
| AdaBoost | 82.20% ± 3.08% |
| Gradient Boosting | 80.00% ± 3.01% |
| Naïve Bayes | 78.73% ± 3.49% |

The LDA model was selected for deployment due to its balance of interpretability and performance.

---

## Installation (Local Development)

### Prerequisites
- Python 3.8+

- pip

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/SeraphineMukabugingo/stroke-prediction.git
   cd stroke-prediction