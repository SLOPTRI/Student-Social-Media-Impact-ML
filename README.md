# 📱 Early Detection: Impact of Social Media on Student Mental Health

## 📌 Project Overview

This **Machine Learning (Multiclass Classification)** project aims to predict whether a student's social media usage will have a **Negative, Neutral, or Beneficial** impact on their life.

The model is designed as a **preventive tool** for school counselors, universities, and psychologists. Its goal is to identify students at risk of experiencing a drop in academic performance or a deterioration in mental health, allowing for early intervention before the issue escalates.

## 📊 Dataset

* **Source:** [Impact of Social Media on Life - Kaggle](https://www.kaggle.com/datasets/harishyadav0506/impact-of-social-media-on-life/data)
* **Volume:** 4,500 records and 16 features.
* **Target Variable:** `Overall_Impact`
* **Key Features:**
  * `Academic_Performance_GPA` (Academic standing)
  * `Sleep_Duration_Hours` & `Sleep_Quality_Score` (Sleep metrics)
  * `Mental_Health_Index` (Psychological well-being)
  * `Daily_Usage_Hours` (Daily screen time)
  * `Primary_Platform` (Most used social network)

## 🛠️ Tech Stack

* **Language:** Python 3
* **Data Manipulation:** Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (`Pipelines`, `ColumnTransformer`, `GridSearchCV`, `RandomForestClassifier`, `LogisticRegression`)

## ⚙️ Workflow & Methodology

### 1. Exploratory Data Analysis (EDA)
Distributions and correlations were analyzed to understand the underlying patterns. Findings confirmed that lower daily social media usage correlates with both increased sleep duration and higher sleep quality. Furthermore, a **severe class imbalance** was detected: the vast majority of records belonged to the *Beneficial* class (736 in the test set) compared to a clear minority in the *Negative* class (33 in the test set).

### 2. Robust Preprocessing (Pipelines)
To ensure scalability and prevent data leakage, all data processing was encapsulated using `scikit-learn Pipelines`:
* **Feature Engineering:** Creation of a combined `Sleep_Score` variable (Duration $\times$ Quality).
* **Imputation & Scaling:** Handling missing values via median imputation and normalizing numerical variables using `MinMaxScaler`.
* **Encoding:** Transformation of ordinal variables (`OrdinalEncoder`) and nominal variables (`OneHotEncoder`).

## 🧠 Analytical Criteria & Key Decisions

1. **Strategic Outlier Management:**
   During the EDA, extreme values were detected in variables such as the mental health index and daily usage hours. A conscious decision was made **not to remove or force-normalize them**. In the context of this specific problem, these data points are not measurement errors; they represent **real critical profiles** (e.g., students with severe insomnia or depression). Removing these outliers would have destroyed valuable information regarding the minority class that the model specifically aims to predict.

2. **Handling Class Imbalance:**
   Cross-validation was implemented using the `f1_macro` scoring metric, and class weights (`class_weight='balanced'`) were applied directly within the algorithms to force the models to heavily penalize errors made on the minority classes.

## 📈 Results & Model Selection

Two primary models were trained and optimized via `GridSearchCV`: *Random Forest Classifier* and *Logistic Regression*.

| Model | Global F1-Score | Recall (Accuracy on 'Negative' class) | 
| ----- | ----- | ----- | 
| Random Forest | 81% | 72.7% | 
| **Logistic Regression** | **84%** | **87.9%** | 

**Final Decision:**
**Logistic Regression** was selected as the final model. While both models might perform similarly in a perfectly balanced scenario, in the context of psychological prevention, **the most costly error is a false negative** (failing to detect an at-risk student). Logistic Regression demonstrated a vastly superior ability to detect critical cases, perfectly aligning with the business objective of the project.

## 🚀 Next Steps

* **Synthetic Balancing:** Apply oversampling techniques such as SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples for the *Negative* and *Neutral* classes.
* **Data Expansion:** Train the model on a larger volume of real-world data to improve generalization and stabilize global metrics.

---
*Developed by Salvador Lopez Trigueros / SLOPTRI - [Linkedin](https://www.linkedin.com/in/salvador-l%C3%B3pez-trigueros-dev/)*