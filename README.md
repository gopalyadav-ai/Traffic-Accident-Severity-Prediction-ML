# Traffic Accident Severity Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning to predict **traffic crash severity** based on structured crash-report data.

The target variable, `crash_type`, is a binary classification problem with two outcomes:

* **Injury and/or Tow Due to Crash**
* **No Injury / Drive Away**

The project compares multiple classification algorithms and evaluates their performance using accuracy, precision, recall, F1-score, and cross-validation.

A key part of this project is preventing **data leakage** by removing injury-related variables that are only known after the crash outcome.

---

## Project Objective

The main objective is to build a Machine Learning model that can predict the severity/type of a traffic crash using information such as:

* Weather conditions
* Lighting conditions
* Roadway surface conditions
* First crash type
* Trafficway type
* Road alignment
* Road defects
* Traffic control device
* Primary contributory cause
* Number of units involved
* Crash hour
* Day of the week
* Crash month

---

## Machine Learning Pipeline

```text
Traffic Report Dataset
        ↓
Data Loading
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Remove Duplicates
        ↓
Remove Data Leakage
        ↓
Feature Encoding
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Hyperparameter Tuning
        ↓
Cross-Validation
        ↓
Feature Importance Analysis
        ↓
Best Model Selection
        ↓
Save Model using Joblib
```

---

## Dataset

**Dataset:** `Traffic Report Dataset.csv`

### Dataset Size

* Original dataset: **209,306 rows × 24 columns**
* Duplicate rows removed: **31**
* Final encoded feature matrix: **209,275 rows × 143 features**

No missing values were found in the dataset.

### Target Variable

`crash_type`

The target contains two classes:

1. Injury and/or Tow Due to Crash
2. No Injury / Drive Away

---

## Data Preprocessing

The following preprocessing steps were performed:

### 1. Duplicate Removal

31 duplicate records were identified and removed.

### 2. Target Encoding

The target variable `crash_type` was converted into numerical form using `LabelEncoder`.

### 3. Date Handling

The raw `crash_date` column was removed because it was not used as a predictive feature.

### 4. Data Leakage Prevention

The following post-crash injury variables were removed:

* `most_severe_injury`
* `injuries_total`
* `injuries_fatal`
* `injuries_incapacitating`
* `injuries_non_incapacitating`
* `injuries_reported_not_evident`
* `injuries_no_indication`

These variables describe injury outcomes that are known after the crash. Including them could allow the model to indirectly see the target outcome during training.

### 5. Categorical Encoding

Categorical variables were converted into numerical features using:

```python
pd.get_dummies(drop_first=True)
```

### 6. Train-Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

A stratified split was used with `random_state=42`.

---

## Models Used

Three Machine Learning algorithms were evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**

Additionally, **Random Forest** was optimized using `GridSearchCV`.

### Hyperparameter Tuning

The best Random Forest parameters were:

```text
max_depth = 10
n_estimators = 50
```

---

## Model Performance

| Model               |   Accuracy | Precision | Recall | F1-Score |
| ------------------- | ---------: | --------: | -----: | -------: |
| Logistic Regression | **74.14%** |      0.74 |   0.74 |     0.74 |
| Decision Tree       |     65.08% |      0.65 |   0.65 |     0.65 |
| Random Forest       |     71.32% |      0.71 |   0.71 |     0.71 |
| Tuned Random Forest |     73.32% |      0.73 |   0.73 |     0.73 |

### Best Model

**Logistic Regression** achieved the highest test accuracy:

**74.14%**

It slightly outperformed the tuned Random Forest, which achieved **73.32%** accuracy.

---

## Logistic Regression — Detailed Performance

| Crash Type                     | Precision | Recall | F1-Score |
| ------------------------------ | --------: | -----: | -------: |
| Injury and/or Tow Due to Crash |      0.71 |   0.70 |     0.70 |
| No Injury / Drive Away         |      0.77 |   0.77 |     0.77 |

### Overall Performance

* **Accuracy:** 74.14%
* **Weighted Precision:** 0.74
* **Weighted Recall:** 0.74
* **Weighted F1-Score:** 0.74

---

## Cross-Validation

5-fold cross-validation was performed on the Logistic Regression model.

Cross-validation scores:

```text
0.7432
0.7376
0.7423
0.7428
0.7405
```

### Cross-Validation Result

**Mean Accuracy: 74.13%**

The low variation between folds indicates that the model's performance was relatively stable across different subsets of the dataset.

---

## Feature Importance Analysis

Feature importance analysis was performed to understand which input variables contributed to the model's predictions.

This step helps identify the most influential factors in predicting traffic crash outcomes and provides better interpretability of the Machine Learning model.

---

## Model Saving

The best-performing Logistic Regression model was saved using `joblib`:

```python
joblib.dump(lr, "traffic_accident_model.pkl")
```

The saved model can later be loaded using:

```python
model = joblib.load("traffic_accident_model.pkl")
```

---

## Technologies Used

### Programming Language

* Python

### Data Processing

* pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* Decision Tree
* Random Forest
* GridSearchCV
* Cross-Validation

### Model Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report
* ROC-AUC

### Model Persistence

* Joblib

---

## Project Structure

```text
Traffic-Accident-Severity-Prediction-ML/
│
├── Traffic Accident Severity Prediction Using Machine Learning.ipynb
└── README.md
```

> The dataset and trained model are not currently included in this repository. The notebook references `Traffic Report Dataset.csv` as the input dataset and saves the trained model as `traffic_accident_model.pkl`.

---

## How to Run the Project

### 1. Install the Required Libraries

Open Command Prompt or a terminal and run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 2. Add the Dataset

Place the following dataset in the same directory as the Jupyter Notebook:

```text
Traffic Report Dataset.csv
```

### 3. Open Jupyter Notebook

Run:

```bash
jupyter notebook
```

Then open:

```text
Traffic Accident Severity Prediction Using Machine Learning.ipynb
```

### 4. Run the Notebook

Run the notebook cells from top to bottom.

The notebook is organized into sequential phases, so later cells depend on variables and preprocessing steps created earlier in the notebook.

---

## Key Highlights

* Worked with **209,306 crash records**
* Removed **31 duplicate records**
* Performed exploratory data analysis and data preprocessing
* Identified and removed **post-crash data leakage**
* Converted categorical variables using one-hot encoding
* Compared **Logistic Regression, Decision Tree, and Random Forest**
* Performed **GridSearchCV hyperparameter tuning**
* Used **5-fold cross-validation**
* Achieved **74.14% test accuracy** with Logistic Regression
* Saved the best-performing model using **Joblib**

---

## Conclusion

Among the evaluated Machine Learning models, **Logistic Regression achieved the best performance with 74.14% test accuracy**.

The tuned Random Forest achieved **73.32%**, while the Decision Tree achieved **65.08%**.

The 5-fold cross-validation result of **74.13% mean accuracy** was very close to the test accuracy, indicating stable model performance.

An important aspect of this project was **data leakage prevention**. Injury-related variables that could reveal the crash outcome were removed before model training, helping create a more realistic prediction setup.

Overall, this project demonstrates an end-to-end Machine Learning workflow covering **data cleaning, exploratory analysis, preprocessing, data leakage prevention, model comparison, hyperparameter tuning, cross-validation, evaluation, and model persistence**.
