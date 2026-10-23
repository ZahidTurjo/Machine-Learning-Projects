# Obesity Level Prediction Using Machine Learning

## 📌 Project Overview

This project builds a machine learning classification system to predict obesity levels based on individuals' eating habits, physical activity, and other lifestyle-related features.

The project follows an end-to-end machine learning workflow, including data exploration, preprocessing, model comparison, cross-validation, hyperparameter tuning, evaluation, and model serialization.

**Problem Type:** Multiclass Classification

**Best Candidate Model:** Gradient Boosting Classifier

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib

**Environment:** Google Colab / Jupyter Notebook

## 🎯 Objectives

- Explore and understand the dataset.
- Identify missing values and duplicate records.
- Separate numerical and categorical features.
- Build a preprocessing pipeline.
- Train and compare multiple machine learning algorithms.
- Evaluate models using Accuracy, Precision, Recall, and F1-score.
- Apply Stratified K-Fold Cross-Validation.
- Tune hyperparameters using GridSearchCV.
- Evaluate the final model on a held-out test set.
- Save and reload the trained model for new predictions.

## 📂 Dataset Link:https://www.kaggle.com/datasets/ezzaldeenesmail/obesitydataset-raw-and-data-sinthetic


The original dataset contains 2,111 rows and 17 columns, including 16 input features and one target variable.

After removing 24 duplicate rows, the dataset contains 2,087 rows.

### Target Variable

`NObeyesdad`

The target contains seven obesity-related categories:

1. Insufficient_Weight
2. Normal_Weight
3. Overweight_Level_I
4. Overweight_Level_II
5. Obesity_Type_I
6. Obesity_Type_II
7. Obesity_Type_III

### Feature Description

| Feature | Description |
|---|---|
| Gender | Gender category |
| Age | Age of the individual |
| Height | Height in meters |
| Weight | Weight in kilograms |
| family_history_with_overweight | Family history of overweight |
| FAVC | Frequent consumption of high-calorie food |
| FCVC | Frequency of vegetable consumption |
| NCP | Number of main meals |
| CAEC | Consumption of food between meals |
| SMOKE | Smoking status |
| CH2O | Daily water consumption |
| SCC | Monitoring of calorie consumption |
| FAF | Physical activity frequency |
| TUE | Time using technological devices |
| CALC | Alcohol consumption frequency |
| MTRANS | Main mode of transportation |
| NObeyesdad | Target obesity category |

## 🛠️ Technologies Used

- **Python:** Main programming language
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical operations
- **Matplotlib:** Data visualization
- **Seaborn:** Statistical visualization
- **Scikit-learn:** Preprocessing, model training, validation, and evaluation
- **Joblib:** Saving and loading the trained model

## 🔄 Machine Learning Workflow

### 1. Data Loading and Understanding

- Load the CSV dataset into a Pandas DataFrame.
- Inspect the dataset shape, columns, data types, and statistical summary.

### 2. Data Cleaning

- Check for missing values.
- Identify duplicate records.
- Remove duplicate rows and reset the index.

### 3. Exploratory Data Analysis

- Examine the distribution of the target classes.
- Explore numerical and categorical features.
- Visualize the target distribution using Seaborn.

### 4. Feature and Target Separation

Separate the input features and target variable:

- `X`: 16 input features
- `y`: Target variable, `NObeyesdad`

### 5. Train-Test Split

Use an 80:20 train-test split with stratification.

- Training set: 1,669 samples
- Test set: 418 samples
- Random state: 42

Stratification helps preserve class proportions in the training and test sets.

### 6. Data Preprocessing

A Scikit-learn preprocessing pipeline is used.

**Numerical features**
- Apply `StandardScaler`.

**Categorical features**
- Apply `OneHotEncoder`.
- Use `handle_unknown="ignore"` to handle previously unseen categories.

`ColumnTransformer` combines both transformations, while `Pipeline` connects preprocessing with the classifier.

### 7. Model Training and Comparison

Five classification algorithms are compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. K-Nearest Neighbors (KNN)

The models are evaluated using:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score

### 8. Cross-Validation

Use `StratifiedKFold` with five folds to estimate model performance across multiple validation splits.

The mean scores are used to compare the candidate models.

### 9. Hyperparameter Tuning

`GridSearchCV` is used to tune the Gradient Boosting Classifier.

Parameters explored:

| Hyperparameter | Values |
|---|---|
| `n_estimators` | 100, 200 |
| `learning_rate` | 0.05, 0.1 |
| `max_depth` | 1, 2, 3 |
| `min_samples_leaf` | 1, 2 |

The search evaluates 24 parameter combinations using five-fold cross-validation, resulting in 120 model fits.

The selection metric is weighted F1-score.

### 10. Final Evaluation

Evaluate the tuned model using the held-out test set.

Metrics and visualizations include:

- Classification report
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### 11. Model Saving and Prediction

Save the complete fitted pipeline using Joblib.

The saved pipeline includes preprocessing and the trained classifier, allowing new input data to be processed consistently.

The project also demonstrates class prediction and prediction probabilities.

## 📊 Model Comparison Results

The following results were obtained from the notebook's initial held-out test-set evaluation.

| Model | Accuracy | Weighted F1-score |
|---|---:|---:|
| Gradient Boosting | 97.37% | 97.37% |
| Random Forest | 94.02% | 94.12% |
| Decision Tree | 92.58% | 92.55% |
| Logistic Regression | 89.95% | 89.78% |
| KNN | 82.78% | 81.36% |

### Five-Fold Cross-Validation Results

| Model | Mean Accuracy | Mean Weighted F1 |
|---|---:|---:|
| Gradient Boosting | 96.46% | 96.45% |
| Random Forest | 94.20% | 94.27% |
| Decision Tree | 93.58% | 93.57% |
| Logistic Regression | 87.40% | 87.24% |
| KNN | 83.18% | 81.80% |

These results identify Gradient Boosting as the strongest candidate among the five models evaluated in this notebook.

## ⚙️ Hyperparameter Tuning Results

The best parameter combination returned by GridSearchCV was:

```python
{
    "classifier__learning_rate": 0.1,
    "classifier__max_depth": 3,
    "classifier__min_samples_leaf": 1,
    "classifier__n_estimators": 200
}
```

**Best GridSearchCV weighted F1-score:** 95.79%

This score is the mean cross-validation score obtained during hyperparameter tuning on the training set.

## 🏆 Final Model Performance

The tuned Gradient Boosting pipeline achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 98% |
| Macro F1-score | 98% |
| Weighted F1-score | 98% |

The notebook's classification report shows strong performance across all seven target categories.



## 💾 Save and Load the Model

Save the trained pipeline:

```python
import joblib

joblib.dump(
    final_model,
    "obesity_classification_model.pkl"
)
```

Load the saved pipeline:

```python
loaded_model = joblib.load(
    "obesity_classification_model.pkl"
)
```

Make a prediction:

```python
prediction = loaded_model.predict(row)

print("Predicted class:", prediction[0])
```

Obtain prediction probabilities:

```python
probabilities = loaded_model.predict_proba(row)

print(probabilities)
```

The saved model file is named `obesity_classification_model.pkl`.

## 🚀 How to Run the Project

1. Download the dataset from Kaggle.
2. Open Google Colab.
3. Upload the notebook or create a new notebook.
4. Upload `ObesityDataSet_raw_and_data_sinthetic.csv`.
5. Update the CSV path if necessary.
6. Run the notebook cells in order.
7. Review the model comparison and cross-validation results.
8. Run hyperparameter tuning and final evaluation.
9. Save the trained model using Joblib.

Install the required libraries if needed:

```python
%pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

The dataset link are given in the notebook

## 🔍 Key Learnings

- Working with numerical and categorical data.
- Handling duplicate records and inspecting missing values.
- Building reusable preprocessing pipelines.
- Comparing different classification algorithms.
- Understanding Stratified K-Fold Cross-Validation.
- Selecting models using weighted F1-score.
- Performing hyperparameter tuning with GridSearchCV.
- Evaluating multiclass predictions using a classification report and confusion matrix.
- Saving and reusing a complete machine learning pipeline.

## ⚠️ Important Evaluation Note

The notebook calculates cross-validation scores on the full cleaned dataset after creating a train-test split. For a strictly held-out final test evaluation, cross-validation and model selection should use only the training set; the test set should remain untouched until final evaluation.

For future experiments, use the training data for model comparison and cross-validation, then evaluate the selected and tuned pipeline on the test set only once.

## 👨‍💻 Author

**Zahidul Islam Turjo**

- GitHub: [ZahidTurjo](https://github.com/ZahidTurjo)
- LinkedIn: [Zahidul Islam Turjo](https://www.linkedin.com/in/zahidul-islam-turjo-609549320/)
- YouTube: [@zahidul_turjo](https://www.youtube.com/@zahidul_turjo)

---


