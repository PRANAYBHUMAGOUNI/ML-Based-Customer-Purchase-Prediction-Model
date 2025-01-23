# ML Based Customer Purchase Prediction Model

## Project Overview
A machine learning project predicting customer purchase behavior using multiple classification algorithms.

## Dataset Description
- **Total Rows**: 10,000
- **Total Columns**: 9

### Columns
1. **customer_id**: Unique customer identifier
2. **age**: Customer's age
3. **gender**: Customer's gender
4. **annual_income**: Customer's annual income
5. **last_visited_days_ago**: Days since last website visit
6. **session_duration**: Time spent on website
7. **pages_visited**: Number of pages browsed
8. **device**: Device used for browsing
9. **purchase**: Target variable (Purchase/No Purchase)

## Sample Data

Below is a sample dataset that provides customer details, session activity, and purchase behavior:

| customer_id | age | gender  | annual_income | last_visited_days_ago | session_duration | pages_visited | device   | purchase |
|-------------|-----|---------|---------------|------------------------|------------------|---------------|----------|----------|
| 1           | 56  | male    |               | 7                      | 17               | 15            | desktop  | 0        |
| 2           | 69  | female  | 47617         | 4                      | 35               | 19            | mobile   | 0        |
| 3           | 46  | male    | 94258         | 30                     |                  | 15            | mobile   | 0        |
| 4           | 32  | female  | 70075         | 19                     | 4                | 12            | mobile   | 0        |
| 5           | 60  | male    | 146998        | 16                     | 51               |               | mobile   | 0        |
| 6           | 25  | male    | 42631         | 8                      | 31               | 16            | desktop  | 1        |
| 7           | 38  | female  | 143120        |                        | 31               | 6             | desktop  | 1        |
| 8           | 56  | male    | 117158        | 24                     | 9                | 20            | tablet   | 0        |
| 9           | 36  | female  | 158955        | 12                     | 31               | 15            | desktop  | 0        |


## Model Performance
- **Logistic Regression**: 48% accuracy
- **Random Forest**: 70% accuracy
- **Gradient Boosting**: 70% accuracy

### Final Model
- **Selected Model**: Random Forest
- **Best Accuracy**: 70%

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Installation
1. Clone the repository
```bash
git clone https://github.com/PRANAYBHUMAGOUNI/ML-Based-Customer-Purchase-Prediction-Model/
```

2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Example Prediction
```python
new_customer = pd.DataFrame({
    'age': [69],
    'gender': ['female'],
    'annual_income': [47617],
    'last_visited_days_ago': [4],
    'session_duration': [35],
    'pages_visited': [19],
    'device': ['mobile']
})

# Prediction
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)
```

### Prediction Result
- **Prediction**: Will not purchase
- **Purchase Probability**: 21.48%

## Feature Engineering
1. **days_since_visit_ratio**: Ratio of days since last visit to session duration
2. **pages_per_minute**: Pages visited per minute
3. **income_age_ratio**: Annual income divided by age
4. **engagement_score**: Combination of pages visited and session duration

## Preprocessing Steps
- Handled missing values
- Encoded categorical variables
- Created engineered features
- Scaled numerical features

## Hyperparameter Tuning
- Used GridSearchCV for Random Forest and Gradient Boosting
- Explored parameters:
  - Number of estimators
  - Max depth
  - Min samples split
  - Learning rate

## Future Improvements
- Collect more diverse data
- Experiment with advanced ensemble methods
- Incorporate more complex feature engineering
- Try deep learning approaches
