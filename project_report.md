<center>

# Assignment 4: Noise Level Binary Classification Using Machine Learning
**Name:** Syed Abed Hossain  
**ID:** DV68018  
**Course:** SENG 691- AI Agent Computing  
**Instructor:** Dr. Mohammad Samarah  
**Institution:** University of Maryland Baltimore County (UMBC)  
**Date:** 11/5/2025  

</center>  

## 1. Report of Collected Data
The dataset used for this project is stored in `sound_data_formatted.csv`. It contains 601 rows of sound data collected from various locations and times. The columns include:
- `timestamp`: Time of data collection (8:30:00 AM).
- `location`: Categorical feature indicating the location (University Center, Commons building, etc.).
- `DB_value`: Numeric feature representing the decibel value (continuous, ranging from 55 to 79).
- `Area_size`: Categorical feature for area size (Large, Medium, Small).
- `Noise_level`: Target variable for binary classification (high or moderate).
- `Location_type`: Categorical feature for location type (Indoor or Outdoor).

The data was collected from multiple sites, with timestamps spanning different times of day. No missing values were present in the key columns. However, preprocessing included imputation strategies.

## 2. Tools Used
- **Programming Language**: Python 3.12.3
- **Libraries**:
  - `pandas`: For data loading and manipulation.
  - `numpy`: For numerical operations.
  - `scikit-learn`: For preprocessing (imputation, encoding, scaling), model training, and evaluation.
- **Environment**: Virtual environment managed via VS Code, with dependencies installed from `requirements.txt`.
- **Other Tools**: VS Code for development, terminal for running scripts.

## 3. Features Engineered
The collected data includes manually gathered columns such as `location` and `DB_value`. The engineered features are `Area_size`, `Noise_level`, and `Location_type`, which went through preprocessing:
- `Area_size` and `Location_type`: Categorical features imputed with most frequent value, then one-hot encoded using OneHotEncoder.
- `Noise_level`: Target variable cleaned (stripped, lowercased) and restricted to "high" or "moderate".

`DB_value` was imputed with median if missing and standardized using StandardScaler. `location` was one-hot encoded as part of the categorical preprocessing pipeline.

## 4. Summary of Classification & Algorithms Used
The dataset was split into training (80%) and testing (20%) sets using stratified sampling for balance.

Four algorithms were trained and evaluated:
- **Logistic Regression**: Linear model with LBFGS solver.
- **Decision Tree**: Tree-based model with default parameters (ccp_alpha=0.0).
- **Random Forest**: Ensemble of 200 trees with sqrt max_features.
- **Linear SVM**: Support Vector Machine with linear kernel.

All models were trained using scikit-learn's Pipeline for consistent preprocessing.

## 5. Performance Comparison and Evaluation Results
Evaluation was performed on the test set using macro averaged metrics to account for class balance. Results are summarized below:

| Model          | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|----------------|----------|-------------------|----------------|------------|
| Logistic Regression | 0.992    | 0.989             | 0.994          | 0.991     |
| Decision Tree  | 1.000    | 1.000             | 1.000          | 1.000     |
| Random Forest  | 1.000    | 1.000             | 1.000          | 1.000     |
| Linear SVM     | 0.992    | 0.989             | 0.994          | 0.991     |

- **Best Performing Models**: Decision Tree and Random Forest achieved perfect scores (1.000) across all metrics, indicating they could perfectly classify the test set. This suggest the data is linearly separable and the features strongly correlate with the target.
- **Comparison**: Logistic Regression and Linear SVM performed nearly perfectly (99.2% accuracy) but slightly lower than tree based models. Tree models captured non linear patterns better.
- **Feature Importance**: For tree models, `DB_value` and one-hot encoded categories (specific locations or area sizes) were key. Linear models showed coefficients aligned with these.
- **Limitations**: Perfect scores on a small dataset indicate overfitting or data simplicity. Real-world generalization should be tested with more data.

## Conclusion
The project successfully implemented a binary classifier for noise levels using standard ML practices. Tree based models outperformed linear ones, achieving perfect test performance. Future work could include cross validation, hyperparameter tuning, or expanding to multi class prediction.