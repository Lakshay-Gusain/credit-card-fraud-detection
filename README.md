# Credit Card Fraud Detection

This project implements a machine learning approach to detect fraudulent credit card transactions using a publicly available dataset. It explores data preprocessing, visualization, and model training using Logistic Regression and Neural Networks, with a focus on handling highly imbalanced data.

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Description:**  
  The dataset contains transactions made by European cardholders in September 2013.  
  - **Features:** 30 (including PCA-transformed features V1–V28, `Time`, and `Amount`)  
  - **Target:** `Class` — 0 for normal transactions, 1 for fraud  
  - **Size:** 284,807 transactions, 492 fraud cases (0.172% fraud rate)

## Project Workflow

1. **Data Loading**  
   - Reads dataset into a pandas DataFrame.
   - Separates features and labels.

2. **Exploratory Data Analysis (EDA)**  
   - Class distribution check.
   - Histograms for transaction time comparison between fraud and normal cases.
   - Boxplots to compare transaction amounts.
   - Statistical summaries.

3. **Data Preprocessing**  
   - Scaling features using `StandardScaler`.
   - Undersampling majority class for balanced training.

4. **Model Training**  
   - **Logistic Regression:** Used as a baseline model.  
   - **Neural Network:** Built with TensorFlow/Keras for improved performance.  
   - Regularization applied to prevent overfitting.

5. **Evaluation Metrics**  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  
   - KS-Statistic for distribution comparison.

## Results

- The Logistic Regression model provides a simple yet effective baseline.
- The Neural Network model improves detection capability for minority fraud cases.
- Class imbalance handling is crucial for realistic performance.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow scipy
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Lakshay-Gusain/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the project folder.
3. Run the notebook:
   ```bash
   jupyter notebook CREDIT_CARD_FRAUD_DETECTION.ipynb
   ```
   Or open it in Google Colab.

## Future Improvements

- Try advanced algorithms like XGBoost, Random Forest, or LightGBM.
- Use SMOTE or ADASYN for synthetic oversampling.
- Deploy as a web API for real-time fraud detection.

## License

This project is licensed under the MIT License.
