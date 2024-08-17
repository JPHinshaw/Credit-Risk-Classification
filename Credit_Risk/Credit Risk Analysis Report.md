
# Module 20 Report Template

  

## Overview of the Analysis

  

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

  

* Explain the purpose of the analysis.
-The purpose of the analysis is to determine if the Logistc Regression machine learning model can more accurately predict healthy loans vs. high risk loans using the original dataset or a dataset that is resampled to increase the size of the minority class.
* Explain what financial information the data was on, and what you needed to predict.
- There were 77,536 loans in this dataset. The data includes loan size, interest rate, borrower income, DTI, number of accounts, derogatory marks, total debt and loan status. We are trying to predict "Loan Status". The remaining columns will be used as features to train the data / inform predictions.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).

* Describe the stages of the machine learning process you went through as part of this analysis.
-Prepare the data - Import the file, make the DataFrame, evaluate columns and features

- Separate data into features and labels. The labels are what we are trying to predict. (Healthy (0) or high-risk(1). The features are the remaining data we will use to train / test the model.
- 
-Use train_test_split function to separate the features and labels data into training and testing datasets. 

-Import the machine learning model from SKLearn. (from sklearn.linear_model import LogisticRegression)

-Initiate the model

-Fit the model using the training data

- Use the model to make predictions 

-Evaluate the predictions - Evaluations are done by calculating and comparing metrics like accuracy score, a confusion matrix and a classification report.

**Machine Learning Methods Used**
The primary model used in this analysis is:
-LogisticRegression model from SKLearn

Supporting functions used in this analysis are:

-train_test_split from SKLearn
-Random Oversampler from imblearn

Models are evaluated using the following functions

-confusion_matrix from SKLearn
-classification_matrix from SKLearn
  

## Results

  

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

  

* Machine Learning Model - Logistic Regression:
* **Accuracy Score:** 99%
* **Precision Scores:** 
* Class 0(Healthy Loans): 100%
* Class 1(High-risk Loans): 85%
* **Recall Scores**
	* Class 0(Healthy Loans): 99%
	* Class 1(High-risk Loans): 91%


*Machine Learning Model - Logistic Regression with Oversampled Data:
**Accuracy Score:** 99%
**Precision Scores:**
Class 0(Healthy Loans): 100%
Class 1(High-Risk Loans): 84%
**Recall Scores:**
Class 0(Healthy Loans): 99%
Class 1(High-Risk Loans): 99%

  

## Summary

  

Machine Learning Model 1:
**Question:** How well does the logistic regression model predict both the `0` (healthy loan) and `1` (high-risk loan) labels?

**Answer:** The model predicted healthy loans at a very high accuracy, Precision was 100% and recall was 99%. It predicted unhealthy loans at a reasonably high accuracy. Especially given the disparity between the number of healthy loans vs unhealthy loans in the dataset. 

Machine Learning Model 2:
**Question:** How well does the logistic regression model, fit with oversampled data, predict both the `0` (healthy loan) and `1` (high-risk loan) labels?

**Answer:** The Oversampled Logistic Regression model predicts healthy loans just as accurately as the first model. Precision was 100% and recall was 91%.
The Oversampled Logistic Regression model's precision is 1% lower but it's recall and f1-score are much higher. Overall this model is highly accurate.

* Overall, Model 2 is generally better because it has a higher recall and F1-score for high-risk loans while maintaining reasonably high precision. This suggests that Model 2 would be more effective at identifying high-risk loans. 

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? ) 
* Yes, performance does depend on the problem we are trying to solve. For instance, if we are predicting high-risk loans to minimize financial risk then model 1 might be a better model as it has higher precision at predicting high-risk loans. Both of these models can accurately make predictions.

  


