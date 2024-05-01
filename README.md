# Credit Default Risk Machine Learning Pipeline

## Project Overview

This project aims to develop a classifier model to estimate the likelihood of loan applicants repaying their loans on time. The classifier is designed to optimize the credit application process, benefiting both the client and the company by ensuring that viable applicants are approved while reducing the risk of non-payment by denying those less likely to repay.hich covers all the necessary steps to get a prediction from the data, from data processes to prediction acquisition. 

## Objectives
The main goal is to create a situation where:

- Applicants capable of repaying the loan are approved.
- The company minimizes financial losses from non-repayment.

## Approach and Methodology

To develop a robust model, the project was structured into several key phases:

1. Business Understanding: Define the project objectives from a business perspective, translate these into a data science framework, and outline the expected model outcomes.
2. Data Preparation and Understanding: Prepare and explore the dataset to understand the attributes and data structure. This includes cleaning data, handling missing values, and identifying potential predictors.
3. Feature Engineering: Develop and select features that are predictive of the loan repayment ability based on domain knowledge and exploratory data analysis.
4. Modeling:

    - Develop various classification models to predict the likelihood of repayment.
   - Employ techniques for feature selection and hyperparameter tuning to optimize the models.
5. Evaluation: Assess model performance using unseen data, focusing on metrics appropriate for classification problems like ROC AUC and Precision-Recall AUC.
6. Business Rule Definition: Use the model's scoring output to set decision thresholds that align with business objectives regarding risk tolerance and customer experience.

## Project Implementation

To ensure the model is scalable, reproducible, and easily maintainable, I implemented an ML pipeline that automates the journey from data processing to making predictions. This pipeline encapsulates all the steps necessary to prepare the data, execute the model, and generate predictions, facilitating both deployment and future adjustments.

## Evaluation Metrics

- ROC AUC: A primary metric for evaluating the model’s ability to distinguish between the classes effectively.
- Precision-Recall AUC (PR AUC): Given the class imbalance observed with a higher number of non-repaid loans, PR AUC provides a robust measure to evaluate model performance in such scenarios.
- Kolmogorov-Smirnov Statistic (KS): This statistic is used to measure the degree of separation between the distributions of the model's scores for the positive and negative classes. 



## Description of project steps

### 1. Business understanding
The objective variable of this problem is to estimate the probability that the applicants can pay the loan on time. To address this problem, I built a binary classification model. I used a sample of requests associated with their respective label indicating whether the loan was repaid or the loan was not repaid. In addition, I used a subset of information on the characteristics of the requested credit and the client, such as their historical credit behavior. Finally, I used the predicted score for each application that the model returns to make decisions.
To define how this probability will be used is necessary to determine the business objectives:
* Option 1: We want business rules focused on maximizing the ability to identify bad applicants. But the number of mislabels on good applicants increases (false negative rate). With this approach, we would minimize the credit risk but increase the users' friction by rejecting that they can pay the loan.

* Option 2: We want a business rule where we would minimize the number of mislabels on good applicants (false positive rate). But the ability to identify bad applicants decreases. With this approach, we would minimize the users' friction but increase the credit risk by approving that they cannot pay the loan.

On the other hand, to choose the structure of the model and evaluate its performance, We sought to maximize the ROC AUC. This metric is more related to the evaluation of classifiers based on their efficiency for class separation. We can isolate the modeling process by maximizing accuracy and recovery metrics for different probability thresholds and use score for decision-making. With this, we can choose the threshold that best suits the needs of the business to define whether an application is highly successful or not.

Additionally, one of the characteristics of this problem is the not repaid loan sample is much higher than the repaid loan sample. Therefore, We also evaluated the Precision-Recall AUC (PR AUC), which is a more robust metric to evaluate a classifier in these situations to deal with class imbalance.

### 2. Data preparation and featuring Engineering



## 3. Featuring Engineering

I created new variables by calculating statistics on the historical information of the clients' credit behavior, such as the average, max, min, and sum. With this, I passed from a data frame of 116 features to one of around 600.

#### More details in Notebook:  2. Machine Learning Pipeline - Home Credit Default Risk: Feature Engineering

## 4. Modeling
### a. Train baseline model

I trained a base logit regression as a benchmark to measure future progress when applying feature selection and hyperparameter tuning. The results were the following:
        
  - ROC AUC Train: 75.97%
  - ROC AUC Validation: 75.11%
  - PR AUC Train: 23.95%
  - PR AUC Validation: 23.1%

### b. Train other models

    - Random Forest 
        - ROC AUC Train: 100.0%
        - ROC AUC Validation: 70.1%
        - PR AUC Train: 100.0%
        - PR AUC Validation: 19.3%

    - LightGBM
        - ROC AUC Train: 83.5%
        - ROC AUC Validation: 76.67%
        - PR AUC Train: 38.44%
        - PR AUC Validation: 25.8%

Finally, I chose the lightgbm because this model had the highest ROC AUC among the three tested models. Also, this model can deal with null values in the features, and its performance is not affected by the magnitudes of the feature values.

### c. Forward Feature selection
- Get Feature Importance from Shap Values
  <img src="image/shap_values.png">
  - Choose the number of features.
  <img src="image/roc_auc_forward_selection_features.png">
  <img src="image/pr_auc_forward_selection_feature_set.png">
The number of features that optimize the performance of the model is 80.

### d. Hyperparameter Tuning
The following LightGBM hyperparameters were chosen to optimize:
- learning_rate
- num_estimators
- num_leaves
- max_depth
- min_data_in_leaf
- bagging_fraction
- feature_fraction

The best iteration That I got was the following definition:
- number_feature: 70
- model_parameters: 
  - num_estimators: 142
  - learning_rate': 0.1261689397241983
  - num_leaves: 36
  - max_depth: 375
  - min_data_in_leaf: 69
  - bagging_fraction: 0.7812134356069105
  - feature_fraction: 0.6116389811545921
   
 With the following results: 
  - ROC AUC Train: 86.02%
  - ROC AUC Validation: 76.73%
  - PR AUC Train: 37.32%
  - PR AUC Validation: 26.21%


#### More details in Notebook:  3. Machine Learning Pipeline - Home Credit Default Risk: Modeling and Evaluation


## 5. Evaluation on Test Dataset

Final Model:
- number_feature: 70
- model_parameters: 
  - num_estimators: 142
  - learning_rate': 0.1261689397241983
  - num_leaves: 36
  - max_depth: 375
  - min_data_in_leaf: 69
  - bagging_fraction: 0.7812134356069105
  - feature_fraction: 0.6116389811545921
		
 With the following results: 
  - ROC AUC Train: 85.4%
  - ROC AUC Validation: 77.82%
  - PR AUC Train: 36.07%
  - PR AUC Validation: 26.81%

#### More details in Notebook:  3. Machine Learning Pipeline - Home Credit Default Risk: Modeling and Evaluation

## 6. Definition of business rules

Finally, the final decision system based on the predictions of the model must be adequately balanced to identify customers who cannot repay the credit without the system being too strict and incorrectly rejecting those who can pay, affecting the customer experience. Therefore, to decide whether or not to deny a loan, we can use a customer's credit default score assigned by the model and define several thresholds to act based on where that score is.

To increase the default identification rate and ensure a smooth experience for most customers, we defined the following business rules:

- Extreme risk: This rule seeks to correctly identify customers who cannot pay the credit with a very low rate of false positives. Requests identified by this rule may be automatically rejected. Therefore, a false positive rate of less than 5% will be allowed. This rate corresponds to requests whose score is between 0.77 and 1.0.

- High risk: This rule seeks to identify more customers who cannot pay, assuming a higher rate of false positives. The false positive rate tolerance level for this rule equals 20%, and this rate corresponds to requests whose score is between 0.57 and 0.77. But as the number of mislabeled good customers identified by the model increases, customers who fall into this segment must manually go through a more extensive review before being denied.

- Low risk: Applications with a score lower than 0.57 can be automatically approved.

<img src="image/false_positive_rate.png">

<img src="image/business_rule.png">

Based on the set of tests, the number of clients that the model identifies as a potential risk of non-repayment is evaluated, that is, those whose score is greater than 0.57. 
The following results are obtained: 
- Accuracy: 79.12%.
- Recall rate: 58.19%.
- Percentage Amount of credit risk detected: 54.76%
<img src="image/confu_matrix.png">