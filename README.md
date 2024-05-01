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



## Description of the project notebooks.

### 0.0.0. Business understanding
The objective variable of this problem is to estimate the probability that the applicants can pay the loan on time. To address this problem, I built a binary classification model. I used a sample of requests associated with their respective label indicating whether the loan was repaid or the loan was not repaid. In addition, I used a subset of information on the characteristics of the requested credit and the client, such as their historical credit behavior. Finally, I used the predicted score for each application that the model returns to make decisions.
To define how this probability will be used is necessary to determine the business objectives:
* Option 1: We want business rules focused on maximizing the ability to identify bad applicants. But the number of mislabels on good applicants increases (false negative rate). With this approach, we would minimize the credit risk but increase the users' friction by rejecting that they can pay the loan.

* Option 2: We want a business rule where we would minimize the number of mislabels on good applicants (false positive rate). But the ability to identify bad applicants decreases. With this approach, we would minimize the users' friction but increase the credit risk by approving that they cannot pay the loan.

On the other hand, to choose the structure of the model and evaluate its performance, We sought to maximize the ROC AUC. This metric is more related to the evaluation of classifiers based on their efficiency for class separation. We can isolate the modeling process by maximizing accuracy and recovery metrics for different probability thresholds and use score for decision-making. With this, we can choose the threshold that best suits the needs of the business to define whether an application is highly successful or not.

Additionally, one of the characteristics of this problem is the not repaid loan sample is much higher than the repaid loan sample. Therefore, We also evaluated the Precision-Recall AUC (PR AUC), which is a more robust metric to evaluate a classifier in these situations to deal with class imbalance.


### [1.0.0 Build inputs from Previous Internal Applications](https://github.com/MFrancys/credit-risk-machine-learning-pipeline-/blob/build-risk-ml-pipeline/notebook/1.0.0_build_inputs_from_previous_internal_applications.ipynb)
#### - Methodology
This section aims to create informative and meaningful features that capture customers' past interactions with credit products, particularly their Buy Now, Pay Later (BNPL) applications and SF applications. 

Here are some features we explored: 
- Raw Features:
  - Account to Application Days (previous_internal_apps__account_to_application_days): This directly captures the duration from account creation to loan application, providing insights into the customer’s planning or urgency in financial matters.
  - Number of Smartphone Financing Applications (previous_internal_apps__n_sf_apps): Reflects the customer's interest in financing options specifically for smartphones, which can be indicative of their spending habits and preferences.
  - Total BNPL Applications and Approvals:
    - Applications (previous_internal_apps__n_bnpl_apps): Total number of BNPL applications made.
    - Approvals (previous_internal_apps__n_bnpl_approved_apps): Number of BNPL applications that were approved.
  - Credit Inquiries:
    - Last 3 Months (previous_internal_apps__n_inquiries_l3m): Inquiries in the last 3 months.
    - Last 6 Months (previous_internal_apps__n_inquiries_l6m): Inquiries in the last 6 months.

- Derived Features:  
  - BNPL Approval Ratio (previous_internal_apps__ratio_bnpl_approved): The ratio of approved BNPL applications to the total number of BNPL applications (n_bnpl_approved_apps / n_bnpl_apps).
  - Days from Last BNPL Application to Loan Application (previous_internal_apps__last_bnpl_app_to_application_days): The number of days between the date of the last BNPL application and the date of the current loan application (application_datetime - last_bnpl_app_date).
  - Days from First BNPL Application to Loan Application (previous_internal_apps__first_bnpl_app_to_application_days): The number of days between the date of the first BNPL application and the date of the current loan application (application_datetime - first_bnpl_app_date).

### [1.1.0 Build inputs from credit reports dataset](https://github.com/MFrancys/credit-risk-machine-learning-pipeline-/blob/build-risk-ml-pipeline/notebook/1.1.0_build_inputs_from_credit_reports_dataset.ipynb) 

#### - Methodology
The goal of this section is to construct informative and actionable features from the credit reports dataset that encapsulate each customer's credit history effectively. This involves a meticulous aggregation and transformation of credit-related data

Here are some features we explored, overall and by credit time: 
- Raw Features:
    - Total Loans Count (credit_reports__loans_count): Captures the total number of loans associated with each customer, providing a direct measure of credit usage.
    - Maximum Credit Used (credit_reports__max_credit_max): Represents the peak credit amount utilized by the customer, indicating their highest financial leverage or needs.
- Derived Features:
    - Credit Utilization Ratios (credit_reports__debt_ratio): Calculated as the ratio of current balance to credit limit, this metric helps in assessing how much of the available credit is being utilized by the customer.
    - Delayed Payment Indicators (credit_reports__has_delayed_payments): Reflects whether there have been any payments past their due date, which is a critical indicator of potential default risk.
    - Diversity in Credit Types (credit_reports__credit_type_nunique): The count of unique types of credit, which illustrates the variety of credit facilities used by the customer.
    - Age of Credit (credit_reports__age): Measures the duration from the opening to the closing of the credit or to the current date if it's still active, providing insights into the longevity of credit relationships.

### [1.2.0 Build Final Dataset] (https://github.com/MFrancys/credit-risk-machine-learning-pipeline-/blob/build-risk-ml-pipeline/notebook/1.2.0_build_final_dataset.ipynb)

#### Methodology:
The main goal of this stage is to compile the final dataset that merges the target variable with features generated from previous steps, specifically those created by build_previous_internal_app_features and build_aggregate_credit_report_information_by. This comprehensive dataset will be the foundation for all subsequent modeling efforts, such as training and validation of machine learning models to predict customer creditworthiness.

Data Integration:
1. Merge Target Data: Start by merging the dataset containing the target variable (e.g., loan default status) with features derived from the build_previous_internal_app_features function, which provides insights into the customer's past interactions with credit products.
2. Add Aggregated Credit Information: Integrate additional features from the build_aggregate_credit_report_information_by function. This includes detailed credit behavior metrics at the customer level, further enriching the dataset.


### 1.3.0 Target Analysis

#### Methodology
This section of the analysis focuses on examining the target variable, which identifies whether a loan is bad (i.e., the customer was 34 days late or more in 77 days of contract). Understanding patterns in the bad rate is crucial for assessing risk and calibrating the model accordingly.
1. Overall Bad Rate: Calculate the overall default rate by dividing the sum of defaulted loans by the total number of loans.
    Results: Based on the data provided:
    - Total Loans: 14,454
    - Total Defaults: 2,700
    - Overall Default Rate: 18.68%

2. Bad Rate Analysis by Loan Origination Month: To determine how the bad rate varies by the month the loan was originated.
   - Data Segmentation: Group the dataset by the loan_origination_datetime_month to analyze loans based on the month they were disbursed.
   - Rate Calculation: For each group, calculate the total number of loans (LOANS), the number of bad loans (BAD_LOANS), and the bad rate (BAD_RATE as the ratio of BAD_LOANS to LOANS).

3. Bad Rate Analysis by Credit Experience: This study analyzes the default rates among different segments of customers based on their credit experience (customers with no credit history, with less credit experience, and with high credit experience).
   - Customer Segmentation: Classify customers into categories based on their credit experience. The criteria for segmentation (e.g., number of prior loans, credit age) should be clearly defined.
   - Rate Calculation: Calculate the default rate for each category to identify which segment poses higher risks.


#### Conclusions
- Insights from Loan Origination Month Analysis
    - Trends Observed:
        - The bad rate tends to increase as the year progresses, peaking in early 2023 before showing a variable trend.
        - Notable peaks are observed in October 2022, December 2022, and January 2023, suggesting potential seasonality or economic factors impacting default rates (e.g. Buen Fin, Christmas).
    
        <img src="image/default_rate_image.png">
      
- Insights from Credit Experience Analysis
    - It is hypothesized that customers with no or limited credit history might exhibit higher default rates due to unproven creditworthiness. Conversely, those with extensive credit history might show lower default rates if they have a history of good credit management.

    Below is a table showing the number of loans, bad loans, and the bad rate segmented by the count range of credit reports:
    
    | credit_reports__loans_count_range | LOANS | BAD_LOANS | BAD_RATE |
    |-----------------------------------|-------|-----------|----------|
    | (-1, 0]                           | 5282  | 982       | 0.185914 |
    | (0, 5]                            | 983   | 183       | 0.186165 |
    | (5, 10]                           | 1292  | 260       | 0.201238 |
    | (10, 15]                          | 1100  | 233       | 0.211818 |
    | (15, 30]                          | 2433  | 493       | 0.202630 |
    | (30, 300]                         | 3364  | 549       | 0.163199 |


### 2.0.0 - Data Split

#### Methodology

This section details the process of splitting the dataset into training, validation, and testing sets. The dataset was split using a time-based strategy to segregate the data into distinct periods for training, validation, and testing. Furthermore, a spatial segregation approach was employed within the training data to create a validation set.

#### Conclusion:

- Data Split

    - Training Set: Data from July 2022 to February 2023, used for initial model training.
    - Validation Set: A subset of the training set, spatially distinct, used for tuning model parameters and initial evaluation.
    - Testing Set: Data from March 2023 to April 2024, used to simulate model performance on future, unseen data.
- Data Distribution 
    - The split resulted in the following distribution of samples:
    - Training Set: 9479 samples with a default rate (bads) of 19.4%.
    - Validation Set: 1053 samples with a default rate of 19.94%.
    - Testing Set: 3845 samples with a default rate of 16.9%. 
    
  This distribution ensures that each set is representative of the overall data, with the training set encompassing the majority of the data (65.59%), followed by the testing set (26.7%) and the validation set (7.3%).

| Split      | n_samples | bads | Start Date | End Date | Target Distribution | Samples Distribution |
|------------|-----------|------|------------|----------|---------------------|----------------------|
| Train      | 9479      | 1839 | 2022-07    | 2023-02  | 0.194008            | 0.659317             |
| Validation | 1053      | 210  | 2022-07    | 2023-02  | 0.199430            | 0.073242             |
| Test       | 3845      | 650  | 2023-03    | 2023-04  | 0.169051            | 0.267441             |


### 3.0.0. EDA

#### Methodology

As part of our data preparation and understanding phase, we conducted an extensive exploratory data analysis on the training dataset using the `pandas_profiling` package. This tool enables an automated and comprehensive EDA, generating a detailed report that includes:

- Statistics: Descriptive statistics that summarize a dataset's distribution's central tendency, dispersion, and shape.
- Correlations: Analysis of the relationships between features, identifying which pairs have the strongest correlations with the target variable.
- Missing Values: Identifying and visualizing missing data patterns, helping to decide necessary preprocessing steps.
- Distributions: Visualizations of data distributions and variance to understand the skewness and outliers that might influence model performance.