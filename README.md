                          					Customer Churn Prediction System

Project Overview:

Customer churn is one of the major challenges faced by subscription-based businesses.Churn occurs when a customer stops using a company’s service. Predicting customer churn helps organizations improve customer retention and reduce revenue loss.
This project uses Machine Learning to predict whether a customer is likely to churn based on customer behavior,subscription details,service usage and satisfaction level.


Problem Statement:

Businesses often lose customers due to poor satisfaction, low service usage, frequent complaints, or short customer tenure. Identifying customers who are likely to leave can help companies take proactive measures to improve customer retention.


Objective:

The objective of this project is to:

		•Generate a realistic customer churn dataset.

		•Perform data preprocessing and exploratory data analysis.

		•Visualize customer behavior patterns.

		•Build a Machine Learning model to predict churn.

		•Evaluate model performance using classification metrics.

		•Create an interactive prediction system for customer churn analysis.


Dataset Information:

Features Used : Feature Description

CustomerID : Unique customer identifier

Age : Customer age

SubscriptionType : Basic,Premium,Enterprise

MonthlyCharges : monthly subscription fee

TenureMonth : Number of months as a customer

SupportTickets : Number of support requests raised

SatisfactionScore : Customer satisfaction rating (1–10)

UsageHoursPerWeek : Weekly service usage

Churn: Target variable (Yes/No)


Dataset Generation Process:

A synthetic dataset containing 3000 customer records was generated using NumPy and Pandas.

Realistic churn behavior was created using the following logic:

	•Low satisfaction increases churn probability.

	•Low weekly usage increases churn probability.

	•High support ticket count increases churn probability.

	•Short customer tenure increases churn probability.

	•Basic subscription customers have a slightly higher churn tendency.

The dataset was exported as a CSV file for further analysis.


Libraries Used:

	•NumPy

	•Pandas

	•Matplotlib

	•Seaborn

	•Scikit-learn


Data Preprocessing:

The following preprocessing steps were performed:
	
	•Missing value checking

	•Duplicate record removal

	•Label encoding of categorical variables

	•Data validation


Exploratory Data Analysis:

EDA was performed to understand customer behavior and churn patterns.

The following analyses were conducted:

	•Churn distribution analysis

	•Subscription-wise churn analysis

	•Satisfaction score analysis

	•Correlation analysis


Visualizations:

	The project includes the following visualizations:

		1.Churn Distribution Pie Chart
	
		2.Churn by Subscription Type Bar Chart

		3.Correlation Heatmap

		4.Monthly Charges Histogram


Machine Learning Model:

	Model Used:
		Random Forest Classifier

	Why Random Forest?
		Random Forest combines multiple decision trees and produces more reliable predictions. It helps reduce overfitting and generally provides strong classification performance.


Results:

	Key Insights;

		•Customers with low satisfaction scores are more likely to churn.

		•Customers with low service usage show higher churn rates.

		•Customers who raise more support tickets are more likely to leave.

		•Customers with shorter tenure tend to churn more frequently.

		•Basic subscription users have a higher churn tendency compared to Premium and Enterprise users.


Installation Instructions:

	Install the required libraries using:

		pip install numpy pandas matplotlib seaborn scikit-learn


Running the Project:
		
		1.Clone the repository.

		2.Open the project folder.

		3.Run the Python file:customer_churn_prediction_system.py

		4.The dataset will be generated automatically.

		5.Visualizations will be displayed.

		6.Model evaluation results will be shown.

		7.Users can enter customer details through the interactive prediction system.


Conclusion:

	This project demonstrates how Machine Learning can be used to predict customer churn and support customer retention strategies.By analyzing customer behavior and service usage patterns,businesses can proactively identify customers at risk of leaving and take corrective actions to improve retention.
