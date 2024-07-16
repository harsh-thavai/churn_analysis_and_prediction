# Customer Churn Analysis and Prediction

## Project Overview
This project delves deep into customer churn analysis for a telecommunications company, leveraging advanced data analysis techniques and machine learning. Our goal is twofold: to accurately predict which customers are at risk of churning and to uncover the key factors driving customer churn. Through extensive Exploratory Data Analysis (EDA) and the application of state-of-the-art machine learning models, we've developed a highly accurate predictive model and gained valuable insights into customer behavior.

## Table of Contents
1. [Key Achievements](#key-achievements)
2. [Data Overview](#data-overview)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Model Development](#model-development)
6. [Results and Insights](#results-and-insights)
7. [Future Work](#future-work)
8. [License](#license)

## Key Achievements
- Developed a high-performance churn prediction model with 94% accuracy and F1 score
- Uncovered critical factors influencing customer churn through comprehensive EDA
- Engineered innovative features to enhance model performance
- Implemented advanced techniques to handle class imbalance, ensuring robust predictions

## Data Overview
Our analysis is based on the Telco Customer Churn dataset, which includes:
- Customer demographics (gender, age, partners, dependents)
- Account information (tenure, contract type, payment method)
- Services used (phone, internet, tech support, etc.)
- Billing information (monthly charges, total charges)
- Churn status

## Exploratory Data Analysis (EDA)
Our EDA revealed several critical insights:

1. **Contract Type Impact**: Month-to-month contracts have significantly higher churn rates compared to long-term contracts.
2. **Service Impact**: Lack of online security and tech support correlates strongly with higher churn rates.
3. **Tenure Influence**: Customers in their first year show higher churn propensity, while those with 5+ years of engagement are more likely to stay.
4. **Payment Method**: Customers using electronic checks as their payment method have higher churn rates.
5. **Additional Services**: Customers with more additional services (e.g., online backup, device protection) are less likely to churn.

## Feature Engineering
To enhance our model's predictive power, we implemented several feature engineering techniques:
- Created a 'tenure_group' feature to categorize customers based on their length of service
- Developed a 'total_services' feature to capture the breadth of services each customer uses
- Engineered an 'average_monthly_charges' feature to normalize charges across different tenure lengths

## Model Development
We employed a rigorous approach to model development:
1. **Data Preprocessing**: Handled missing values, encoded categorical variables, and scaled numerical features.
2. **Class Imbalance**: Utilized SMOTEENN to address the imbalance in churn vs. non-churn customers.
3. **Model Selection**: Tested various algorithms including Decision Trees, Random Forests, and Gradient Boosting.
4. **Hyperparameter Tuning**: Employed grid search with cross-validation to optimize model parameters.
5. **Ensemble Methods**: Leveraged ensemble techniques to further improve model performance.

## Results and Insights

Our final model achieved outstanding performance:
- **Accuracy**: 94%
- **F1 Score**: 94%
- **Precision**: 93%
- **Recall**: 95%

These metrics demonstrate the model's exceptional ability to identify both customers likely to churn and those likely to remain, providing a balanced and reliable prediction tool.

Key factors influencing churn, in order of importance:
1. Contract type (month-to-month being highest risk)
2. Tenure (shorter tenure correlating with higher churn risk)
3. Total charges
4. Monthly charges
5. Lack of online security and tech support
6. Internet service type (fiber optic customers showing higher churn)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
