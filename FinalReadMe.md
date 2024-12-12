# **Stock Price Prediction: Final Milestone** 

![Banner Image](https://as2.ftcdn.net/v2/jpg/04/01/43/81/1000_F_401438164_xZEh1so5FIVpo3fd71xqgEPdu33dyKZm.jpg)  
*_A journey towards profitable trades._*

---

## 🗂️ **Table of Contents**
1. [Introduction](#Introduction)
2. [Methods Section](#Methods-Section)
3. [Results Section](#Results-Section)
4. [Discussion Section](#Discussion-Section)
5. [Conclusion](#Conclusion)
6. [Statement of Collaboration](#Statement-of-Collaboration)


##  **Introduction**
The stock market is a dynamic system that is influenced by a wide variety of quantitative and qualitative variables. For most investors, analyzing stock data and market trends is a tedious process, and making informed and accurate decisions is often extremely difficult to get right consistently. Our group’s goal is to create a project that considers an extremely thorough dataset of stock market data with an extensive range of features to forecast stock prices accurately and efficiently. 

As students who are fascinated by the complexities of public trading and invest in the market personally, we recognize how stock market predictions have significant personal financial implications. Utilizing advanced tools like machine learning can enable better investment decisions and risk management, and at the very least serve as a comprehensive aid for stock market analysis. It can also help investors feel more confident in their investment decisions, and users can feel confident in their decision making being backed by thorough consideration of real data. 

This project was fascinating because of the overlap of machine learning model development and real-world applications in finance. This project serves as a basis for a broad, yet detailed understanding of the course material while simultaneously enabling us to apply this knowledge to a valuable, real-world application for financial literacy and personal investment. While sentiment analysis and reactions to market trends may yield substantial investment results, this project is a testament to the fact that data-driven decision-making is a reliable, and more importantly, consistent method of generating positive returns in the stock market. 

##  **Methods Section**

#### Data Exploration
The data exploration phase was conducted in two steps by our group. The first was understanding the structure in which the data was stored. Through printing shapes and unqiues of the data we determined that the dataset tracked various stock indicators from January 3rd 2022 to December 30th for 31 unique stocks. Then the second step was exploring the feature types, identifying features with missing data, and creating pairplot maps to highlight correlation between features.

#### Preprocessing
Our original dataset contains **7,781 observations** and **1,285 features**, with empty values and no scaling/standardization. Pre-processing is a crucial step in making our data usable and effective for the models we have built. We first cleaned our dataset by replacing missing values with column medians with a median imputer, and filtering our data for only numerical features. We also scaled those numerical features through standard and min/max scalers. Additionally, it was crucial to decide which of our extensive set of features to use for predictions, so we employed a correlation matrix, and only used features with a correlation value greater than 0.6. Finally, we employed a standard test/train split of 80:20. 

#### Model 1: **Polynomial Regression**
Our first model was polynomial regression, for which we tested several different polynomial degrees. For polynomial regression, we generated models for several different polynomial degrees, starting with linear regression, and going up until a degree of 5. For each degree, we performed polynomial feature expansion to prepare our data, and generated MSE and R^2 values to evaluate performance and overfitting/underfitting analysis. 
```
# Store evaluation results
        model_name = f'Polynomial Degree {degree}'
        if ticker not in evaluation_results:
            evaluation_results[ticker] = {}
        evaluation_results[ticker][model_name] = {
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse,
            "Train MAE": train_mae,
            "Test MAE": test_mae,
            "Train R2": train_r2,
            "Test R2": test_r2
        }
```
#### Model 2: **Lasso and Ridge Regression**
Our second model was built upon the fact that we recognized overfitting in our polynomial regression models, for which the best performance came from degree 1 (linear). We used Ridge and Lasso regression as regularized linear regression models to mitigate overfitting in our linear model. For these models, hyperparameter tuning was important to determine which regularization strengths would be most effective in mitigating overfitting without compromising performance. We tested alpha (regularization strength) values of 0.01, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 1, 10, and 100 to determine optimal results. 

    lasso_alphas = [0.01, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 1, 10, 100]
    ridge_alphas = [0.01, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 1, 10, 100]

#### Model 3: **Support Vector Regression (SVR)**
The third model we built was Support Vector Regression, which uses support vectors to predict continuous stock price values while minimizing errors within a specified tolerance. This model was used given the high dimensionality of our data set, which SVR works well for. For this model, the most critical step was hyperparameter tuning. We had to test our model on different **kernel**, **C**, **epsilon**, and **gamma** settings/values. We tested 72 different hyperparameter combinations and found that the most optimal hyperparameter combinations were (linear, 0.1, 0.01, and auto) and (linear, 0.1, 0.01, and scale) respectively. 

*CSV file created to analyze hyper parameter combinations and outputs*
<img height="500" alt="SVR Hyper Perameter Sheet" src="https://github.com/user-attachments/assets/36f9ed42-06dd-4385-baea-fc9d64e179a6" />

##  **Results Section**

##  **Discussion Section**
# Method Discussion # 
The first step in our data exploration was to examine our dataset in order to understand the structure, characteristics, format, and interpret them before we started the preprocessing step. As highlighted in the data exploration section, we first inspected the contents of our dataset through the code snippets provided in the section to find information, descriptions, dataset shape, feature types, and look for missing data. Using this information, we were able to identify the most important features that were essential and peek into the datatypes of each feature to see whether we had to employ encoding or not. These include features covering technical indicators, moving averages, volatility indicators, price range indicators, and other custom metrics. Furthermore, with information about the structure of the dataset, we realized that it had an unnecessary amount of features (1683 observations by 1285 columns) that were about candle patterns. Candle patterns are outside of the scope of our financial knowledge, and via the pairplots seems to have little correlation with the other features. The dataset contained 3 different datatypes: object and float/int64 where one represented the date and the other represented ticker which is the name of the stock company. This enabled us to think about tackling time series and we also noticed our dataset had a significant amount of missing data, both of which would be taken into consideration during preprocessing. 

To understand the importance of the features that we would later take into consideration when filtering, we generated pairplots, histograms, and box plots to visualize our data and their distributions for scaling in preprocessing. Using histograms for every feature, we were able to see their type of distribution, range, skewness, and unusual data patterns which aided us in transforming features that had highly skewed distributions to improve model performance. Box plot visualizations allowed us to see mean, highs, lows, quartiles, and mainly outliers which allowed us to determine which features to scale and normalize in later steps to stabilize model predictions. We originally intended to remove significant or extreme outliers by using box plots but decided against it since those data points still provided significant oversight and information when predicting future prices so we decided to scale/normalize instead to reduce their negative impact on model performance. 

In regards to feature selection, the main factor we took into consideration when looking for features to retain was their correlation to our target feature ‘close’, which represents the closing price of a certain stock. Correlation analysis would help us identify features with a strong linear relationship to our target variable which is represented with a high absolute correlation indicating that the feature can be a strong predictor to be included in our regression models. To do so, we plotted a correlation heat map for every single feature for a visual representation of the relationships among all features and a corresponding correlation matrix that provides exact correlation values that we used in future reference in processing to determine highly correlated features(red/blue on heatmap or close to values -1 or 1) related to our target that we wanted to keep and identify features with low correlation(white on heatmap or close to value 0). Finally, our pairplots between all features allowed us to analyze all of the metrics above by providing another form of visualization to analyze trends, correlation, and outliers while also providing visual insight on the relationships between different features for both deciding which features will contribute to the accuracy of our model while simultaneously informing us of possible transformations we need to impose onto the data. 


##  **Conclusion**

##  **Statement of Collaboration**
