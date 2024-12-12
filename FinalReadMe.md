# **Stock Price Prediction: Final Milestone** 

![Banner Image](https://img.pikbest.com/wp/202347/stock-market-trend-minimal-trading-graph-flourishing-in-green-3d-render-isolated-on-white-for-data-analysis_9751367.jpg!bw700)  
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

*Prices over time for 5 selected stocks*
![image](https://github.com/user-attachments/assets/253a6534-eee9-44e0-b8a8-2cfbcd11e745)

#### Preprocessing
Our original dataset contains **7,781 observations** and **1,285 features**, with empty values and no scaling/standardization. Pre-processing is a crucial step in making our data usable and effective for the models we have built. We first cleaned our dataset by replacing missing values with column medians with a median imputer, and filtering our data for only numerical features. We also scaled those numerical features through standard and min/max scalers. Additionally, it was crucial to decide which of our extensive set of features to use for predictions, so we employed a correlation matrix, and only used features with a correlation value greater than 0.6. Finally, we employed a standard test/train split of 80:20. 
```
# Select numerical features
numerical_features = df_original.select_dtypes(include=['float64', 'int64']).columns

# Calculate correlation with 'close' price
close_correlation = df_original[numerical_features].corr()['close'].drop('close')
```

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

**Data Exploration Discussion**
The first step in our data exploration was to examine our dataset in order to understand the structure, characteristics, format, and interpret them before we started the preprocessing step. As highlighted in the data exploration section, we first inspected the contents of our dataset through the code snippets provided in the section to find information, descriptions, dataset shape, feature types, and look for missing data. Using this information, we were able to identify the most important features that were essential and peek into the datatypes of each feature to see whether we had to employ encoding or not. These include features covering technical indicators, moving averages, volatility indicators, price range indicators, and other custom metrics. Furthermore, with information about the structure of the dataset, we realized that it had an unnecessary amount of features (1683 observations by 1285 columns) that were about candle patterns. Candle patterns are outside of the scope of our financial knowledge, and via the pairplots seems to have little correlation with the other features. The dataset contained 3 different datatypes: object and float/int64 where one represented the date and the other represented ticker which is the name of the stock company. This enabled us to think about tackling time series and we also noticed our dataset had a significant amount of missing data, both of which would be taken into consideration during preprocessing. 

```
print(df_original.shape)
print(df_original.columns)

Output:
(7781, 1285)
Index(['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker',
       'RSIadjclose15', 'RSIvolume15',
       ...
       'high-15', 'K-15', 'D-15', 'stochastic-k-15', 'stochastic-d-15',
       'stochastic-kd-15', 'volumenrelativo', 'diff', 'INCREMENTO', 'TARGET'],
      dtype='object', length=1285)

```

To understand the importance of the features that we would later take into consideration when filtering, we generated pairplots, histograms, and box plots to visualize our data and their distributions for scaling in preprocessing. Using histograms for every feature, we were able to see their type of distribution, range, skewness, and unusual data patterns which aided us in transforming features that had highly skewed distributions to improve model performance. Box plot visualizations allowed us to see mean, highs, lows, quartiles, and mainly outliers which allowed us to determine which features to scale and normalize in later steps to stabilize model predictions. We originally intended to remove significant or extreme outliers by using box plots but decided against it since those data points still provided significant oversight and information when predicting future prices so we decided to scale/normalize instead to reduce their negative impact on model performance. 

*Correlation heatmap generated in milestone 2*                                                            *A segment of our pairplot genrated in milestone 2*

<img height="350" width="350" alt="Correlation Heatmap" src="https://github.com/user-attachments/assets/50cdfb88-69cd-40ac-94d7-42be71e0b1c7" />


In regards to feature selection, the main factor we took into consideration when looking for features to retain was their correlation to our target feature ‘close’, which represents the closing price of a certain stock. Correlation analysis would help us identify features with a strong linear relationship to our target variable which is represented with a high absolute correlation indicating that the feature can be a strong predictor to be included in our regression models. To do so, we plotted a correlation heat map for every single feature for a visual representation of the relationships among all features and a corresponding correlation matrix that provides exact correlation values that we used in future reference in processing to determine highly correlated features(red/blue on heatmap or close to values -1 or 1) related to our target that we wanted to keep and identify features with low correlation(white on heatmap or close to value 0). Finally, our pairplots between all features allowed us to analyze all of the metrics above by providing another form of visualization to analyze trends, correlation, and outliers while also providing visual insight on the relationships between different features for both deciding which features will contribute to the accuracy of our model while simultaneously informing us of possible transformations we need to impose onto the data. 

**Pre-Processing Discussion**

*Feature selection based on correlation with target ‘close’*

Due to the fact that we had 1285 features, most of which didn’t contribute much to the model due to low correlation, we decided to employ feature selection based on correlation to our target feature ‘close’. As shown in the code snippet above, we used our correlation matrix in our data exploration by filtering for all features that initially had an absolute correlation of 0.9 and above in milestone 3. The reasoning behind this was because selecting strong correlations to our target variable allowed our models to focus on features that were most relevant for our target predictions by reducing the noise. However, in milestone 4, we realized that our polynomial regression model had extreme overfitting and we reasoned that part of the cause could be that the cutoff of 0.9 can possibly lead to redundant features that add no new information or insights. Paired up with our analysis of what each feature represented in our data exploration step, we decided that this was definitely the case since many of our features included repetitive and summarized information so we decided to relax the correlation threshold to 0.6 after testing many different values from 0.5 to 0.9 in 0.1 increments in our milestone 4. In doing so, we were able to emphasize the most important features and significantly reduce dimensionality and computational complexity which was very important for our model runtimes. 

 ```
# Keeping only highly correlated features
high_correlation_features = close_correlation[close_correlation.abs() > 0.9]
df = df_original[['date', 'ticker', 'close'] + high_correlation_features.index.tolist()]
```

*Encoding of ‘date’ and ticker*

Another issue that we had to tackle was the encoding of non float/int64 datatypes which were ‘ticker’ and ‘date’ described earlier. By using steps taking in data exploration, we noticed that due to the nature of our dataset, each individual observation represented information about one particular ‘ticker’, or company, so our dataset could essentially be split into different tickers. For example, we would have a bunch of data/observations for one stock, the same for another, and so on. Thus, encoding it and using it as a feature in our models would lead to horrendous results due to the fact that we are combining data from multiple stocks so we decided to instead run a for loop over every unique ‘ticker’ or take a subset of them due to large amount of ‘tickers’ and generate the models for each stock. In doing so, we would be able to train and test every model for every stock for the most accurate and correct results. In terms of handling the ‘date’ feature, we initially thought ordinal encoding would be a good idea in milestone 3 due to the fact that ‘date’ was time related. Our goal was for the model to train/test itself on the data so that it could be used to predict future prices so having a way to represent ‘date’ in a numerical format was very important and since ordinal encoding captures the sequential nature of the feature. In milestone 4, we were inspired by neural networks, specifically long short term memory models due to its effectiveness in predicting sequential data and decided to switch to a sorted date by converting our ‘date’ feature to datetime instead as a way to represent time steps which fit better for a time-series context since the temporal order of our data is extremely important for accurate predictions. By doing so, we were able to represent the temporal nature of our stock prices significantly better with a better representation of time steps and ensure our models were trained more realistically and accurately. 

*Train-test-split*

For our training and testing, we decided to go with the standard 80:20 train-test-split to ensure that our models are trained with a majority of our data and tested with a large enough sample size to see whether our model generalizes well to unseen data/future prices. As explained previously, the presence of time series posed a problem since we had sequential data. In response, after sorting the dates, we decided to take the first 80% of them and use them as training and the latter 20% for testing which would allow our models to learn from earlier dates and test their performance/predictions on the latter 20% demonstrating its ability to predict future ‘close’ prices. 

*Handling missing data with simple median inputer*

Due to our approach in feature selection, we retained features that had missing data that we had to take into account. To handle this, we decided to look at our pair plots and other figures generated in our data exploration step in order to figure out how to handle them. As displayed above, the pairplots and other figures demonstrated that almost all our features had significant skewness, outliers, and long tailed distributions so instead of taking the mean which could introduce bias, we decided to take the median to make it much less susceptible to outliers. By doing so, we introduced a simple median inputer to handle missing data to stabilize the dataset and make sure that it doesn’t skew or distort any of our predictions. 

*Feature scaling with Standard and MinMax Scaler*

Using similar reasoning, the distribution of our features enabled us to employ feature scaling with Standard and MinMax scalers where we again looked at our data exploration figures to figure out whether to employ Standard or MinMax Scalers to stabilize our data. Looking through our features in data exploration, features such as ‘open’, ‘atr5’, ‘atr10’, etc displayed signs of Gaussian-like distributions so normalizing the features with a standard scaler allowed them to have a mean of 0 and std of 1. For the remaining features we employed a MinMax scaler to ensure that they were scaled to a range of between 0 and 1 inclusive due to the volatile ranges of stock data.

**Model 1 - Polynomial Regression Discussion**

In milestone 3, we decided that our first model should be a form of regression. Since linear regression felt like it was too simple to capture the complexity of our dataset and project, we decided to go with polynomial regression of degree 2. After all pre-processing etc, we looped through every ticker/stock in our company, trained and tested the models, and printed the fitting graphs and evaluation metrics for all tickers afterwards. Upon reviewing the metrics, we realized that we had severe overfitting in terms of several magnitudes. For example, when we trained our model on the ASML ticker, our train RMSE was 0.152 and our test RMSE was 4.411. Realizing our issue, we decided on a different approach in Milestone 4 and decided to employ both hyper parameter tuning and regularization to combat the overfitting. For hyper parameter tuning, we used degree 1, 2, 3, 4, and 5 for polynomial regression and used polynomial feature expansion for each degree in order to find the best parameter and observe how the performance of our models changes depending on the degree. After printing and visualizing the performance metrics, we noticed that a clear trend where higher degree parameters exhibited significantly more overfitting. For example, our evaluation results for the ticker ASML gave us a jump from a test RMSE of 0.7 and train RMSE of 2.5 for degree 1 to a train RMSE of 0.2 to 5.8 which was an enormous jump in magnitude for overfitting and continued to increase as the degree increased. We reasoned that this was caused by the fact that using a higher degree allowed the model to become excessively complex and enable it to fit the training data too well which included noise and random fluctuations which was counterintuitive to our initial expectation that a simple linear regression model would be too simple to capture the complexities. In the end, we decided that the best degree would be of degree 1, or linear regression and although it had the best results out of every parameter we tried, it still showed a significant sign of overfitting. Therefore, we decided to pursue more tuning through Lasso and Ridge regularizations for our second models to hopefully reduce overfitting.

**Model 2 - Ridge And Lasso Regression L1/L2 Discussion**

**Model 3 - Support Vector Regression (SVR) Discussion** 



##  **Conclusion**

In regards to future possible directions, we all collectively agreed that the best model for this project would have been a Long Short Term Model, a type of neural network that can be used to learn and predict sequential data. By the end of the project it was clear the main issues we were dealing with was the sequential nature of the time series and overfitting, which an LSTM or other neural networks could’ve handled better. As explained above, time series and time steps was one issue that baffled us for a while since our project aimed to predict future close prices based on past ones and that heavily relied on how we dealt with the ‘date’ feature we overlooked for a while. The main issue in our project was struggling with overfitting which we initially thought was due to our models but ended up being a problem with the nature of our dataset. After research, we realized that the volatility in stock market data will always lead to some degree of overfitting in any regression, whether it be polynomial, linear, or L1/L2 regression. This is due to the fact that the closing price or ‘close’ feature in real stock markets are severely impacted by outside events and news, like new laws etc, which are qualitative factors that we simple regression models can’t predict or handle. For example, linear regression aims to find a simple pattern or trend in the data to predict future values but our data will only indicate that there seems to be a trend or pattern when in reality, there isn’t. Thus, the use of LSTM neural networks would be the best choice for a model in regards to this project due to its ability to properly handle temporal dependencies through time based cycles, trends, and patterns that go beyond the simplistic nature of regression and its ability to model non-linear and complex relationships that take into account multiple qualitative factors that regression can’t handle such as sentiment, events, geopolitical changes, and etc. 


##  **Statement of Collaboration**


Some Exploratory Data Analysis Images:

![Screenshot 2024-12-11 at 8 36 51 PM](https://github.com/user-attachments/assets/b3518958-109b-48af-a556-9677208ab6dd)

![Screenshot 2024-12-11 at 8 37 01 PM](https://github.com/user-attachments/assets/909eaad8-057a-4013-9b6a-7711ba039f77)

![Screenshot 2024-12-11 at 8 37 14 PM](https://github.com/user-attachments/assets/0d0ceddb-b899-4f0e-b90a-29a0e1a5ac83)



Polynomial Regression:
![Screenshot 2024-12-11 at 8 20 20 PM](https://github.com/user-attachments/assets/b52cb0e6-117b-4e15-917b-8fbe5314e861)

![Screenshot 2024-12-11 at 8 20 32 PM](https://github.com/user-attachments/assets/dfcbb346-3ba6-4243-ab55-d02445eacf6f)

![Screenshot 2024-12-11 at 8 20 45 PM](https://github.com/user-attachments/assets/4fc2ed76-6eb7-4bfb-ad6b-bfcea7ab3085)



Lasso:

![Screenshot 2024-12-11 at 8 21 15 PM](https://github.com/user-attachments/assets/f97cb37f-6508-4c6a-9bba-37575fe21e3b)

![Screenshot 2024-12-11 at 8 21 34 PM](https://github.com/user-attachments/assets/83b63617-0a17-4ead-bbae-6784ccbe5731)

![Screenshot 2024-12-11 at 8 21 59 PM](https://github.com/user-attachments/assets/88e2eb66-e0af-41ea-95ed-873797b2ffb9)



Ridge:
![Screenshot 2024-12-11 at 8 21 24 PM](https://github.com/user-attachments/assets/7fdd039f-850d-42ff-bb75-8d5921744091)

![Screenshot 2024-12-11 at 8 21 45 PM](https://github.com/user-attachments/assets/f04b07ff-00d4-4e00-90c0-1174d94f32d3)

![Screenshot 2024-12-11 at 8 22 16 PM](https://github.com/user-attachments/assets/95583b82-55f0-4b2a-a187-1de556f5a9f1)





SVR:

![Screenshot 2024-12-11 at 8 22 48 PM](https://github.com/user-attachments/assets/7c8ab550-1fbe-4b45-859b-486908f2f521)

![Screenshot 2024-12-11 at 8 22 55 PM](https://github.com/user-attachments/assets/09b4314b-00d2-4d01-8e3e-3824cc7c7ede)

![Screenshot 2024-12-11 at 8 23 05 PM](https://github.com/user-attachments/assets/faa86b44-7e4b-4f69-8bf6-6e90c35cb59b)




