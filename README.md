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

## Milestones(Prior Submissions):
1. [Milestone 2](https://github.com/d-patravali/CSE151A_Project_2024/blob/main/CSE_151A_Project_Milestone_2-8.ipynb)
2. [Milestone 3](https://github.com/d-patravali/CSE151A_Project_2024/blob/main/CSE_151A_Project_Milestone_3%20final.ipynb)
3. [Milestone 4](https://github.com/d-patravali/CSE151A_Project_2024/blob/main/CSE_151A_Finalized_Milestone_4.ipynb)
4. [Final Milestone(Jupyter Notebook)](https://colab.research.google.com/drive/1KaqfwRywSC7gcmULAiPsagVZ70-M_kUB?usp=sharing)

##  **Introduction**
The stock market is a dynamic system that is influenced by a wide variety of quantitative and qualitative variables. For most investors, analyzing stock data and market trends is a tedious process, and making informed and accurate decisions is often extremely difficult to get right consistently. Our group’s goal is to create a project that considers an extremely thorough dataset of stock market data with an extensive range of features to forecast stock prices accurately and efficiently. 

As students who are fascinated by the complexities of public trading and invest in the market personally, we recognize how stock market predictions have significant personal financial implications. Utilizing advanced tools like machine learning can enable better investment decisions and risk management, and at the very least serve as a comprehensive aid for stock market analysis. It can also help investors feel more confident in their investment decisions, and users can feel confident in their decision making being backed by thorough consideration of real data. 

This project was fascinating because of the overlap of machine learning model development and real-world applications in finance. This project serves as a basis for a broad, yet detailed understanding of the course material while simultaneously enabling us to apply this knowledge to a valuable, real-world application for financial literacy and personal investment. While sentiment analysis and reactions to market trends may yield substantial investment results, this project is a testament to the fact that data-driven decision-making is a reliable, and more importantly, consistent method of generating positive returns in the stock market. 

[Back to Table of Contents](#%EF%B8%8F-table-of-contents)

##  **Methods Section**

#### Data Exploration
The data exploration phase was conducted in two steps by our group. The first was understanding the structure in which the data was stored. Through printing shapes and unqiues of the data we determined that the dataset tracked various stock indicators from January 3rd 2022 to December 30th for 31 unique stocks. Then the second step was exploring the feature types, identifying features with missing data, and creating pairplot maps to highlight correlation between features.

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

*Closing Prices of All Stocks*
![Screenshot 2024-12-11 at 8 37 01 PM](https://github.com/user-attachments/assets/909eaad8-057a-4013-9b6a-7711ba039f77)

<div align="center">
<img height="450" width="450" alt="Correlation Heatmap" src="https://github.com/user-attachments/assets/50cdfb88-69cd-40ac-94d7-42be71e0b1c7" />
</div>




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

    lasso_alphas = [0.01, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 1, 10, 100, 200]
    ridge_alphas = [0.01, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 1, 10, 100, 200]

#### Model 3: **Support Vector Regression (SVR)**
The third model and final model we built was Support Vector Regression, which uses support vectors to predict continuous stock price values while minimizing errors within a specified tolerance. This model was used given the high dimensionality of our data set, which SVR works well for. For this model, the most critical step was hyperparameter tuning. We had to test our model on different **kernel**, **C**, **epsilon**, and **gamma** settings/values. We tested 72 different hyperparameter combinations and found that the most optimal hyperparameter combinations were (linear, 0.1, 0.01, and auto) and (linear, 0.1, 0.01, and scale) respectively. 

*CSV file created to analyze hyper parameter combinations and outputs*
<img height="500" alt="SVR Hyper Perameter Sheet" src="https://github.com/user-attachments/assets/fd529b18-9939-464d-806b-6ee1fff13bda" />


[Back to Table of Contents](#%EF%B8%8F-table-of-contents)

##  **Results Section**
 
#### **Model 1: Polynomial Regression**
For polynomial regression, the main analysis that went into finding our optimal results was figuring out which degree of polynomial yielded the best results for our data. We had clear results, showing that a polynomial degree of 1 (linear) was most optimal for modelling our stock price data. As the polynomial degree went up, we saw a significant increase in overfitting, as train RMSE was driven towards 0, while test RMSE increased. For polynomial degree 1 (linear regression) we saw a train RMSE of 0.2457 and test RMSE of 0.8667. Here we can see that regardless of the fact that the test RMSE is still quite good for our data range for stock prices (in the hundreds of dollars per share), that there is significant overfitting. From the graphs, we can see that the model still closely predicts the ‘close’ price with a high degree of accuracy, but this overfitting issue led us towards our next models, which were chosen to address this concern.

![Screenshot 2024-12-11 at 8 20 20 PM](https://github.com/user-attachments/assets/b52cb0e6-117b-4e15-917b-8fbe5314e861)


#### **Model 2: Lasso and Ridge Regression**
For Lasso and Ridge Regression, we were able to use regularization to address the overfitting concerns with our polynomial degree 1 (linear) regression from our previous model. As stated, we used hyperparameter tuning to test different alpha values (regularization strengths) to find our optimal setting, which yielded the following results. We found that increasing the regularization strength caused a significant tradeoff between error (RMSE) and overfitting. What was more obvious was our optimal Ridge regression. We saw that an changing the regularization strength did not do much for mitigating the overfitting, so by focusing on RMSE, we found an optimal alpha value of 0.08. This showed a train RMSE of 0.157 and a test RMSE of 0.382, which shows extremely accurate predictions for both train and test data. However, while the overfitting is less than linear regression, we can still see it is still too high to consider successful. With Lasso regression, we saw more interesting results, as we got noticeably lower overfitting as we increased regularization strength, down to a percent difference between train and test RMSE of under 50%, which was a significant improvement from Ridge and regular linear regression. However, the regularization strength of 100+ generalized too much, and even though the RMSE was still relatively low at under 2.0, it was much higher than we were able to achieve through other models because it dropped too many weights to 0 and essentially oversimplified. This led us to our next model, which allowed us to model complex relationships, while still regularizing and handling outliers/noise.  

Lasso:


![Screenshot 2024-12-11 at 10 23 36 PM](https://github.com/user-attachments/assets/18c52f66-8ac1-4426-ac8f-474dded578c6)

Ridge:

![Screenshot 2024-12-11 at 10 23 46 PM](https://github.com/user-attachments/assets/ad53c138-d3ac-43d0-a203-3ba24b6b5569)

#### **Model 3: Support Vector Regression**
Support Vector Regression is the model in which we saw the best results. With relatively no overfitting, as well as low RMSE and high R^2 values for both test and train sets, support vector regression was the most successful model in stock market ‘close’ price predictions. The key step for SVR, as stated earlier, was the emphasis on hyperparameter tuning, to ensure that we were using the best parameters for optimal results. We tested 72 hyperparameter combinations, and found that the optimal combination of Kernel, C, Epsilon, and Gamma settings/values were linear, 1.0, 0.01, and scale, respectively. From this combination of hyperparameters, we observed an average train RMSE of 0.206 and an average test RMSE of 0.231, showing a percent difference of only 11%, indicating an extremely low degree of overfitting. The model showed R^2 values of over .95 for both Train and Test data as well, showing that the model fits the data very well. This showed that we had an extremely low RMSE, meaning our model gave incredibly accurate results, while also having almost no overfitting, which was exactly what our project aimed to do. 

![Screenshot 2024-12-11 at 8 23 05 PM](https://github.com/user-attachments/assets/faa86b44-7e4b-4f69-8bf6-6e90c35cb59b)

##  **Discussion Section**

**Data Exploration Discussion**
The first step in our data exploration was to examine our dataset in order to understand the structure, characteristics, format, and interpret them before we started the preprocessing step. As highlighted in the data exploration section, we first inspected the contents of our dataset through the code snippets provided in the section to find information, descriptions, dataset shape, feature types, and look for missing data. Using this information, we were able to identify the most important features that were essential and peek into the datatypes of each feature to see whether we had to employ encoding or not. These include features covering technical indicators, moving averages, volatility indicators, price range indicators, and other custom metrics. Furthermore, with information about the structure of the dataset, we realized that it had an unnecessary amount of features (1683 observations by 1285 columns) that were about candle patterns. Candle patterns are outside of the scope of our financial knowledge, and via the pairplots seems to have little correlation with the other features. The dataset contained 3 different datatypes: object and float/int64 where one represented the date and the other represented ticker which is the name of the stock company. This enabled us to think about tackling time series and we also noticed our dataset had a significant amount of missing data, both of which would be taken into consideration during preprocessing. 



To understand the importance of the features that we would later take into consideration when filtering, we generated pairplots, histograms, and box plots to visualize our data and their distributions for scaling in preprocessing. Using histograms for every feature, we were able to see their type of distribution, range, skewness, and unusual data patterns which aided us in transforming features that had highly skewed distributions to improve model performance. Box plot visualizations allowed us to see mean, highs, lows, quartiles, and mainly outliers which allowed us to determine which features to scale and normalize in later steps to stabilize model predictions. We originally intended to remove significant or extreme outliers by using box plots but decided against it since those data points still provided significant oversight and information when predicting future prices so we decided to scale/normalize instead to reduce their negative impact on model performance. 

*Fig 1: Correlation heatmap generated in milestone 2*                                                            
*Fig 2: A segment of our pairplot genrated in milestone 2*


In regards to feature selection, the main factor we took into consideration when looking for features to retain was their correlation to our target feature ‘close’, which represents the closing price of a certain stock. Correlation analysis would help us identify features with a strong linear relationship to our target variable which is represented with a high absolute correlation indicating that the feature can be a strong predictor to be included in our regression models. To do so, we plotted a correlation heat map for every single feature for a visual representation of the relationships among all features and a corresponding correlation matrix that provides exact correlation values that we used in future reference in processing to determine highly correlated features(red/blue on heatmap or close to values -1 or 1) related to our target that we wanted to keep and identify features with low correlation(white on heatmap or close to value 0). Finally, our pairplots between all features allowed us to analyze all of the metrics above by providing another form of visualization to analyze trends, correlation, and outliers while also providing visual insight on the relationships between different features for both deciding which features will contribute to the accuracy of our model while simultaneously informing us of possible transformations we need to impose onto the data. 


[Back to Table of Contents](#%EF%B8%8F-table-of-contents)

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
```
# Handle missing data
median_imputer = SimpleImputer(strategy='median')
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = median_imputer.fit_transform(df[num_cols])
```

*Feature scaling with Standard and MinMax Scaler*

Using similar reasoning, the distribution of our features enabled us to employ feature scaling with Standard and MinMax scalers where we again looked at our data exploration figures to figure out whether to employ Standard or MinMax Scalers to stabilize our data. Looking through our features in data exploration, features such as ‘open’, ‘atr5’, ‘atr10’, etc displayed signs of Gaussian-like distributions so normalizing the features with a standard scaler allowed them to have a mean of 0 and std of 1. For the remaining features we employed a MinMax scaler to ensure that they were scaled to a range of between 0 and 1 inclusive due to the volatile ranges of stock data.
```
# Feature Scaling
# Define the features to be scaled
standard = ['open', 'vwapadjclosevolume', 'atr5', 'atr10', 'atr15', 'atr20']
min_max = ['open', 'high', 'low', 'low-5', 'high-5', 'low-10', 'high-10', 'low-15', 'high-15']
...
df[standard] = stdScaler.fit_transform(df[standard])
df[min_max] = minMaxScaler.fit_transform(df[min_max])
```


**Model 1 - Polynomial Regression Discussion**

In milestone 3, we decided that our first model should be a form of regression. Since linear regression felt like it was too simple to capture the complexity of our dataset and project, we decided to go with polynomial regression of degree 2. After all pre-processing etc, we looped through every ticker/stock in our company, trained and tested the models, and printed the fitting graphs and evaluation metrics for all tickers afterwards. Upon reviewing the metrics, we realized that we had severe overfitting in terms of several magnitudes. For example, when we trained our model on the ASML ticker, our train RMSE was 0.152 and our test RMSE was 4.411. Realizing our issue, we decided on a different approach in Milestone 4 and decided to employ both hyper parameter tuning and regularization to combat the overfitting. For hyper parameter tuning, we used degree 1, 2, 3, 4, and 5 for polynomial regression and used polynomial feature expansion for each degree in order to find the best parameter and observe how the performance of our models changes depending on the degree. After printing and visualizing the performance metrics, we noticed that a clear trend where higher degree parameters exhibited significantly more overfitting. For example, our evaluation results for the ticker ASML gave us a jump from a test RMSE of 0.7 and train RMSE of 2.5 for degree 1 to a train RMSE of 0.2 to 5.8 which was an enormous jump in magnitude for overfitting and continued to increase as the degree increased. We reasoned that this was caused by the fact that using a higher degree allowed the model to become excessively complex and enable it to fit the training data too well which included noise and random fluctuations which was counterintuitive to our initial expectation that a simple linear regression model would be too simple to capture the complexities. In the end, we decided that the best degree would be of degree 1, or linear regression and although it had the best results out of every parameter we tried, it still showed a significant sign of overfitting. Therefore, we decided to pursue more tuning through Lasso and Ridge regularizations for our second models to hopefully reduce overfitting.




**Model 2 - Ridge And Lasso Regression L1/L2 Discussion**

The second model(s) we used was Lasso and Ridge Regression, which we grouped together due to their inherent similarity. After our first model, polynomial regression, we observed that the best degree of polynomial was 1, meaning that linear regression was the most successful in giving accurate predictions with the least overfitting (compared to that of other polynomial models). However, it was evident that the model still struggled with overfitting. As a result, we decided to try Lasso and Ridge regression, as the regularization techniques within these models could help to mitigate overfitting by adding a penalty to the loss function. These regularization techniques are catered towards better generalization and discourage the inclusion of disproportionately heavy weights and capturing overly complex relationships.

The hyperparameter tuning in this step was testing which alpha (regularization strength) values would yield the best results, checking for prediction accuracy and overfitting. We tested a substantial range of regularization strengths for both models, testing values of 0.01, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 1, 10, and 100, to ensure that we have a comprehensive range of regularization strengths to test our data on. As expected, with Lasso Regression, we saw a decrease in accuracy, but a corresponding decrease in overfitting (based on significantly lower difference in train and test RMSE) as we increased the regularization strength, results. However, the hyperparameter tuning for Ridge regression did not seem to be as effective in mitigating the overfitting. This is likely due to the fact that overfitting could be more attributed to feature selection rather than model complexity, as Lasso Regression can essentially sink weights down to 0, essentially performing feature selection. This showed us that based on our goals for Lasso and Ridge Regression, Lasso was better suited for our model as it helped us understand where our overfitting was coming from and address it in a targeted manner. 

However, as we increased the regularization strength, we observed that our RMSE was increasing noticeably for both Train and Test data. This showed us that there is a balance aspect to using Lasso Regression, where we need to find out how much error can we induce in order to reduce overfitting before our model becomes too inaccurate with its prediction capabilities. 

*Lasso and Ridge RMSEs Over Alpha*



**Model 3 - Support Vector Regression (SVR) Discussion** 

Our third and final model was Support Vector Regression (SVR), which was certainly our most accurate model with the least overfitting. Seeing as our previous model (Lasso and Ridge Regression) gave us a problematic tradeoff between accuracy and overfitting, we decided to move in another direction. We chose Support Vector Regression due to its ability to capture complex, non-linear patterns in data without causing significant overfitting, as it uses kernels to map features into spaces of different dimensions to capture complex relationships in data. Additionally, we recognized that we still wanted to utilize some regularization technique, which proved to be helpful for dealing with overfitting in our previous model. SVR conveniently includes a hyperparameter C that regularizes features, and on top of that, helps mitigate noise and outliers, as SVR ignores error up until a specified degree based on the epsilon term, which we also address in hyperparameter tuning. 

Based on this rationale, it was clear that the key to success with SVR would be thorough hyperparameter tuning, as there are four hyperparameters to be modified for SVR, giving us a wide range of combinations to try. We set up our model using for loops to test performance on 72 total combinations of hyperparameters, for which we saw extremely varying results. Many combinations gave negative R^2 values and/or extremely high RMSE values, showing extremely poor performance, however, some combinations showed great results that were much better than our first or second models. To reiterate what we covered in our results section, we found an optimal hyperparameter combination of Kernel: linear, C: 1.0, Epsilon: 0.01, and Gamma: Auto or Scale. As explained in the results, these hyperparameters gave us incredibly accurate results, and negligible overfitting, with a difference in Train and Test RMSE being only just over 11%. This showed that our thought process from polynomial regression to support vector regression to SVR led us in the right direction in recognizing that modelling complex patterns in data was important, but we also needed the right regularization tools to mitigate model overfitting. 


[Back to Table of Contents](#%EF%B8%8F-table-of-contents)

##  **Conclusion**

In regards to future possible directions, we all collectively agreed that the best model for this project would have been a Long Short Term Model, a type of neural network that can be used to learn and predict sequential data. By the end of the project it was clear the main issues we were dealing with was the sequential nature of the time series and overfitting, which an LSTM or other neural networks could’ve handled better. As explained above, time series and time steps was one issue that baffled us for a while since our project aimed to predict future close prices based on past ones and that heavily relied on how we dealt with the ‘date’ feature we overlooked for a while. The main issue in our project was struggling with overfitting which we initially thought was due to our models but ended up being a problem with the nature of our dataset. After research, we realized that the volatility in stock market data will always lead to some degree of overfitting in any regression, whether it be polynomial, linear, or L1/L2 regression. This is due to the fact that the closing price or ‘close’ feature in real stock markets are severely impacted by outside events and news, like new laws etc, which are qualitative factors that we simple regression models can’t predict or handle. For example, linear regression aims to find a simple pattern or trend in the data to predict future values but our data will only indicate that there seems to be a trend or pattern when in reality, there isn’t. Thus, the use of LSTM neural networks would be the best choice for a model in regards to this project due to its ability to properly handle temporal dependencies through time based cycles, trends, and patterns that go beyond the simplistic nature of regression and its ability to model non-linear and complex relationships that take into account multiple qualitative factors that regression can’t handle such as sentiment, events, geopolitical changes, and etc. 


[Back to Table of Contents](#%EF%B8%8F-table-of-contents)

##  **Statement of Collaboration**



[Back to Table of Contents](#%EF%B8%8F-table-of-contents)


Some Exploratory Data Analysis Images:





### *Extra Snippets*
Polynomial Regression:

![Screenshot 2024-12-11 at 8 20 20 PM](https://github.com/user-attachments/assets/b52cb0e6-117b-4e15-917b-8fbe5314e861)

![Screenshot 2024-12-11 at 8 20 32 PM](https://github.com/user-attachments/assets/dfcbb346-3ba6-4243-ab55-d02445eacf6f)

![Screenshot 2024-12-11 at 8 20 45 PM](https://github.com/user-attachments/assets/4fc2ed76-6eb7-4bfb-ad6b-bfcea7ab3085)



Lasso:
ASML
![Screenshot 2024-12-11 at 10 23 36 PM](https://github.com/user-attachments/assets/18c52f66-8ac1-4426-ac8f-474dded578c6)

![Screenshot 2024-12-11 at 10 24 03 PM](https://github.com/user-attachments/assets/e863c948-dee8-48d5-b12b-980aec024b7b)

![Screenshot 2024-12-11 at 10 24 32 PM](https://github.com/user-attachments/assets/046cfb49-a49f-46c1-a826-747528a307c5)


Ridge:
ASML
![Screenshot 2024-12-11 at 10 23 46 PM](https://github.com/user-attachments/assets/ad53c138-d3ac-43d0-a203-3ba24b6b5569)

![Screenshot 2024-12-11 at 10 24 21 PM](https://github.com/user-attachments/assets/66f8a990-50ff-45e3-a303-dee39cc35fa4)

![Screenshot 2024-12-11 at 10 24 42 PM](https://github.com/user-attachments/assets/ae98139d-e66c-45bc-b1b6-65696054bea6)



SVR:

![Screenshot 2024-12-11 at 8 22 48 PM](https://github.com/user-attachments/assets/7c8ab550-1fbe-4b45-859b-486908f2f521)

![Screenshot 2024-12-11 at 8 22 55 PM](https://github.com/user-attachments/assets/09b4314b-00d2-4d01-8e3e-3824cc7c7ede)

![Screenshot 2024-12-11 at 8 23 05 PM](https://github.com/user-attachments/assets/faa86b44-7e4b-4f69-8bf6-6e90c35cb59b)

## Statement of Collaboration: 
** Derick Jang **
- Collaborations: Completed all data exploration, preprocessing, building model 1, model 1 hyper parameter tuning, and model 1 visualization figures and evaluation metrics. Also helped with write ups and README discussion section

** Dhruv Patravali **
- Collaborations: Developed the Lasso/Ridge regression models, and the Support Vector Regression models, and contributed to the write up. 

** Ivor Myers **
- Collaborations: Focused on data analysis, interpreting data and aiding with hyperparameter tuning, research for which models to try based on previous milestones, optimizing code, and he contributed to the writeup. 

** Cole Carter **
- Collaborations: Focused on generating the graphs for our all our of preprocessing, and models, which was crucial in understanding our data and our model predictions, and contributed to the writeup.

** All team members contributed equally in one way or another. 


## Prior Submissions(Write Ups):
## Milestone 2 Work and Changes:
Milestone 2 Q5: How will you preprocess your data? You should only explain (do not perform pre-processing as that is in MS3) this in your README.md file and link your Jupyter notebook to it. All code and  Jupyter notebooks have be uploaded to your repo.

The first steps of our preprocessing will be to handle the outliers we have. From the graphing in this step, we have a fair amount, so we need to adaquitley handle them for our model. We also have a couple missing values in two of our features, so we will explore leaving them be and adding in average values at the missing spots. For our features that are skewed we want to experiment with normalizing said data when employing it in the model. Additionally, in the pre-processing step we are considering adding in a couple new features that highlight price precentage changes from day to day and throughout each day. We will likely use a 80-10-10 split for training, validation, and testing, but want to try a couple different splits to find an optimal approach. 

## Milestone 3 New Work and Changes: 
Building off of milestone 2, we went to office hours and talked to TAs about our ideas of preprocessing for our dataset. We were told that we had too many features and we should focus on only the most important ones. Since our dataset includes the features 'date' which is a datetype, 'ticker' which is an object representing the name of the stock, 'close' which is our target feature, and a large amount of numerical features, we decided that the best course of action to find the most important features was to build off what we did in Milestone 2. We filtered through all the numerical features by calculating their correlation with 'close', our target feature, and dropped every numerical feature that had a correlation of less than 0.9 as they wouldn't have a significant enough impact on our target. In doing so, we reduced our features down to 17. In our pre-processing, we had two categorical features: date and ticker in which we took different routes. For dates, we needed to use ordered encoding so we employed ordinal encoding to our date column. For ticker, we took a different route since our dataset included many tickers which represented many different stocks so we decided to loop through every ticker and process them individually so the model wouldn't be corrupted. In doing so, through the loop we created a new dataframe for each ticker and dropped every observation that didn't include the current ticker and we then did data splitting as our first step to prevent memory leaks. Afterwards, we handled missing data points by using an inputer to replace missing data with the median and employed two different types of scaling. For features with a Gaussean distribution demonstrated by our graphs from Milestone 2, we used a standard scaler to take care of skewed data and for data that needed to remain bounded, we used a MinMax scaler to ensure that all features are scaled. We then used feature expansion through polynomial transformations and after the pre-processing, we trained the model with polynomial regression and calculated the train and test error for every ticker and output them. Finally, we averaged out all the train and test errors for all the tickers and plotted the fitting graph. 

Milestone 3 Q4: Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?
As we can see from our charts, our train predictions and test predictions are extremely similar and have almost the exact same shape for every single ticker. That implies that our model fits extremely well in the fitting graph as both the charts and the low RMSE prove that the model performs and fits well on both training and testing. Despite that, the next model we are thinking of is a neural network model because polynomial regression doesn't capture time series properly. Our project is aimed at predicting future stock price movements and with polynomial regression, the predictions are unstable/inaccurate because our data doesn't allow it to properly train the model through time series. In our next model, we plan on employing neural networks through long short-term memory so our model can correctly capture the dates and create time series where the model can convert the dates into sequences and predict the next movement based on the closing prices of the previous days which our polynomial regression model can't do neccesarily. 

Milestone 3 Q5: What is the conclusion of your 1st model? What can be done to possibly improve it?
As stated above, our first model has shown amazing results as it can clearly fit and model both the training and test data as shown by the graphs and RMSE. I think our model was a success but can also have improvements. In regards to pre-processing, many of our features had significantly skewed distributions and long-tailed distributions which standardization may have helped slightly but could have been improved even moreso with log transformations. Furthermore, our ordinal encoding of the 'date' feature doesn't necessarily capture the importance of the feature as it just turns it into an ordered feature without as much meaning as it is supposed to have. For example, certain dates such as such as March 26th, the day when bitcoin splits, or certain days of the week may have significant impact on the closing price of a stock and that value can't be captured in ordinal encoding. Therefore, a different model such as neural nets which employ time series and other tactics will provide a much better result. 

## Milestone 4 New Work and Changes:

Q1: We worked on training 3 new models for this milestone. We wanted to work on four regression models so we decided on Lasso, Ridge, Random Trees, and SVR. These models reduce the risk of overfitting and are robust for sparse datasets. We were able to tune our hyperparameters in order to reduce overfitting further and find models that would accurately fit both our training and testing datasets. We also worked on improving our Polynomial Regression as for Milestone 3 we had problems with overfitting. 

Q2: For our three models, we had generally low differences in our training versus testing error. First for the Lasso model, when looking at the averages of our RMSEs we saw that for the training data, we had an RMSE of 0.156786 and a test RMSE of 0.381917. Considering that these are the averages they don't show many signs of overfitting as Lasso has built in regularization. Now looking at Ridge, we saw an average train RMSE of 0.03902 and an average test RMSE of 0.090765. Again Ridge has built-in regularization so the fact that there weren't signs of overfitting was a good reassurance that this model was a good selection. For SVR we had a training RMSE of 0.620019 and a test RMSE of 1.761308. For Random Trees we saw an average train RMSE of 0.1274 and an average test RMSE of 0.9109. Finally, for our improved Polynomial Regression, we had a training RMSE of 0.0349 and a test RMSE of 0.1242. This was much better than our overfit Polynomial Regression model from Milestone 3. 
 

Q3: The Lasso model fits the data well, with minimized overfitting as seen in our error metrics. However, this model is able to lose some of the complexity of the data, using a model like ElasticNet could be interesting. For Ridge, we again saw that it was pretty balanced and able to generalize without overfitting the data. I think next time we could also consider Gradient Boosting. For SVR, . For Random Trees, in our model, we saw that there were some signs of overfitting with the error being lower for the training set. This could be because there wasn't as much regularization involved. Overall, to improve any potential overfitting we can use other techniques to regularize our data in the preprocessing step and minimize any residual overfitting. With regularization penalizing large coefficients, we can hope to lower any overfitting that does occur.


### Milestone 4 Conclusion:
Overall for this milestone, we wanted to really focus on hyperparameterization and preventing overfitting. We were able to parse through the best hyperparameters for each of the three models that we chose and regularize the data when necessary. The benefit of using Lasso and Ridge is that they have built-in regularization using the L1 and L2 regularization. When looking at Random Trees, using the tree structure with the separate samples and features per leaf node allows for us to reduce the change of overfitting and get a more robust model that can perform well on test sets. For SVR, the benefit is using things like Standard Scalar in order to standardize the data to combat the potential for overfitting. The name of the game for us was dialing in the hyper parameterization and we were able to do this with looping over certain key parameters and then with more complex methods like GridSearchCV. We were able to use our features to more accurately make better predictions about stock price. In Milestone 3, we came up short with the Polynomial Regression that we had, however, for this Milestone, we were able to make lots of improvements in our predictions by way of using models that were better suited for our dataset. 
