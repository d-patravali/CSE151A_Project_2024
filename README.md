# **Stock Price Prediction Models** 

![Banner Image](https://as2.ftcdn.net/v2/jpg/04/01/43/81/1000_F_401438164_xZEh1so5FIVpo3fd71xqgEPdu33dyKZm.jpg)  
*_A journey towards profitable trades._*

---

## 🗂️ **Table of Contents**
1. [Introduction](#introduction)
2. [Exploratory Data Analysis](#-exploratory-data-analysis)
3. [Model Processes](#-model-processes)
4. [Results and Visualizations](#-results-and-visualizations)
5. [Conclusion](#conclusion)
6. [Installation and Usage](#installation-and-usage)



---

##  **Introduction**

The stock market is a dynamic system that is influenced by a wide variety of quantitative and qualitative variables. For most investors, analyzing stock data and market trends is a tedious process, and making informed and accurate decisions is often extremely difficult to get right consistently. Our group’s goal is to create a project that considers an extremely thorough dataset of stock market data with an extensive range of features to forecast stock prices accurately and efficiently. <br>
As students who are fascinated by the complexities of public trading and invest in the market personally, we recognize how stock market predictions have significant personal financial implications. Utilizing advanced tools like machine learning can enable better investment decisions and risk management, and at the very least serve as a comprehensive aid for stock market analysis. It can also help investors feel more confident in their investment decisions, and users can feel confident in their decision making being backed by thorough consideration of real data. <br>
This project was fascinating because of the overlap of machine learning model development and real-world applications in finance. This project serves as a basis for a broad, yet detailed understanding of the course material while simultaneously enabling us to apply this knowledge to a valuable, real-world application for financial literacy and personal investment. While sentiment analysis and reactions to market trends may yield substantial investment results, this project is a testament to the fact that data-driven decision-making is a reliable, and more importantly consistent method of generating positive returns in the stock market.
  
### What We Did

We took our Kaggle Dataset and boiled down the features that we thought were most useful...we built Polynomial Regression, Lasso, Ridge, and Random Trees models to predict the prices.
- Perform Data Preprocessing and Exploration.
- Train our models (Polynomial, Lasso, Ridge, Random Trees).
- Vizualise our predictions, analyze metrics, and iteratively improve  
....‼️
###  Goals
- Define the scope of the problem.
- Establish key metrics for success.
- Gather preliminary data or resources.  
- 

### [Dataset](https://www.kaggle.com/datasets/luisandresgarcia/stock-market-prediction)

[Back to Table of Contents](#%EF%B8%8F-table-of-contents)

---

## 📊 **Exploratory Data Analysis**

###  Preprocessing 
Our original dataset contains 7,781 observations and 1,285 features, with empty values and no scaling/standardization. Pre-processing is a crucial step in making our data usable and effective for the models we have built. We first cleaned our dataset by replacing missing values with column medians with a median imputer, and filtering our data for only numerical features. We also scaled those numerical features through standard and min/max scalers. Additionally, it was crucial to decide which of our extensive set of features to use for predictions, so we employed a correlation matrix, and only used features with a correlation value greater than 0.6. Finally, we employed a standard test/train split of 80:20. 
  ‼️
  
### Data Exploration/Visualization

### Deciding on Models

#### Model 1: Polynomial Regression
We Chose Polynomial Regression because... ‼️


#### Model 2: Lasso and Ridge (L1, L2)
We Chose Lasso and Ridge because... ‼️


#### Model 3: Support Vector Regression
We Chose Support Vector Regression because... ‼️


#### Model 4: Random Forest Regression
We Chose Random Forest because... ‼️



[Back to Table of Contents](#%EF%B8%8F-table-of-contents)


## 🛠 **Model Processes**

#### Model 1: Polynomial Regression
Our first model was polynomial regression, for which we tested several different polynomial degrees. For polynomial regression, we generated models for several different polynomial degrees, starting with linear regression, and going up until a degree of 5. For each degree, we performed polynomial feature expansion to prepare our data, and generated MSE and R^2 values to evaluate performance and overfitting/underfitting analysis. 

```
model = PolynomialRegression.fit(X)
...‼️
```


#### Model 2: Lasso and Ridge (L1, L2)
Our second model was built upon the fact that we recognized overfitting in our polynomial regression models. We used Ridge and Lasso regression as regularized linear regression models to mitigate overfitting in our linear model. For these models, hyperparameter tuning was important to determine what regularization strengths would be most effective in mitigating overfitting without compromising performance. We tested alpha (regularization strength) values of 0.01, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 1, 10, and 100 to determine optimal results. <br>
lasso_alphas = [0.01, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 1, 10, 100]<br>
ridge_alphas = [0.01, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 1, 10, 100]

```
model = Lasso
model1 = Ridge
...‼️
```


#### Model 3: Support Vector Regression
The third model we built was Support Vector Regression, which uses support vectors to predict continuous stock price values while minimizing errors within a specified tolerance. This model was used given the high dimensionality of our data set, which SVR works well for. For this model, the most critical step was hyperparameter tuning. We had to test our model on different kernel, C, epsilon, and gamma settings/values. We tested 72 different hyperparameter combinations and found that the most optimal hyperparameter combinations were (linear, 0.1, 0.01, and auto) and (linear, 0.1, 0.01, and scale) respectively. 

```
model = SVR.fit(X)
...‼️
```

#### Model 4: Random Forest Regression
The fourth and final model that we utilized was Random Forest Trees, which splits our data into regions based on feature thresholds. We employed random forest regression to capture the more complex, non-linear patterns in our stock market data. 
```
model = RandomForest.fit(X)
...‼️
```


[Back to Table of Contents](#%EF%B8%8F-table-of-contents)

---

## 📊 **Results and Visualizations**


#### Model 1: Polynomial Regression
Here were our results for Poly Regression:
- Training RMSE: 
- Test RMSE:
- Training R²: 0.9932320517075023
- Test R²: 0.9808777373054388
- Cross-Validation R² Scores: [-2.78935438, 0.28107945, 0.84705524, -0.49929551, -0.19701962]
- Mean Cross-Validation R²: -0.4715069631131919


... and here is the graph of our other error metrics... ‼️


#### Model 2: Lasso and Ridge (L1, L2)
Here were our results for Poly Regression:
- Training RMSE: 
- Test RMSE:
- Training R²: 0.9932320517075023
- Test R²: 0.9808777373054388
- Cross-Validation R² Scores: [-2.78935438, 0.28107945, 0.84705524, -0.49929551, -0.19701962]
- Mean Cross-Validation R²: -0.4715069631131919


... and here is the graph of our other error metrics... ‼️


#### Model 3: Support Vector Regression
Here were our results for Poly Regression:
- Training RMSE: 
- Test RMSE:
- Training R²: 0.9932320517075023
- Test R²: 0.9808777373054388
- Cross-Validation R² Scores: [-2.78935438, 0.28107945, 0.84705524, -0.49929551, -0.19701962]
- Mean Cross-Validation R²: -0.4715069631131919


... and here is the graph of our other error metrics... ‼️


#### Model 4: Random Forest Regression
Here were our results for Poly Regression:
- Training RMSE: 
- Test RMSE:
- Training R²: 0.9932320517075023
- Test R²: 0.9808777373054388
- Cross-Validation R² Scores: [-2.78935438, 0.28107945, 0.84705524, -0.49929551, -0.19701962]
- Mean Cross-Validation R²: -0.4715069631131919


... and here is the graph of our other error metrics... ‼️



[Back to Table of Contents](#%EF%B8%8F-table-of-contents)

---

##  **Conclusion**

###  Description  
Summarize the outcomes and deliverables achieved at the end of the project.  

### Final Outputs
 

###  Key Takeaways
- The proposed methodology demonstrates [specific success].  
- Future research could explore [specific limitations].  


[Back to Table of Contents](#%EF%B8%8F-table-of-contents)

---

##  **Installation and Usage**

```bash
# Clone the repository
git clone https://github.com/d-patravali/CSE151A_Project_2024.git


# Install dependencies
pip install -r requirements.txt

```

[Back to Table of Contents](#%EF%B8%8F-table-of-contents)

## **Past Write-Ups**



# CSE151A_Project_2024
Stock Market Prediction Model for CSE 151A Fall 2024

Link to Data File: https://drive.google.com/file/d/1TNiScpuu1YHd3VzpKw-R64aatgQNukcD/view?usp=sharing

Link To Google Colab For Milestone 2: https://colab.research.google.com/drive/1LGUcWIPsxZ01dDa88lvVxoKfzZO5921M?usp=sharing
(Can only be accessed when using UCSD email)

Link To Google Colab For Milestone 3: https://colab.research.google.com/drive/1uKWRTpOmAjVMnLwXn6AM4jrfSNOKDguN?usp=sharing
(Can only be accessed when using UCSD email)


Milestone 2 Q5: How will you preprocess your data? You should only explain (do not perform pre-processing as that is in MS3) this in your README.md file and link your Jupyter notebook to it. All code and  Jupyter notebooks have be uploaded to your repo.

The first steps of our preprocessing will be to handle the outliers we have. From the graphing in this step, we have a fair amount, so we need to adaquitley handle them for our model. We also have a couple missing values in two of our features, so we will explore leaving them be and adding in average values at the missing spots. For our features that are skewed we want to experiment with normalizing said data when employing it in the model. Additionally, in the pre-processing step we are considering adding in a couple new features that highlight price precentage changes from day to day and throughout each day. We will likely use a 80-10-10 split for training, validation, and testing, but want to try a couple different splits to find an optimal approach. 

Milestone 3 New Work and Changes: 
Building off of milestone 2, we went to office hours and talked to TAs about our ideas of preprocessing for our dataset. We were told that we had too many features and we should focus on only the most important ones. Since our dataset includes the features 'date' which is a datetype, 'ticker' which is an object representing the name of the stock, 'close' which is our target feature, and a large amount of numerical features, we decided that the best course of action to find the most important features was to build off what we did in Milestone 2. We filtered through all the numerical features by calculating their correlation with 'close', our target feature, and dropped every numerical feature that had a correlation of less than 0.9 as they wouldn't have a significant enough impact on our target. In doing so, we reduced our features down to 17. In our pre-processing, we had two categorical features: date and ticker in which we took different routes. For dates, we needed to use ordered encoding so we employed ordinal encoding to our date column. For ticker, we took a different route since our dataset included many tickers which represented many different stocks so we decided to loop through every ticker and process them individually so the model wouldn't be corrupted. In doing so, through the loop we created a new dataframe for each ticker and dropped every observation that didn't include the current ticker and we then did data splitting as our first step to prevent memory leaks. Afterwards, we handled missing data points by using an inputer to replace missing data with the median and employed two different types of scaling. For features with a Gaussean distribution demonstrated by our graphs from Milestone 2, we used a standard scaler to take care of skewed data and for data that needed to remain bounded, we used a MinMax scaler to ensure that all features are scaled. We then used feature expansion through polynomial transformations and after the pre-processing, we trained the model with polynomial regression and calculated the train and test error for every ticker and output them. Finally, we averaged out all the train and test errors for all the tickers and plotted the fitting graph. 

Milestone 3 Q4: Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?
As we can see from our charts, our train predictions and test predictions are extremely similar and have almost the exact same shape for every single ticker. That implies that our model fits extremely well in the fitting graph as both the charts and the low RMSE prove that the model performs and fits well on both training and testing. Despite that, the next model we are thinking of is a neural network model because polynomial regression doesn't capture time series properly. Our project is aimed at predicting future stock price movements and with polynomial regression, the predictions are unstable/inaccurate because our data doesn't allow it to properly train the model through time series. In our next model, we plan on employing neural networks through long short-term memory so our model can correctly capture the dates and create time series where the model can convert the dates into sequences and predict the next movement based on the closing prices of the previous days which our polynomial regression model can't do neccesarily. 

Milestone 3 Q5: What is the conclusion of your 1st model? What can be done to possibly improve it?
As stated above, our first model has shown amazing results as it can clearly fit and model both the training and test data as shown by the graphs and RMSE. I think our model was a success but can also have improvements. In regards to pre-processing, many of our features had significantly skewed distributions and long-tailed distributions which standardization may have helped slightly but could have been improved even moreso with log transformations. Furthermore, our ordinal encoding of the 'date' feature doesn't necessarily capture the importance of the feature as it just turns it into an ordered feature without as much meaning as it is supposed to have. For example, certain dates such as such as March 26th, the day when bitcoin splits, or certain days of the week may have significant impact on the closing price of a stock and that value can't be captured in ordinal encoding. Therefore, a different model such as neural nets which employ time series and other tactics will provide a much better result. 
