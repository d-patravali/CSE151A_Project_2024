# CSE151A_Project_2024
Stock Market Prediction Model for CSE 151A Fall 2024

Link to Data File: https://drive.google.com/file/d/1TNiScpuu1YHd3VzpKw-R64aatgQNukcD/view?usp=sharing

Link To Google Colab For Milestone 2: https://colab.research.google.com/drive/1LGUcWIPsxZ01dDa88lvVxoKfzZO5921M?usp=sharing
(Can only be accessed when using UCSD email)

Link To Google Colab For Milestone 3: https://colab.research.google.com/drive/1uKWRTpOmAjVMnLwXn6AM4jrfSNOKDguN?usp=sharing
(Can only be accessed when using UCSD email)


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

Q1: We worked on training 3 new models for this milestone. We wanted to work on three regression models so we decided on Lasso, Ridge, and SVR. These models reduce the risk of overfitting and are robust for sparse datasets. We were able to tune our hyperparameters in order to reduce overfitting further and find models that would accurately fit both our training and testing datasets. We also worked on improving our Polynomial Regression as for Milesone 3 we had problems with overfitting. 

Q2: For our three models, we had generally low differences in our training versus testing error. First for the Lasso model, when looking at the averages of our RMSEs we saw that for the training data, we had an RMSE of 0.3194 and a test RMSE of 0.6064. Considering that these are the averages they don't show many signs of overfitting as Lasso has built in regularization. Now looking at Ridge, we saw an average train RMSE of 0.2115 and an average test RMSE of 0.3599. Again Ridge has built-in regularization so the fact that there weren't signs of overfitting was a good reassurance that this model was a good selection. For SVR we had a training RMSE of , and a test RMSE of . Finally for our improved Polynomial Regression, we had a training RMSE of 0.0349 and a test RMSE of 0.1242. This was much better than our overfit Polynomial Regression model from Milestone 3. 
 

Q3: The Lasso model fits the data well, with minimized overfitting as seen in our error metrics. However, this model is able to lose some of the complexity of the data, using a model like ElasticNet could be interesting. For Ridge, we again saw that it was pretty balanced and able to generalize without overfitting the data. I think next time we could also consider Gradient Boosting. For SVR, .Overall, to improve any potential overfitting we can use other techniques to regularize our data in the preprocessing step and minimize any residual overfitting. With regularization penalizing large coefficients, we can hope to lower any overfitting that does occur.


### Milestone 4 Conclusion:
Overall for this milestone, we wanted to really focus on hyperparameterization and preventing overfitting. We were able to parse through the best hyperparameters for each of the three models that we chose and regularize the data when necessary. The benefit of using Lasso and Ridge is that they have built-in regularization using the L1 and L2 regularization. For SVR


