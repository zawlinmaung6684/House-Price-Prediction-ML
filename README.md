# House Price Prediction using Ridge Regression

# 🌟 Description

This is my ‘house price prediction’ machine learning project using Python `sklearn`, in which I trained a linear model to predict house prices using “Ridge Regression.” The major challenge doing this project was estimating an optimal lambda $(\lambda)$ value for ridge penalty. Here, I used **_RidgeTrace_**, **_Bias-Variance Tradeoff_**, and **_RidgeCV_** to find a good $\lambda$ value for the best fit line.

# 👨‍💻 Process

## Download and Explore the Data

The dataset used in this project was taken from Kaggle. You can learn more about the original dataset [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

A quick look at the dataset tells us that the data has numerous attributes, over half of which are categorical and the rest are numeric.

**Exploratory Data Analysis**

“Understanding the distribution of `SalePrice`" and “Its Relationships with Other Features” were my primary focus in this phase, so I created histograms and scatter plots to learn more about these. Understandably, the prices are more clustered around 100k-200k, while just a handful of attributes have a positive or negative linear correlation with `SalePrice`. Also, I plotted histograms for numeric attributes to learn more about their distributions.

## Data Preprocessing

**Feature Engineering**

I created a new column called `HouseAge` before removing the `YearBuilt` feature from the dataset. The reason was that `HouseAge` is more interpretable and helpful when training the model compared to `YearBuilt`. I also dropped a number of columns such as `Id` and other attributes that I think they are not as important as others.

**Dealing with Multicollinearity**

Multicollinearity refers to the occurrence of high intercorrelations between two or more independent variables. Here, I used a correlation matrix to check multicollinearity in my data and found a number of features suffering from this. There are a number of ways to handle that kind of problem. One simple technique is removing those variables, and that’s what I exactly did here.

**Imputing**

The dataset has a considerable amount of missing instances, which can trigger serious problems when training the model. Also learning from the exploratory data analysis above, the data has a number of outliers, which could affect the `mean` value if I decided to use it for the imputing strategy. That’s why, I finally decided to go with `median` to fill missing values in numeric data.

**OneHotEncoding**

It is more like my personal preference to use `OneHotEncoder` instead of `OrdinalEncoder` for categorical features. 

## Training the Model

**Ridge Regression (L2 Regularization)**

$$RSS_{ridge}(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - f_{\overrightarrow{w}, b}(\vec{x}))^2 + \lambda \sum_{j=1}^{k} w_j^2$$

where

$$f_{\overrightarrow{w}, b}(\vec{x}) = b + w_1x_1 + w_2x_2 + ... + w_nx_n$$

The dataset has over 70 attributes, some of which are potentially suffering from multicollinearity that could result in the overfitting problem. That’s why, I chose Ridge Regression (L2 Regularization) over Simple Regression to train the model. The main idea of this technique is that it reduces **_variance_** while introducing a certain amount of **_bias_** to the model by shrinking less important coefficients to zero.

A major challenge of this regularization method is estimating an optimal $\lambda$ value for the penalty term to find the best fit line of the model. A very small $\lambda$ value will definitely cause an overfitting problem because the model is learning every single feature; whereas a very large $\lambda$ value will lead to an underfitting problem because the model will not be able to learn the features well enough.

I first trained the model with an alpha value of `0.01`, which produced the training score of $0.9325$ and the validation score of $0.8501$. Although the model did a good job predicting the house prices of the training dataset, it did a poor job when it comes to the validation dataset. This is a sign of overfitting. 

Here, I used three different ways to find a good $\lambda$ value: **Ridge Trace**, **Bias-Variance Tradeoff**, and **RidgeCV**.
1. **Ridge Trace.** Ridge Trace is simply visualizing how weights get closer to zero, as $\lambda$ value increases. We can also learn from Ridge Trace how the model accuracy becomes smaller with the increase of $\lambda$. Actually, it makes sense because increasing $\lambda$ values makes the model lose its power to learn every feature in the dataset.
2. **Bias-Variance Tradeoff.** It is obvious in Ridge Regression that the bias increases while the variance decreases with the increase of $\lambda$ value.
3. **RidgeCV.** RidgeCV is a `sklearn` built-in method to which you can throw a number of $\lambda$ values, and it will automatically choose the best one for your model.

After training the model again with a $\lambda$ value of 10, the training score was reduced to $0.890$, but the validation data performed better with an accuracy score of $0.868$.
