# Exploratory Data Analysis: Ames Housing

This dataset can be downloaded from [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). 

The Ames Housing dataset serves as an incredible modern and comprehensive alternative to the more commonly used Boston Housing dataset. Despite being a dataset primarily designed for regression, our focus in this project will be exploratory data analysis.

Let's begin...

#### Outlier detection and Imputation

**Null Values**

First, let's look for null values in our training dataset. To do this we'll plot a heatmap showing us null values for each variable.

<p align="center">
  <img title='Null Values' src='https://github.com/anxrxdh/eda-ames-housing/blob/master/plots/plot1.jpg'>
</p>

A brief glance shows us that there are multiple variables that have too many null values. Our next step is to impute these.

Deleting these is tricky. If you were to simply use `.dropna(inplace=True)`, you'll notice that all rows get deleted. This can be checked by using the `.shape` function and before and after results are (x, y) and (0,y). The reason this happened is because in each row some or the other variable has a missing data point.

Further investigation shows that there are as many as 19 variables that have null values.

**Percentage of null values**

We then move on to printing value counts for 'objects' with more than one null value. This was achieved by writing a custom function `object_vcs_and_nulls()`. The function provides the values of each _categorical_ feature that has null values. Some have a high percentage of nulls (Alley has 93.8% null values), while some have a very low percentage of nulls (Electrical has only one null value).

This indicates that there are houses that don't have certain features like some might not have an Alley, while others may not have a Pool or Misc features.

What this tells us is that 'null' or zero is not _missing_. It is giving us the information that certain houses do not have certain features.

For numeric variables that follow a normal distribution, null handling is easy. Just replace them with the mean of the variable. But there are surprisingly few null values in the numerical variables of this dataset. however, almost all of our nulls are in the categorical variables. 

**Imputations**

After having researched for hours for a better solution,  I have come to the conclusion that the best way to impute the NA values is to change it to '0' for numerical variables and to 'None' for categorical variables.

For this we have two custom functions `replace_na_numeric()` and `replace_na_categorical()`. After running these functions we see that we have zero NA values. 

We'll use this clean dataset for further analysis.

**Correlations**

To know which features have the most influence on the target variable, SalePrice, let's plot the correlation. It is good to check whether independent features are correlated among themselves, which will result in _Multicollinearity_. 

<p align="center">
  <img title='Correlations' src='https://github.com/anxrxdh/eda-ames-housing/blob/master/plots/plot2.jpg'>
</p>

We can see clearly from the heatmap that Overall quality (OverallQual) and Ground Living Area (GrLivArea) have the highest correlation with the SalePrice. This is obvious since larger houses will be costlier than smaller ones and even though it is not clear on what ground the overall quality is calculated in the dataset, it's safe to assume that higher quality will lead to costlier homes.

Other features that are significantly highly correlated with SalePrice are TotalBsmtSF, 1stFlrSurface, FullBath, GarageCars and GarageArea.

To confirm these correlation, let's look at the scatter plot for these features.

<p align="center">
  <img title='Correlations' src='https://github.com/anxrxdh/eda-ames-housing/blob/master/plots/plot3.jpg'>
</p>

All these features show a positive correlation with SalePrice

In order to get the magnitude of correlation, we'll first use the `.corr()` function to get the correlation. Then, we'll be turning this information into a dictionary using `.to_dict` and `.items()` to turn the correlations into key-value pairs. Finally, we'll use the `.sorted()` function to sort the values with `reverse=True` so that the highest correlations are on the top. 

Here's a screenshot of the same

<p align="center">
  <img title='Correlation Values of each feature' src='https://github.com/anxrxdh/eda-ames-housing/blob/master/plots/plot4.jpg'>
</p>

As we can see this sits exactly with our correlation heatmap with Overall quality and Ground Living Area being the top highly correlated features.

**Outlier Removal ** 

For outlier detection we'll plot correlation plot again of all variables wrt SalePrice. To begin this process we need to make sure that we're are using only those variables that are numeric in nature. The plot looks as follows:

<p align="center">
  <img title='Correlation Plots of all features' src='https://github.com/anxrxdh/eda-ames-housing/blob/master/plots/plot5.jpg'>
</p>

From these plots we can clearly see that there are significant outliers in GrLivArea. We'll plot this individually to get a closer look at the data and determine our cutoff for outliers.

<p align="center">
  <img title='Gr Living Ar. vs Sale Price correlation' src='https://github.com/anxrxdh/eda-ames-housing/blob/master/plots/plot6.jpg'>
</p>

It is evident that for GrLivArea, everything above 4000 sq ft can be considered an outlier. These are figures based on my personal estimation and yours can vary. Feel free to experiment on your own.

Now, let us drop the rows that go beyond out cutoff for outliers. 

At this point we have a clean dataset that can be used for regression. Of course, a ton of preprocessing is still required like standardizing the data but we're in a good position to go about it from here.

#### Future Scope

Since the data is completely clean now we can use Lasso, Ridge, XGBoost algorithms to name a few for regression and pick the best model.
