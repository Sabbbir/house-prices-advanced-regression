# Libraries used
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb

train_df = pd.read_csv("/home/druglord/Documents/ML/Competetion/house-prices-advanced-regression-techniques/train.csv")
# train_df
flag = False

sns.set_theme(style="darkgrid")
def analyze_outliers(arg1, arg2, arg1_val, arg2_val, quan_1, quan_3):
    # Define your conditions
    # condition = f"{arg1} > {arg1_val}" or f"{arg2}.between({arg2_val}, train_df['{arg2}'].max())"
    condition = f"{arg1} > {arg1_val}" or f"{arg2}.between({train_df['{arg2}'].max() * .8}, train_df['{arg2}'].max())"
    # condd = f"{arg1} > {arg1_val} or {arg2} > {arg2_val}"
    # outliers_q = train_df.query(condd)
    outliers_q = train_df.query(condition)

    q1 = train_df[arg2].quantile(quan_1)
    q3 = train_df[arg2].quantile(quan_3)
    iqr = q3 - q1
    up = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr

    # Mean/median outlier
    outliers_m = train_df[(train_df[arg2] < low) | (train_df[arg2] > up)]

    # Zscore
    outliers_z_t = stats.zscore(train_df[arg2]).sort_values().tail(10)
    outliers_z = train_df.loc[train_df['Id'].isin(outliers_z_t.index.tolist())]

    # Query
    if flag == True:
        print("Outliers using query for", arg2, len(outliers_q))
        # print(outliers_q["Id"].tolist())
        print(', '.join(map(str, outliers_q["Id"].tolist())))

    # Mean
        print("Outliers using mean/median", arg2, len(outliers_m["Id"].tolist()))
        print(', '.join(map(str, outliers_m["Id"].tolist())))

    # Train Raw
        print("Sale price zscore with train_df raw", arg2, len(outliers_z_t.index.tolist()))
        print(outliers_z_t)
    # print(outliers_z_t.index.tolist())

        print(', '.join(map(str, [index + 1 for index in outliers_z_t.index.tolist()])))

    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(x=arg1, y=arg2, data=train_df, label="train", ax=axes[0])
    sns.scatterplot(x=arg1, y=arg2, data=outliers_q, label="q", ax=axes[0])
    sns.scatterplot(x=arg1, y=arg2, data=outliers_m, label="m", ax=axes[0])
    sns.scatterplot(x=arg1, y=arg2, data=outliers_z, label="z", ax=axes[0])

    # KDE plots
    sns.kdeplot(x=arg1, y=arg2, data=train_df, color='blue', ax=axes[1])
    sns.kdeplot(x=arg1, y=arg2, data=outliers_q, color="orange", ax=axes[1])
    sns.kdeplot(x=arg1, y=arg2, data=outliers_m, color='green', ax=axes[1])
    sns.kdeplot(x=arg1, y=arg2, data=outliers_z, color='red', ax=axes[1])
    print(axes[0].set_title(f"Scatter Plot of {arg2} vs. SalePrice"))
    print(axes[1].set_title(f"KDE Plot of {arg2} vs. SalePrice"))
    # Set titles for subplots

    # Show plots
    plt.show()

numeric_columns = train_df.select_dtypes(include=['int64', 'float64']).columns
for column in numeric_columns:
    analyze_outliers(arg1="SalePrice", arg2=column, arg1_val=700000, arg2_val=(train_df[column].max())*.8, quan_1=0.21, quan_3=0.95)
