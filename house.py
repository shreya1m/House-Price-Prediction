import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


housing=pd.read_csv("C:/users/HP/OneDrive/Desktop/house_pred/house.csv")

housing.head() # To get top 5 values of dataset

housing.info() # To get information about features

housing.describe() # It gives count,mean,std,min,25%,50%,75%and max of each feature

housing.hist(bins=50,figsize=(15,10)) # Plotting Histogram of each feature

# Spilting data in Test and Train Data

def split_train_test(data,test_ratio):
   np.random.seed(42)
   suffled=np.random.permutation(len(data))
   print(suffled)
   test_data_size=int(len(data)*test_ratio)
   test_indices=suffled[:test_data_size]
   train_indices=suffled[test_data_size:]
   return data.iloc[train_indices],data.iloc[test_indices]

train_set,test_set=split_train_test(housing, 0.2)

print(f"rows in train set:{len(train_set)}\n and rows in test_set{len(test_set)}\n")

# Doing Straitied Shuffle Split on "CHAS" variable 

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]

housing = strat_train_set.copy()

# Doing Analysis on Training  Set 

strat_test_set['CHAS'].value_counts

corr_matrix=housing.corr() # Correlation  matrix
corr_matrix['MEDV'].sort_values(ascending=False)

housing['CHAS'].value_counts() 

attributes=['MEDV','RM','LSTAT','ZN']
scatter_matrix(housing[attributes], figsize=(20,8)) # Scatter Matrix

housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)

housing['TAXRM']=housing['TAX']/housing['RM']
housing.head()


housing = strat_train_set.drop("MEDV",axis=1) # Dropping the Training Set Target Column 
housing_labels = strat_train_set["MEDV"].copy() # Separating out the Training Set Target Column

# Performing Standardization , imputation on Training Data

my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),('stdscaler',StandardScaler())
    ])

housing_num_tr=my_pipeline.fit_transform(housing)

print(housing_num_tr)

housing_num_tr.shape

# Creating Random Forest Regressor Model 

model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)

housing_predictions = model.predict(housing_num_tr) # Training Set Predictions

print('Predictions on Training Set')
print(housing_predictions[:10])
print(housing_labels[:10])

# Calculating Mse and Rmse

mse=mean_squared_error(housing_labels, housing_predictions)
rmse=np.sqrt(mse)
rmse

# Calculating Cross Val Score

score=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error")
rmse_score=np.sqrt(-score)

rmse_score

def print_scores(score):
    print("scores:",score)
    print("mean:",score.mean())
    print("Standard Deviation:",score.std())

print_scores(rmse_score)


# Testing Model on test data

x_test=strat_test_set.drop(['MEDV'],axis=1)
y_test=strat_test_set['MEDV'].copy()

x_test_prepared=my_pipeline.transform(x_test)
final_predictions=model.predict(x_test_prepared)
final_mse=mean_squared_error(y_test, final_predictions)
final_rmse=np.sqrt(final_mse) # RMSE

print (final_predictions,list(y_test))

final_rmse 

# Saving and Using the model

joblib.dump(model,'Dragon.joblib')
model=joblib.load('Dragon.joblib')
input=np.array([[-0.43942006,  7.12777891, -1.99460764, -0.87288841, -1.92019489,
       -0.24091795, -1.31342188,  2.60907176, -1.00223847, -0.5766368 ,
       -0.98041585,  0.41162689, -0.96331713]])

model.predict(input) # Model prediction



