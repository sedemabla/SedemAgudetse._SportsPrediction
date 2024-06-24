# %% [markdown]
# importing necessary libraries
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %% [markdown]
# mounting google drive to load  my dataset

# %%
#from google.colab import drive
#drive.mount('/content/drive')

# %% [markdown]
# assigning the players22 dataset to df_22

# %%
data_22  = pd.read_csv('players_22 .csv', low_memory= False)

# %% [markdown]
# assigning the male_palyers dataset to df_male

# %%
data_male=pd.read_csv('male_players (legacy) .csv',low_memory= False)

# %% [markdown]
# 

# %% [markdown]
# Making a copy of the datasets to keep the original as reference

# %% [markdown]
# #Exploratory data analysis

# %%
corr_matrix = data_22.corr(numeric_only=True)
corr_matrix['overall'].sort_values(ascending=False)

# %% [markdown]
# removing useless variables( negative correlation with target stat)
# 
# 
# 
# 

# %%
data_22.drop(['goalkeeping_diving','goalkeeping_handling','goalkeeping_kicking','goalkeeping_reflexes','goalkeeping_positioning','nation_jersey_number','nationality_id','club_jersey_number','club_team_id','league_level','nation_team_id', 'sofifa_id'], axis=1, inplace=True)


# %% [markdown]
# Assigning all columns that end with "_url" to url_columns

# %%
url_cols = [col for col in data_22.columns if col.endswith('_url')]


# %% [markdown]
# Removing all columns  that end eith _url from data_22 dataset

# %%
data_22.drop(columns=url_cols, axis=1, inplace=True)

# %% [markdown]
# 

# %% [markdown]
# Checking the correlation of all the columns to "overrall" column  to  see which has a high and low effect .

# %%
corr_matrix = data_male.corr(numeric_only=True)
sorted_corr = corr_matrix['overall'].sort_values(ascending=False)
pd.set_option('display.max_rows', None)
print(sorted_corr)


# %% [markdown]
# Dropping columns with negative correlation becuase they have low to no  effect on the overrall column .

# %% [markdown]
# 

# %%
data_male.drop(['goalkeeping_diving','goalkeeping_handling','goalkeeping_kicking','goalkeeping_reflexes','goalkeeping_positioning','nation_jersey_number','nationality_id','club_jersey_number','club_team_id','league_level','nation_team_id', 'fifa_update','player_id'], axis=1, inplace=True)


# %% [markdown]
# Assigning all columns that end with "_url" to url_columns

# %%
url_cols_male = [col for col in data_male.columns if col.endswith('_url')]
data_male.drop(columns=url_cols_male, axis=1, inplace=True)

# %% [markdown]
# 

# %% [markdown]
# Dropping object columns in data_22 dataset before checking  before  imputation(replace missing values)

# %%
data_22.drop(['club_name','league_name','club_position','club_loaned_from','club_joined','club_contract_valid_until','nation_position','player_tags','player_traits','real_face','player_positions', 'goalkeeping_speed', 'dob'], axis=1, inplace=True)

# %% [markdown]
# Dropping object columns in data_22 dataset before checking before  imputation(replace missing values)

# %%
data_male.drop(['club_name','league_name','club_position','club_loaned_from','nation_position','club_joined_date','club_contract_valid_until_year','player_tags','player_traits','real_face','player_positions', 'goalkeeping_speed', 'dob'], axis=1, inplace=True)

# %% [markdown]
# assisgn all columns with data type object to drop_cols

# %%
drop_cols = data_22.select_dtypes(include=['object'])


# %% [markdown]
# drop all the object columns in data_22

# %%
data_22.drop(data_22.select_dtypes(include=['object']), axis=1, inplace=True)


# %%
data_22.info()

# %% [markdown]
# assisgn all columns with data type object in data_male to drop_cols_male

# %%
drop_cols_male = data_male.select_dtypes(include=['object'])


# %% [markdown]
# drop all the object columns in data_male

# %%
data_male.drop(data_male.select_dtypes(include=['object']), axis=1, inplace=True)


# %% [markdown]
# 

# %%
data_22.info()

# %% [markdown]
# check for missing values in data_22

# %%
missing_values = data_22.isnull().sum()
print(missing_values)

# %% [markdown]
# check for missing values in data_22

# %%
missing_values_2 = data_male.isnull().sum()
print(missing_values_2)

# %% [markdown]
# Using simple imputer to replace the  missing values in the data_22. with the mean

# %%
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
imputed_data_22 = imp.fit_transform(data_22)


# %% [markdown]
# Assigning a  new DataFrame data_22 from imputed_data and assign it as the original data_22 DataFrame.

# %%
data_22= pd.DataFrame(imputed_data_22, columns=data_22.columns)


# %% [markdown]
# Adding the object columns back into the data_22 dataset after replacing missing values

# %%
data_22 = pd.concat([drop_cols, data_22], axis=1)


# %% [markdown]
# Using simple imputer to replace the  missing values in the data_22. with the mean
# 

# %%
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
imputed_data_male = imp.fit_transform(data_male)


# %% [markdown]
# Assigning a  new DataFrame data_male from imputed_data and assign it as the original data_male DataFrame.

# %%
data_male= pd.DataFrame(imputed_data_male, columns=data_male.columns)


# %% [markdown]
# Adding the object columns back into the data_male  dataset after replacing missing values

# %%
data_male = pd.concat([drop_cols_male, data_male], axis=1)


# %%
# data_22.drop(['player_face_url', 'club_logo_url', 'club_flag_url','nation_logo_url', 'nation_flag_url'], axis=1, inplace=True)

# %%
# data_male.drop('player_face_url')

# %% [markdown]
# assigning all object columns in data_22 dataset to Unique cols except (unecessary  and encoded variables)

# %%
unique_cols = data_22.select_dtypes(include=['object']).drop(['short_name','long_name','nationality_name','preferred_foot','work_rate','body_type'], axis=1)



# %% [markdown]
# coverting all columns in unique cols from object into strings

# %%
for col in unique_cols.columns:
    unique_cols[col] = unique_cols[col].astype(str)



# %% [markdown]
# assigning all object columns in data_male dataset to Unique cols male except (unecessary  and encoded variables)

# %%
unique_cols_male = data_male.select_dtypes(include=['object']).drop(['short_name','long_name','nationality_name','preferred_foot','work_rate','body_type'], axis=1)



# %% [markdown]
# coverting all columns in unique_cols_male from object into strings

# %%
for col in unique_cols_male .columns:
    unique_cols_male [col] = unique_cols_male [col].astype(str)



# %% [markdown]
# creating a function that  takes all strings in the columns with "-" and "+"  , coverts them into integer and  adds them  together

# %%
def convert_and_sum_value(value):
    if pd.isna(value):
        return 0
    else:
        parts = str(value).replace('-', '+').split('+')
        num1 = int(parts[0])
        num2 = int(parts[1]) if len(parts) > 1 else 0
        return num1 + num2



# %% [markdown]
# looping through all  columns in in Unique_cols and pushing  it through the convert and sum function

# %%
for col in unique_cols.columns:
    unique_cols[col] = unique_cols[col].apply( convert_and_sum_value)


# %% [markdown]
# looping through all  columns in in Unique_cols_male and pushing  it through the convert and sum function

# %%
for col in unique_cols_male.columns:
   unique_cols_male[col] = unique_cols_male[col].apply( convert_and_sum_value)



# %% [markdown]
# created a copy of the data i transformed

# %%
# correcting values like "89+3"
data_male.drop(data_male.select_dtypes(include=['object']).drop(['short_name','long_name', 'nationality_name','preferred_foot','work_rate','body_type'], axis=1), axis=1, inplace=True)
begin_cols = data_male[data_male.columns[:8].tolist()]
data_male.drop(begin_cols, axis=1, inplace=True)
data_male = pd.concat([unique_cols_male, data_male], axis=1)
data_male = pd.concat([begin_cols, data_male], axis=1)

# %%
# correcting values like "89+3"
data_22.drop(data_22.select_dtypes(include=['object']).drop(['short_name','long_name', 'nationality_name','preferred_foot','work_rate','body_type'], axis=1), axis=1, inplace=True)
begin_cols2 = data_22[data_22.columns[:8].tolist()]
data_22.drop(begin_cols2, axis=1, inplace=True)
data_22 = pd.concat([unique_cols, data_22], axis=1)
data_22= pd.concat([begin_cols2, data_22], axis=1)

# %% [markdown]
# importing label enconder to convert my categorical data into numerical

# %%
from sklearn.preprocessing import LabelEncoder

# %%

p_foot = data_22['preferred_foot']
work_r = data_22['work_rate']
body_t = data_22['body_type']


# %% [markdown]
# converting the categorical data of work rate column into numerical data  for data_22 dataset

# %%

work_r_label_encoder = LabelEncoder()
work_r_encoded = work_r_label_encoder.fit_transform(work_r)
work_r = pd.Series(work_r_encoded, name='work_rate')

# %% [markdown]
# converting the categorical data of body_type column into numerical data  for data_22 dataset

# %%

body_t_label_encoder = LabelEncoder()
body_t_encoded = body_t_label_encoder.fit_transform(body_t)
body_t = pd.Series(body_t_encoded, name='body_type')



# %% [markdown]
# converting the categorical data of preferred foot rate column into numerical data  for data_22 dataset

# %%

p_foot_label_encoder = LabelEncoder()
p_foot_encoded = p_foot_label_encoder.fit_transform(p_foot)
p_foot = pd.Series(p_foot_encoded, name='preferred_foot')


# %% [markdown]
# 

# %%
data_22 = pd.concat([data_22, p_foot, work_r, body_t], axis=1)

# %% [markdown]
# Asssigning the following columns  to  the respective  variables for encoding

# %%

p_foot2 = data_male['preferred_foot']
work_r2 = data_male['work_rate']
body_t2 = data_male['body_type']

# %% [markdown]
# converting the categorical data of work rate  column into numerical data  for data_22 dataset
# 
# ---
# 
# 

# %%
work_r2_label_encoder = LabelEncoder()
work_r2_encoded = work_r2_label_encoder.fit_transform(work_r2)
work_r2 = pd.Series(work_r2_encoded, name='work_rate')

# %% [markdown]
# converting the categorical data of body type column into numerical data  for data_22 dataset

# %%

body_t2_label_encoder = LabelEncoder()
body_t2_encoded = body_t2_label_encoder.fit_transform(body_t2)
body_t2 = pd.Series(body_t2_encoded, name='body_type')


# %% [markdown]
# converting the categorical data of preferred foot column into numerical data  for data_male dataset

# %%

p_foot2_label_encoder = LabelEncoder()
p_foot2_encoded = p_foot2_label_encoder.fit_transform(p_foot2)
p_foot2= pd.Series(p_foot2_encoded, name='preferred_foot')


# %% [markdown]
# joining the converted numerical columns back to the data_male dataset

# %%
data_male = pd.concat([data_male, p_foot2, work_r2, body_t2], axis=1)

# %% [markdown]
# List al columns and the correlation to the "overrall" column in the data 22 dataset

# %%
correlations = data_22.corr(numeric_only=True)
correlations['overall'].sort_values(ascending=False)

# %% [markdown]
# #Feature Engineering
# feature subset extraction.Checking the corrrelation of all columns to see with has a correlation greater than 0.45(meaning they have a huge impact on Overrall stat of player ) in the data_male dataset.

# %%
columns_with_high_correlation = correlations['overall'].sort_values(ascending=False)[correlations['overall'] > 0.45].index.tolist()
chosen_cols = [col for col in columns_with_high_correlation if col not in unique_cols.columns]


# %% [markdown]
# displaying all columns with  correlation of higher than 0.45

# %%
print(chosen_cols)

# %% [markdown]
# List all columns and the correlation to the "overrall" column in the data male dataset

# %%
correlations2 = data_male.corr(numeric_only=True)
correlations2['overall'].sort_values(ascending=False)

# %% [markdown]
# Checking the corrrelation of all columns to see with has a correlation greater than 0.45(meaning they have a huge impact on Overrall stat of player ) in the data_male dataset

# %%
columns_with_high_correlation2 = correlations2['overall'].sort_values(ascending=False)[correlations2['overall'] > 0.45].index.tolist()
chosen_cols2 = [col for col in columns_with_high_correlation2 if col not in unique_cols_male.columns]
print(len(chosen_cols2))

# %% [markdown]
# Assisgning the chosen columns for

# %%
data_male_train=data_male[chosen_cols2]

# %%
data_22 = df_22[chosen_cols]

# %% [markdown]
# 

# %%
data_22_test=data_22[chosen_cols]

# %% [markdown]
# dependent and independent variables of data_22
# 
# 
# 

# %%
Y_male = data_male_train['overall']
X_male = data_male_train.drop('overall', axis=1)


# %% [markdown]
# Importing standard scaler to scale the  data_male dataset .

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_male.copy())
X_male=pd.DataFrame(scaler.transform(X_male.copy()), columns=X_male.columns)

# %%
import pickle
filename = 'Fifa24.pkl'
pickle.dump(scaler, open(filename, 'wb'))


# %% [markdown]
# importing necessary libraries for taining and testing

# %%
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

# %% [markdown]
# splitting the data into training and testing

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X_male, Y_male):
    print(train_index, test_index)
    Xtrain = pd.DataFrame(X_male, index=train_index)
    Xtest = pd.DataFrame(X_male, index=test_index)
    Ytrain = pd.DataFrame(Y_male, index=train_index)
    Ytest = pd.DataFrame(Y_male, index=test_index)

# %% [markdown]
# #using random forest regression

# %%
print("Order of columns in which the model was fitted: ", list(Xtrain.columns))

# %%
rf = RandomForestRegressor(n_estimators=50, max_depth=10, criterion='absolute_error', n_jobs=-1)
rf.fit(Xtrain, Ytrain.values.ravel())
y_pred = rf.predict(Xtest)

# %%
# Save the trained model to a file using pickle
with open('ranf_fifa_model.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

# %%
mean_absolute_error(Ytest, y_pred)

# %%
np.sqrt(mean_squared_error(Ytest, y_pred))

# %% [markdown]
# #Gradient boost

# %%
gboost = GradientBoostingRegressor(init=rf, n_estimators=100, learning_rate=0.001, criterion='friedman_mse')
gboost.fit(Xtrain, Ytrain.values.ravel())
y_pred = gboost.predict(Xtest)

# %%
mean_absolute_error(Ytest, y_pred)

# %%
np.sqrt(mean_squared_error(Ytest, y_pred))

# %%
pip install xgboost

# %%
from xgboost.sklearn import XGBRegressor

# %% [markdown]
# #XGB boost

# %%
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.0001,
    objective='reg:squarederror'
)
xgb.fit(Xtrain, Ytrain)
y_pred=xgb.predict(Xtest)

# %%
mean_absolute_error(Ytest, y_pred)

# %%
np.sqrt(mean_squared_error(Ytest, y_pred))

# %% [markdown]
# #Hyperparameter Tuning with Grid Search

# %%
param_dist = {
    'n_estimators': [10, ],
    'max_depth': [5,],
    'criterion': ['absolute_error']
}

# %%
rf = RandomForestRegressor(n_estimators=10, max_depth=5, criterion='absolute_error', n_jobs=-1)


# %%
kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Reduced number of splits for faster execution


# %%
rf_random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)


# %%
rf_random_search.fit(X_male, Y_male.values.ravel())


# %% [markdown]
# finding the best model out of the models trained

# %%
print("Best Parameters Found: ", rf_random_search.best_params_)
print("Best Score Achieved: ", rf_random_search.best_score_)
print("Best Estimator: ", rf_random_search.best_estimator_)
print("Order of columns in which the model was fitted: ", list(X_male.columns))

# %%
best_rf = rf_random_search.best_estimator_
y_pred = best_rf.predict(Xtest)

# %%
mae = mean_absolute_error(Ytest, y_pred)
rmse = np.sqrt(mean_squared_error(Ytest, y_pred))


# %%
y_22 = data_22_test['overall']
X_22 = data_22_test.drop('overall', axis=1)

# %% [markdown]
# scaling the players22 dataset for testing

# %%

X_22 = X_22.drop('shooting', axis=1)


# %%
print(X_male.columns)

# %%
X_22_reordered = X_22.reindex(columns=X_male.columns)


# %%
print("Columns of X_22_reordered:")
print(X_22_reordered.columns)


# %%
X_22 = pd.DataFrame(scaler.transform(X_22_reordered.copy()), columns=X_22_reordered.columns)

# %% [markdown]
# testing the best model on the players22 dataset

# %%
Xtest_22 = X_22
Ytest_22 = y_22
y_pred_22 = best_rf.predict(Xtest_22)

# %% [markdown]
# #saving the model
# 

# %%
filename = 'ranf_fifa_model.pkl'
pickle.dump(best_rf, open(filename, 'wb'))


