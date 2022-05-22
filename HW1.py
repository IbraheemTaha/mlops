import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


from sklearn.metrics import mean_squared_error




def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        df.pickup_datetime = pd.to_datetime(df.pickup_datetime)
        df.dropOff_datetime = pd.to_datetime(df.dropOff_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)


    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Q1 + Q2
    print(df['duration'].describe())


    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].astype(str)
    

    return df

# The anwer of Q1 + Q2 can be found here
df = read_dataframe ('../data/fhv_tripdata_2021-01.parquet')



#sns.distplot(df.duration)
#plt.show()

## ---------------------------------------------------
df['PUlocationID'] = df['PUlocationID'].fillna(-1)
df['DOlocationID'] = df['DOlocationID'].fillna(-1)

## Q3
print(df['PUlocationID'].value_counts()/df.shape[0])

## ---------------------------------------------------

categorical = ['PUlocationID', 'DOlocationID']
df[categorical] = df[categorical].astype(str)

#change the data into dictionary
train_dicts = df[categorical].to_dict(orient='records')

dv = DictVectorizer()
# transform the dictionary (data) into training data
X_train = dv.fit_transform(train_dicts)

# Q4 
print(X_train.shape)

# ---------------------------------------------------

target = 'duration'
y_train = df[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

# Q5 
print('RMSE Training: ',mean_squared_error(y_train, y_pred, squared=False))

# ---------------------------------------------------

val_df = read_dataframe ('../data/fhv_tripdata_2021-02.parquet')


val_df['PUlocationID'] = val_df['PUlocationID'].fillna(-1)
val_df['DOlocationID'] = val_df['DOlocationID'].fillna(-1)


## ---------------------------------------------------

categorical = ['PUlocationID', 'DOlocationID']
val_df[categorical] = val_df[categorical].astype(str)

#change the data into dictionary
val_dicts = val_df[categorical].to_dict(orient='records')

# transform the dictionary (data) into training data
X_val = dv.transform(val_dicts)

# ---------------------------------------------------
y_val = val_df[target].values

y_pred = lr.predict(X_val)

# Q6
print('RMSE Validation: ',mean_squared_error(y_val, y_pred, squared=False))
