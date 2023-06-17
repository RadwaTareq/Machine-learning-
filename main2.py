import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import category_encoders as ce
import time

# reading the 3  csv files
data1 = pd.read_csv('movies-revenue.csv')
data2 = pd.read_csv('movie-voice-actors.csv')
data3 = pd.read_csv('movie-director.csv')

# using merge function to merge the 3 tables together
data2.rename(columns={"movie": "movie_title"}, inplace=True)
data1 = pd.merge(data1, data2, on='movie_title', how='left')
data3.rename(columns={"name": "movie_title"}, inplace=True)
data1 = pd.merge(data1, data3, on='movie_title', how='left')
data1 = data1[['movie_title', 'release_date', 'genre', 'MPAA_rating', 'character', 'voice-actor', 'director', 'revenue']]

# copying all the data merged to one csv
data1.to_csv('without.csv', index=False)

# dropping columns
data1.drop('movie_title', inplace=True, axis=1)
data1.drop('character', inplace=True, axis=1)

# date
data1['Year'] = pd.DatetimeIndex(data1['release_date']).year
data1.drop('release_date', inplace=True, axis=1)

# rearranging the columns
data1 = data1[['Year', 'genre', 'MPAA_rating', 'voice-actor', 'director', 'revenue']]

# removing dollar sign and commas from revenue column

data1['revenue'] = data1['revenue'].str.replace(',', '')
data1['revenue'] = data1['revenue'].str.replace('$', '')
data1['revenue'].astype('float')
data1['revenue'] = data1['revenue'].astype(float, errors = 'raise')
datatypes = data1.dtypes
print(datatypes)

# filling null values
data1.fillna(data1.select_dtypes(include='object').mode().iloc[0], inplace=True)
print(data1.info())

# checking the unique values for every column to decide which encoder to use
print(data1.genre.unique())
print(data1.MPAA_rating.unique())
print(data1.director.unique())

# strings encoding "Target"
targetencoding = ce.TargetEncoder() 
transformed1 = targetencoding.fit_transform(data1['director'], data1['revenue'])
data1_new = transformed1.join(data1.drop('director', axis=1))
targetencoding2 = ce.TargetEncoder() 
transformed2 = targetencoding2.fit_transform(data1_new['voice-actor'], data1_new['revenue'])
data1_new = transformed2.join(data1_new.drop('voice-actor', axis=1))

targetencoding3 = ce.TargetEncoder() 
transformed3 = targetencoding3.fit_transform(data1_new['genre'], data1_new['revenue'])
data1_new = transformed3.join(data1_new.drop('genre', axis=1))
print(data1_new.head(10))

targetencoding4 = ce.TargetEncoder() 
transformed4 = targetencoding4.fit_transform(data1_new['MPAA_rating'], data1_new['revenue'])
data1_new = transformed4.join(data1_new.drop('MPAA_rating', axis=1))
print(data1_new.head(10))



first_column = data1_new.pop('revenue')
data1_new.insert(5, 'revenue', first_column)
print(data1_new)
# correlation
corr = data1_new.corr()
# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['revenue'])>0]
# Correlation plot
plt.subplots(figsize=(6, 4))
top_corr = data1_new[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)


data1_new.to_csv('DATA.csv', index=False)


# Polynomial Model
movies_data = data1_new.iloc[:, :]
X = data1_new.iloc[:, 0:4]
Y = data1_new['revenue']
columns = 'movie_title'
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=10, shuffle=True)

scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
X_train.head()

poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_train_predicted = poly_model.predict(X_train_poly)
y_prediction = poly_model.predict(poly_features.transform(X_test))
print("--- %s seconds poly---" % (time.time() - start_time))

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))
print('________________________________________________________________')
print('-->Polynomial')
print('Train Mean Square Error Poly', metrics.mean_squared_error(y_train, y_train_predicted))
print('Test Mean Square Error Poly', metrics.mean_squared_error(y_test, prediction))



true_revenue_value = np.asarray(y_test)[2]
predicted_revenue_value = prediction[2]

print('True value for the movie revenue in the test set in millions is : ' + str(true_revenue_value))
print('Predicted value for the movie revenue in the test set in millions is : ' + str(predicted_revenue_value))
print("R2 score Polynomial=", metrics.r2_score(y_test, prediction))
print("--- %s seconds poly---" % (time.time() - start_time))
print('________________________________________________________________')

# multilinear
movieData= data1_new.iloc[:, :]
X2 = data1_new.iloc[:, 0:4]
Y2 = data1_new['revenue']
columns = ('movie_title')
start_time2 = time.time()

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, Y2, test_size=0.20, shuffle=True,random_state=10)

scaler = MinMaxScaler()
X_train2 = pd.DataFrame(scaler.fit_transform(X_train2), columns=X_train2.columns)

X_test2 = pd.DataFrame(scaler.fit_transform(X_test2), columns=X_test2.columns)


cls = linear_model.LinearRegression()
cls.fit(X_train2, y_train2)
y_train_predicted2 = cls.predict(X_train2)
y_prediction = cls.predict(X_test2)

true_price2 = np.asarray(y_test2)[0]
predicted_price2 = y_prediction[0]
print('-->Multi linear')
print('True revenue Multi is : ' + str(true_price2))
print('Predicted revenue Multi is : ' + str(predicted_price2))
print('Train Mean Square Error Multi', metrics.mean_squared_error(y_train2, y_train_predicted2))
print('Test Mean Square ErrorMulti ', metrics.mean_squared_error(y_test2, y_prediction))
print("R2 score for multi linear =", metrics.r2_score(y_test2, y_prediction))

print("--- %s seconds multi---" % (time.time() - start_time2))