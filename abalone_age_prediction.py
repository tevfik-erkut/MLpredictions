                                                        #KÜTÜPHANELER
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, skew

from sklearn.model_selection import validation_curve
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score                                             
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

                                                        # Dataset düzenlemeleri yapıyorum.
column_name = ["Sex","Length","Diameter", "Height","Whole weight", "Shucked weight",
         "Viscera weight", "Shell weight","Rings" ]
data = pd.read_csv("abalone.csv", names = column_name)
data.describe().T
data.info()

data["target"] = data["Rings"] + 1.5    # rings to age
data.drop('Rings', axis = 1, inplace = True)

                                                        # Verimize bir bakalım.                        
data.hist(figsize=(20,10), grid=False, layout=(3, 4), bins = 30)
plt.show()

numerical_features = data.select_dtypes(include=[np.number]).columns
categorical_features = data.select_dtypes(include=[np.object]).columns
skew_values = skew(data[numerical_features], nan_policy = 'omit')                          # Skewness'lara göz gezdiricem.
dummy = pd.concat([pd.DataFrame(list(numerical_features), columns=['Features']), 
           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1)
dummy.sort_values(by = 'Skewness degree' , ascending = False) 

                                                        # Dummy'leri aradan çıkartıyorum.                                     
data = pd.get_dummies(data)


                                                        # Basic LR'Error
#split data
x = data.drop(["target"], axis = 1)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 33)

X_test_data = X_test
y_test_data = y_test

# Standardization
scaler = RobustScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Linear Regression
lr = LinearRegression()
lr.fit(X_train,y_train)
print("Lr Coef:", lr.coef_)
y_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)
mse = mean_squared_error(y_test, y_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error :", mse)
print("Train MSE: ", train_mse)                                    

#Mean Squared Error : 5.27103600668643
#Train MSE:  4.69127331966779

                                                        #EDA  ( corr_matrix'ten correlation btw features lere bakıp threshold belirliycem. Ardından drop işlemi yapicam.)
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlation btw features")
plt.show()

threshold = 0.2
filtre = np.abs(corr_matrix["target"])>threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation btw features")
plt.show()

data = data[corr_features]


                                                        # Multilinearity   Skewness'lara bir de böyle bakalım.
sns.pairplot(data, diag_kind = "kde", markers = "+")
plt.show()

                                                        # Outlier visualization yapicam. ( Subplotlar halinde. ) Görüldüğü üzere birçok outlier var.
                                                        
var = 'Viscera weight'
var1 = "Length"
var2 = "Diameter"
var3 = "Height"

plt.figure(figsize=(27,10))
plt.subplot(2,2,1)
plt.scatter(data[var], data['target'], color = "r")
plt.title("target-wiscera Weight relation")

plt.subplot(2,2,2)
plt.scatter(data[var1], data['target'], color = "b")
plt.title("target-length relation")

plt.subplot(2,2,3)
plt.scatter(data[var2], data['target'], color = "black")
plt.title("target-diameter relation")

plt.subplot(2,2,4)
plt.scatter(data[var3], data['target'], color = "brown")
plt.title("target-height relation")

# IQR method of Outlier Detection kullanacağım. Birden fazla grafiği subplotlar şeklinde göstericem.

plt.figure(figsize= (20,8))
plt.subplot(2,4,1)
number = 1
for column in data.columns[0:8]:
    plt.subplot(2,4,number)
    sns.boxplot(x = column, data = data)
    number +=1
    
data.describe().T                  # datamızın IQR Filtresi uygulanmadan önceki hali

                                   # Birden çok feature'da outlier gördüğüm için bir fonksiyon yazıp kullanacağım.

def iqr(x):
  global data
  thr = 1.5
  length_desc = data[x].describe()
  q3_hp = length_desc[6]
  q1_hp = length_desc[4]
  IQR_hp = q3_hp - q1_hp
  top_limit_hp = q3_hp + thr*IQR_hp
  bottom_limit_hp = q1_hp - thr*IQR_hp
  filter_hp_bottom = bottom_limit_hp < data[x]
  filter_hp_top = data[x] < top_limit_hp
  filter_hp = filter_hp_bottom & filter_hp_top
  data = data[filter_hp]
  
for i in ["Length","Diameter","Height","Whole weight", "Shucked weight", "Viscera weight", "target"]:
  iqr(i)
  
data.describe().T                  # datamızın IQR Filtresi uygulandıktan sonraki hali


# Kullanacağım modeller Regression modelleri olacak, LR, Ridge, Lasso, ElasticNet ve XGBoost kullanacağım. Hyperparameters'leri tune edeceğim CV kullanarak.


                                                        #Linear Regression
#split data
x = data.drop(["target"], axis = 1)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .31, random_state = 33)

# Standardization
scaler = RobustScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
                                                        
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
y_train_pred = lr.predict(X_train)

mse = mean_squared_error(y_test, y_pred)
train_mse = mean_squared_error(y_train, y_train_pred)

print("Mean Squared Error :", mse)
print("Train MSE: ", train_mse)  #Overfit var mı diye bunu da printliyorum.

                                                                        #LR Mean Squared Error : 2.7368784178902783
                                                                        #LR Train MSE:  2.6540934842775576

                                                        #Ridge Regression 

ridge = Ridge(random_state = 33, max_iter = 10000)
alphas = np.logspace(-4,-0.5,60)
tuned_parameters = [{'alpha':alphas}]
n_folds = 5

clf = GridSearchCV(ridge, tuned_parameters, cv = n_folds, scoring = "neg_mean_squared_error", refit = True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

mse = mean_squared_error(y_test, y_pred)
train_mse = mean_squared_error(y_train, y_train_pred)

print("Ridge MSE: ",mse)
print("Ridge Training MSE: ",train_mse )

                                                                        #Ridge MSE:  2.7335816167555795
                                                                        #Ridge Training MSE:  2.654188601542835

                                                        #Lasso Regression (L1)

lasso = Lasso(random_state=33, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_mean_squared_error',refit=True)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

mse = mean_squared_error(y_test,y_pred)
train_mse = mean_squared_error(y_train, y_train_pred)

print("Lasso MSE: ",mse)
print("Lasso Train MSE: ",train_mse)

                                                                        #Lasso MSE:  2.7283457672565583
                                                                        #Lasso Train MSE:  2.654606794958874


                                                        # ElasticNet
                                                        
parametersGrid = {"alpha": alphas,
                  "l1_ratio": np.arange(0.0, 1.0, 0.1)}
                  
eNet = ElasticNet(random_state=33, max_iter=10000)
clf = GridSearchCV(eNet, parametersGrid, cv=n_folds, scoring='neg_mean_squared_error', refit=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

mse = mean_squared_error(y_test,y_pred)
train_mse = mean_squared_error(y_train, y_train_pred)

print("ElasticNet MSE: ",mse)
print("ElasticNet Train MSE: ", train_mse)

                                                                        #ElasticNet MSE:  2.7290212869119714
                                                                        #ElasticNet Training MSE:  2.654548165296342


                                                        #XGBoost
parametersGrid = {'nthread':[4], 
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500,1000]}

model_xgb = xgb.XGBRegressor()
clf = GridSearchCV(model_xgb, parametersGrid, cv = n_folds, scoring='neg_mean_squared_error', refit=True, n_jobs = -1, verbose=True)
clf.fit(X_train, y_train)
model_xgb = clf.best_estimator_

y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
mse = mean_squared_error(y_test,y_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
print("XGBRegressor MSE: ",mse)
print("XGBRegressor Train MSE: ", train_mse)

                                                                        #XGBRegressor MSE:  2.4197799559639246
                                                                        #XGBRegressor Train MSE:  1.1674175666993178
                                                                        #Görüldüğü üzere overfitting var. Bunu düzeltmek için biraz daha parametrelerle oynayacağım.
                                                                        
# colsample_bytree. Lower ratios avoid over-fitting.
# subsample. Lower ratios avoid over-fitting.
# max_depth. Lower values avoid over-fitting.
# min_child_weight. Larger values avoid over-fitting.

# Üsttekileri göz önünde bulundurarak yeni bir parametersGrid tanımlayacağım.

new_params = {'colsample_bytree': [0.4, 0.5],
 'learning_rate': [.005, .009, 0.01],
 'max_depth': [2, 4],
 'min_child_weight': [4,5,6],
 'n_estimators': [500, 1000],
 'nthread': [4],
 'objective': ['reg:linear'],
 'silent': [1],
 'subsample': [0.4, 0.5]}
 
model_xgb = xgb.XGBRegressor()
clf = GridSearchCV(model_xgb, new_params, cv = n_folds, scoring='neg_mean_squared_error', refit=True, n_jobs = -1, verbose=True)
clf.fit(X_train, y_train)
model_xgb = clf.best_estimator_

y_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
mse = mean_squared_error(y_test,y_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
print("XGBRegressor MSE: ",mse)
print("XGBRegressor Train MSE: ", train_mse)

                                                                        #XGBRegressor MSE:  2.397578234350839
                                                                        #XGBRegressor Train MSE:  1.893639602258289   
                                                                        

"""
Final:

>>>>>>>>>>XGBRegressor MSE:  2.397578234350839 <<<<<<<<<<<<<<
ElasticNet MSE:  2.7290212869119714
Lasso MSE:  2.7283457672565583
Ridge MSE:  2.7335816167555795
Linear Regression MSE : 2.7368784178902783

"""
