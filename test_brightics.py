import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

customer = pd.read_csv('C:/Users/hyoung-gyu/Google 드라이브/삼성 Brightics 공모전/mealData_customer.csv')
customer0525_0731 = pd.read_csv('C:/Users/hyoung-gyu/Google 드라이브/삼성 Brightics 공모전/mealData_customer_0525_0731.csv')
meal = pd.read_csv('C:/Users/hyoung-gyu/Google 드라이브/삼성 Brightics 공모전/mealData_meal.csv')
meal_0525_0731 = pd.read_csv('C:/Users/hyoung-gyu/Google 드라이브/삼성 Brightics 공모전/mealData_meal_0525_0731.csv')
weather_log = pd.read_csv('C:/Users/hyoung-gyu/Google 드라이브/삼성 Brightics 공모전/20190811171205_weather_log.csv')
weather_after_crawling = pd.read_csv('C:/Users/hyoung-gyu/Google 드라이브/삼성 Brightics 공모전/weather_2019_08_11.csv')

print(customer.shape)
customer = pd.concat([customer, customer0525_0731], axis=0)
print(customer.shape)
customer = customer.drop_duplicates()
print(customer.shape)
del customer0525_0731

print(meal.shape)
meal = pd.concat([meal, meal_0525_0731], axis=0)
print(meal.shape)
print(meal[meal.duplicated(keep=False)])  # 중복 확인
# meal = meal.drop_duplicates()  # 한 사람이 같은 메뉴를 다른사람에게 사주느라고 이런 거 같은데 2번 산걸로 카운팅이 되지 않음. 일단 선호가 높다고 간주하고 빈도를 높이는 것으로 감.
print(meal.shape)
del meal_0525_0731
# 'Chef`sCounter' : "Chef's Counter"
# KOREAN1
# KOREAN2
# 'TakeOut' : 'Take out'
# Western
# '가츠엔' >> 'JapaneseFood',
# '고슬고슬비빈' : 'bibimbap',
# '나폴리폴리' : 'Napolipoli',
# '스냅스낵' : 'Snack',
# '싱푸차이나' : 'SingpuChina',
# '아시안픽스' : 'AsianPicks',
# '우리미각면' : 'AsianNoodle',
# '탕맛기픈' : 'DeepSoup'
meal.loc[meal.BRAND == 'Chef`sCounter', 'BRAND'] = "Chef's Counter"
meal.loc[meal.BRAND == 'TakeOut', 'BRAND'] = "Take out"
meal.loc[meal.BRAND == '가츠엔', 'BRAND'] = "JapaneseFood"
meal.loc[meal.BRAND == '고슬고슬비빈', 'BRAND'] = "bibimbap"
meal.loc[meal.BRAND == '나폴리폴리', 'BRAND'] = "Napolipoli"
meal.loc[meal.BRAND == '스냅스낵', 'BRAND'] = "Snack"
meal.loc[meal.BRAND == '싱푸차이나', 'BRAND'] = "SingpuChina"
meal.loc[meal.BRAND == '아시안픽스', 'BRAND'] = "AsianPicks"
meal.loc[meal.BRAND == '우리미각면', 'BRAND'] = "AsianNoodle"
meal.loc[meal.BRAND == '탕맛기픈', 'BRAND'] = "DeepSoup"

print(weather_log.shape)
weather_log = pd.concat([weather_log, weather_after_crawling], axis=0)
print(weather_log.shape)
del weather_after_crawling


# cust_meal1 = pd.merge(customer, meal, how='right', on='CUSTOMER_ID')
# cust_meal2 = pd.merge(customer, meal, how='left', on='CUSTOMER_ID')
# print(cust_meal1.shape)
# print(cust_meal2.shape)
# customer['CUSTOMER_ID'] = customer.CUSTOMER_ID.astype("object")
# print(cust_meal1.isna().sum())
# print(cust_meal2.isna().sum())
# del cust_meal1
# del cust_meal2

cust_meal = pd.merge(customer, meal, how='outer', on='CUSTOMER_ID')
print(cust_meal.shape)
print(cust_meal.isna().sum())

print(cust_meal.dtypes)
del customer

major_cust_list = meal[meal['SELL_DATE'] > '2019-05-01'].sort_values(['SELL_DATE'], ascending=[True])
print(major_cust_list.shape)
del meal
major_cust_list = list(major_cust_list['CUSTOMER_ID'].unique())
print(len(major_cust_list))
print(cust_meal.shape)
cust_meal = cust_meal[cust_meal['CUSTOMER_ID'].isin(major_cust_list)]
del major_cust_list
print(cust_meal.shape)
print(cust_meal.isna().sum())

cust_meal['year'] = cust_meal.SELL_DATE.str.slice(start=0, stop=4)
cust_meal['month'] = cust_meal.SELL_DATE.str.slice(start=5, stop=7)
# cust_meal['day'] = cust_meal.SELL_DATE.str.slice(start=8, stop=10)

cust_meal['year'] = cust_meal.year.astype("int")
cust_meal['month'] = cust_meal.month.astype("int")
# cust_meal['day'] = cust_meal.day.astype("int")

cust_meal['QUANTITY'] = cust_meal.QUANTITY.astype("int")
cust_meal['PRICE'] = cust_meal.PRICE.astype("int")
# cust_meal['CUSTOMER_ID'] = cust_meal.CUSTOMER_ID.astype("object")
# cust_meal['CUSTOMER_ID'] = cust_meal.CUSTOMER_ID.astype("int")
print(cust_meal.dtypes)

cust_meal_weather = pd.merge(cust_meal, weather_log, how='left', left_on='SELL_DATE', right_on='date')
del cust_meal
del weather_log
print(cust_meal_weather.isna().sum())
print(cust_meal_weather.shape)
cust_meal_weather['SELL_DATE_dt'] = pd.to_datetime(cust_meal_weather['date'], format='%Y-%m-%d')
cust_meal_weather['weekday'] = cust_meal_weather['SELL_DATE_dt'].dt.dayofweek  # Monday=0, Sunday=6.  # 더미변수로 두는 게 일단은 맞다.
print(cust_meal_weather.dtypes)
cust_meal_weather['weekday'] = cust_meal_weather.weekday.astype("int")


cust_meal_weather['weekday'] = cust_meal_weather[cust_meal_weather.weekday != 5]




cust_meal_weather['weekday'] = cust_meal_weather[cust_meal_weather['weekday' != 6]]
print(cust_meal_weather.shape)
cust_meal_weather.drop(['SELL_DATE_dt'], axis=1, inplace=True)
print(cust_meal_weather.isna().sum())
print(cust_meal_weather.columns)

cust_meal_weather['korean_age'] = cust_meal_weather['year'] - cust_meal_weather['BIRTH_YEAR'] + 1
cust_meal_weather = cust_meal_weather.loc[:, cust_meal_weather.columns != 'MENU']
print(cust_meal_weather.columns)
print(cust_meal_weather.dtypes)

describe = cust_meal_weather.describe()
print(cust_meal_weather.describe())
print(cust_meal_weather.info())

cust_meal_weather.CUSTOMER_ID = cust_meal_weather.CUSTOMER_ID.astype("category")
cust_meal_weather.GENDER = cust_meal_weather.GENDER.astype("category")
cust_meal_weather.BIRTH_YEAR = cust_meal_weather.BIRTH_YEAR.astype("int16")
cust_meal_weather.drop(['SELL_DATE'], axis=1, inplace=True)
cust_meal_weather.BRAND = cust_meal_weather.BRAND.astype("category")
cust_meal_weather.PRICE = cust_meal_weather.PRICE.astype("int16")
cust_meal_weather.QUANTITY = cust_meal_weather.QUANTITY.astype("int8")
# cust_meal_weather.year = cust_meal_weather.year.astype("int16")
cust_meal_weather.drop(['year'], axis=1, inplace=True)
cust_meal_weather.month = cust_meal_weather.month.astype("int8")
# cust_meal_weather.day = cust_meal_weather.day.astype("int8")
# cust_meal_weather.drop(['day'], axis=1, inplace=True)
cust_meal_weather.drop(['date'], axis=1, inplace=True)
cust_meal_weather.max_temper = cust_meal_weather.max_temper.astype("float32")
cust_meal_weather.min_temper = cust_meal_weather.min_temper.astype("float32")
cust_meal_weather.rainfall = cust_meal_weather.rainfall.astype("float32")
cust_meal_weather.weekday = cust_meal_weather.weekday.astype("int8")
cust_meal_weather.korean_age = cust_meal_weather.max_temper.astype("int8")


# CUSTOMER_ID    1095635 non-null int32
# GENDER         1095635 non-null category
# BIRTH_YEAR     1095635 non-null int16
# SELL_DATE      1095635 non-null object
# BRAND          1095635 non-null category
# PRICE          1095635 non-null int16
# QUANTITY       1095635 non-null int8
# year           1095635 non-null int16
# month          1095635 non-null int8
# day            1095635 non-null int8
# max_temper     1095635 non-null float32
# min_temper     1095635 non-null float32
# rainfall       1095635 non-null float32
# snow_depth     1095635 non-null float64
# weekday        1095635 non-null int8
# korean_age     1095635 non-null int8
print(cust_meal_weather.info())

onehot_CUSTOMER_ID = pd.get_dummies(cust_meal_weather['CUSTOMER_ID'], sparse=True)
onehot_CUSTOMER_ID.info()
onehot_CUSTOMER_ID = onehot_CUSTOMER_ID.to_coo()













sparse_cust_meal_weather = pd.concat(onehot_CUSTOMER_ID, cust_meal_weather.loc[:, cust_meal_weather.columns != 'CUSTOMER_ID'], axis=1).density()

onehot_CUSTOMER_ID = sparse.csr_matrix(pd.get_dummies(cust_meal_weather['CUSTOMER_ID'], sparse=True))

onehot_encoder = OneHotEncoder(sparse=True)
label_encoder = LabelEncoder(sparse=True)

onehot_encoded = sparse.csr_matrix(onehot_encoder.fit_transform(cust_meal_weather['GENDER'].values.reshape(-1, 1)))
# onehot_encoded = sparse.csr_matrix(onehot_encoder.fit_transform(cust_meal_weather['CUSTOMER_ID'].values.reshape(-1, 1)))




train_table = cust_meal_weather[cust_meal_weather['date'] < '2019-01-01']
print(train_table.shape)
test_table = cust_meal_weather[cust_meal_weather['date'] >= '2019-01-01']
print(test_table.shape)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(customer['CUSTOMER_ID'].values.reshape(-1,1))
dataset = sparse.csr_matrix(onehot_encoder.fit_transform(customer['CUSTOMER_ID'].values.reshape(-1,1)))
for i in range(customer.ix[:, 1:].shape[1]):  # tf-idf matrix 앞에다 열 하나씩 붙이는데 거꾸로 붙이는 거라 뒤에서 붙어 붙였다. 그래서 -i
    dataset = np.insert(dataset, 1, customer.ix[:, -i].values, axis=1)


