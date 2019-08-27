import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def filter_sat_sun_day_and_attach_weekday(df, col):
    df['SELL_DATE_dt'] = pd.to_datetime(df[col], format='%Y-%m-%d')
    df['weekday'] = df['SELL_DATE_dt'].dt.dayofweek
    df = df[df.weekday.isin(list(range(5)))]
    df.drop(['SELL_DATE_dt'], axis=1, inplace=True)
    return df


def cross_join(left, right):
    # df1['key'] = 0
    # df2['key'] = 0
    # df = df1.merge(df2, how='outer')
    # df.drop(['key'], axis=1, inplace=True)
    # return df
    # return (left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))
    la, lb = len(left), len(right)
    ia2, ib2 = np.broadcast_arrays(*np.ogrid[:la,:lb])
    return pd.DataFrame(np.column_stack([left.values[ia2.ravel()], right.values[ib2.ravel()]]),
                        columns=left.columns.append(right.columns))


def add_one_hot_encoding(df, col):
    onehot = pd.get_dummies(df[col], sparse=True)
    # print(onehot.head())
    df.drop([col], axis=1, inplace=True)
    df = pd.concat([df, onehot], axis=1)
    return df


def add_label_encoding(df, col):
    encoder = LabelEncoder()
    encoder.fit(df[col])
    encoded_Y = encoder.transform(df[col])
    print(encoded_Y)
    encoded_Y = pd.DataFrame(data=encoded_Y, columns=[col])
    print(encoded_Y.head())
    df.drop([col], axis=1, inplace=True)
    df = pd.concat([df, encoded_Y], axis=1)
    return df


def change_scale_to_minmax(df, col):
    scaler = MinMaxScaler()
    df[col] = scaler.fit_transform(df['col'])
    return df


def extract_DerivedVariable_from_date(df):  # 한국식 나이까지 계산하도록 BIRTH_YEAR 포함될 것. date 변수에 날짜가 있을 것.
    df['year'] = df.date.str.slice(start=0, stop=4)
    df['month'] = df.date.str.slice(start=5, stop=7)

    df['year'] = df.year.astype("int")
    df['month'] = df.month.astype("int")

    df['korean_age'] = df['year'] - df['BIRTH_YEAR'] + 1
    df.drop(['year'], axis=1, inplace=True)  # 이제 필요 없음.
    df.drop(['BIRTH_YEAR'], axis=1, inplace=True)  # 이제 필요 없음.

    df['month_day'] = df.date.str.slice(start=5, stop=10)
    return df


customer = pd.read_csv('mealData_customer.csv')
customer0525_0731 = pd.read_csv('mealData_customer_0525_0731.csv')
# customer['CUSTOMER_ID'] = customer.CUSTOMER_ID.astype("category")

meal = pd.read_csv('mealData_meal.csv')
meal_0525_0731 = pd.read_csv('mealData_meal_0525_0731.csv')
# meal['CUSTOMER_ID'] = meal.CUSTOMER_ID.astype("category")  # 다른 결과 값이 나온다.

weather_log = pd.read_csv('20190811171205_weather_log.csv')
weather_after_crawling = pd.read_csv('weather_2019_08_11.csv')

### 크게 3종의 데이터셋을 합해야한다. 그 순서는 중요함.
##### 고객 데이터(고객) #####
# print(customer.shape)
# print(customer0525_0731.shape)
customer = pd.concat([customer, customer0525_0731], axis=0)
# print(customer.shape)
customer = customer.drop_duplicates()
print(customer.shape)
del customer0525_0731

##### 구매 데이터(일자, 고객) #####
# print(meal.shape)
meal = pd.concat([meal, meal_0525_0731], axis=0)
print(meal.shape)
print("동일인이 두번 이상 구매했는데 quantity에 2가 아닌 경우 ", meal[meal.duplicated(keep=False)].shape)  # 중복 확인
print("전체 비중을 생각해보면 ", meal[meal.duplicated(keep=False)].shape[0]/meal.shape[0])
print('고로 생략')
meal = meal.drop_duplicates()  # 한 사람이 같은 메뉴를 다른사람에게 사주느라고 이런 거 같은데 2번 산걸로 카운팅이 되지 않음. 일단 선호가 높다고 간주하고 빈도를 높이는 것으로 감.
# 10032건
# print(meal.shape)
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


##### 날씨 데이터(일자) #####
# print(weather_log.shape)
weather_log = pd.concat([weather_log, weather_after_crawling], axis=0)
print(weather_log.shape)
del weather_after_crawling

##### 손님 정보와 구매정보를 매칭 #####
cust_meal = pd.merge(meal, customer, on='CUSTOMER_ID')
# cust_meal_outer = pd.merge(meal, customer, how='outer', on='CUSTOMER_ID')
print(cust_meal.shape)
# print(cust_meal.isna().sum())
# print(cust_meal_outer.shape)
# print(cust_meal_outer.isna().sum())
# print(cust_meal_outer.shape[0]-cust_meal.shape[0])

# print(cust_meal.dtypes)

##### 5~7월 동안 먹은 사람 한정(어차피 3개월간 안먹었으면 이후에도 안 먹을 가능성이 높다고 판단) #####
# 그외의 사람들 것은 평균적인 경향을 알 수 있긴 한데 연산자원이 너무 들어서 넣은 항목
# 생각이 좀 바뀜. 일단 구매이력이든, 고객 정보든 뭐든 가진 사람의 것을 훈련에 사용. 어차피 none이 더 많을 것.

major_cust_list = meal[meal['SELL_DATE'] > '2019-05-01'].sort_values(['SELL_DATE'], ascending=[True])
major_cust_list = list(major_cust_list['CUSTOMER_ID'].unique())
print(len(cust_meal.CUSTOMER_ID.unique()), '에서')
print(len(major_cust_list), '으로 변함. ',len(cust_meal.CUSTOMER_ID.unique())-len(major_cust_list),'명 정도가 사라진 셈.')
# print(7402*(31+11-1-9-3))  # 8월1일부터 9월 11까지 예측대상 일수는 29일. 예상대로라면 여기에 7402명의 사람들의 반응, 214658이 들어가야 한다.
major_cust = pd.DataFrame(data=cust_meal['CUSTOMER_ID'].unique(), columns=['CUSTOMER_ID'])
print(major_cust.shape)
# print(len(major_cust_list))
# print(cust_meal.shape)
# cust_meal = cust_meal[cust_meal['CUSTOMER_ID'].isin(major_cust_list)]
del major_cust_list
# print(cust_meal.shape)
# print(cust_meal.isna().sum())

really_eating_date_at_train = filter_sat_sun_day_and_attach_weekday(cust_meal, 'SELL_DATE')
really_eating_date_at_train = pd.DataFrame(data=really_eating_date_at_train['SELL_DATE'].unique(),
                                           columns=['SELL_DATE'])
print(really_eating_date_at_train.shape)
# really_eating_date_at_train.drop(['key'], axis=1, inplace=True)
# major_cust.drop(['key'], axis=1, inplace=True)

# really_eating_date_at_train['SELL_DATE'] = really_eating_date_at_train.SELL_DATE.astype("category")
# major_cust['CUSTOMER_ID'] = major_cust.CUSTOMER_ID.astype("category")

date_cust = cross_join(really_eating_date_at_train, major_cust)  # 389*10794=4198866 경우의 수는 이것이고 나머지는 변치 않는 것이 맞다.
print(date_cust.shape)
del really_eating_date_at_train
date_cust.rename(columns={"SELL_DATE": "date"}, inplace=True)
date_cust_weather = date_cust.merge(weather_log, how='left', on='date')
print(date_cust_weather.dtypes)
print(date_cust_weather.shape)
print(date_cust_weather.isna().sum())
# print(date_cust_weather.dtypes)
date_cust_weather = date_cust_weather.merge(customer, how='left', on='CUSTOMER_ID')
print(date_cust_weather.shape)

meal.rename(columns={'SELL_DATE': "date"}, inplace=True)
date_cust_weather_meal = pd.merge(date_cust_weather, meal, how='left', on=['date', 'CUSTOMER_ID'])  # 이 과정에서 같은 걸 두번 산 사람은 한번만 매칭되는게 아니라 반복해서 매칭되므로 사전에 지워야하는 것으로 보임..
print(date_cust_weather_meal.shape)
date_cust_weather_meal['BRAND'].fillna(value='none', inplace=True)
print(date_cust_weather_meal.isna().sum())
# del date_cust
# del date_cust_weather
del meal

###### test table ######
test_table = cross_join(weather_log, major_cust)
test_table = test_table[(test_table['date'] >= '2019-08-01')
                        & (test_table['date'] < '2019-09-12')
                        & (test_table['date'] != '2019-08-15')]

test_table = filter_sat_sun_day_and_attach_weekday(test_table, 'date')
print(test_table.shape)
print(test_table.shape[0]/10794)  # 29이 나옴.

test_table = test_table.merge(customer)  # 당연한 이야기지만 이 시점에선 MENU, PRICE, QUNTITY는 무의미하니 meal은 추가적으로 join안함.
print(test_table.isna().sum())
test_table = extract_DerivedVariable_from_date(test_table)
print(test_table.dtypes)

test_table.CUSTOMER_ID = test_table.CUSTOMER_ID.astype("category")  #
test_table.GENDER = test_table.GENDER.astype("category")  #
test_table.month = test_table.month.astype("int8")  #
test_table.max_temper = test_table.max_temper.astype("float32")  #
test_table.min_temper = test_table.min_temper.astype("float32")  #
test_table.rainfall = test_table.rainfall.astype("float32")  #
test_table.snow_depth = test_table.snow_depth.astype("float32")  #
test_table.weekday = test_table.weekday.astype("int8")  #
test_table.korean_age = test_table.korean_age.astype("int8")  #
test_table.month_day = test_table.month_day.astype("category")  #
print(test_table.info())


# date           실제론 기계학습에선 안씀. 결과 뱉을 때 쓸 것. test table이 쓸모 있음.
# max_temper     사용
# min_temper     사용
# rainfall       사용
# snow_depth     사용
# CUSTOMER_ID    onehot으로 바꿔야할것.
# weekday        onehot으로 바꿔야할것.
# GENDER         onehot으로 바꿔야할것.
# month          onehot으로 바꿔야할것.
# korean_age     사용
# month_day      onehot으로 바꿔야할것.

test_table_addonehot = add_one_hot_encoding(test_table, 'GENDER')
test_table_addonehot = add_one_hot_encoding(test_table_addonehot, 'weekday')
test_table_addonehot = add_one_hot_encoding(test_table_addonehot, 'month')
test_table_addonehot = add_one_hot_encoding(test_table_addonehot, 'month_day')
test_table_addonehot = add_one_hot_encoding(test_table_addonehot, 'CUSTOMER_ID')
print(len(test_table_addonehot.columns))
###### test table ######


# describe = date_cust_weather_meal.describe()
# print(date_cust_weather_meal.describe())

date_cust_weather_meal = filter_sat_sun_day_and_attach_weekday(date_cust_weather_meal, 'date')
date_cust_weather_meal = extract_DerivedVariable_from_date(date_cust_weather_meal)

# date_cust_weather_meal.drop(['date'], axis=1, inplace=True)  # train, validation 데이터셋으로 나누기 위함.
date_cust_weather_meal.CUSTOMER_ID = date_cust_weather_meal.CUSTOMER_ID.astype("category")
date_cust_weather_meal.max_temper = date_cust_weather_meal.max_temper.astype("float32")
date_cust_weather_meal.min_temper = date_cust_weather_meal.min_temper.astype("float32")
date_cust_weather_meal.rainfall = date_cust_weather_meal.rainfall.astype("float32")
date_cust_weather_meal.snow_depth = date_cust_weather_meal.snow_depth.astype("float32")
date_cust_weather_meal.GENDER = date_cust_weather_meal.GENDER.astype("category")
date_cust_weather_meal.BRAND = date_cust_weather_meal.BRAND.astype("category")
date_cust_weather_meal.drop(['MENU'], axis=1, inplace=True)
date_cust_weather_meal.drop(['PRICE'], axis=1, inplace=True)
date_cust_weather_meal.drop(['QUANTITY'], axis=1, inplace=True)
date_cust_weather_meal.weekday = date_cust_weather_meal.weekday.astype("int8")
date_cust_weather_meal.month = date_cust_weather_meal.month.astype("int8")
date_cust_weather_meal.korean_age = date_cust_weather_meal.korean_age.astype("int8")
date_cust_weather_meal.month_day = date_cust_weather_meal.month_day.astype("category")
print(date_cust_weather_meal.info())

# date           object
# CUSTOMER_ID    category
# max_temper     float32
# min_temper     float32
# rainfall       float32
# snow_depth     float32
# GENDER         category
# BRAND          category
# weekday        int8
# month          int8
# korean_age     int8
# month_day      category


# date           실제론 기계학습에선 안씀. 결과 뱉을 때 쓸 것. test table에겐 쓸모 있음.
# CUSTOMER_ID    onehot으로 바꿔야할것.
# max_temper     사용
# min_temper     사용
# rainfall       사용
# snow_depth     사용
# GENDER         onehot으로 바꿔야할것.

# weekday        onehot으로 바꿔야할것.
# month          onehot으로 바꿔야할것.
# korean_age     사용
# month_day      onehot으로 바꿔야할것.

train_table = date_cust_weather_meal[date_cust_weather_meal['date'] < '2019-01-01']
print(train_table.shape)
valid_table = date_cust_weather_meal[date_cust_weather_meal['date'] >= '2019-01-01']
print(valid_table.shape)

train_table = add_label_encoding(train_table, 'GENDER')

train_table = add_one_hot_encoding(train_table, 'GENDER')
train_table = add_one_hot_encoding(train_table, 'month_day')
train_table = add_one_hot_encoding(train_table, 'weekday')
train_table = add_one_hot_encoding(train_table, 'month')
train_table = add_one_hot_encoding(train_table, 'CUSTOMER_ID')

valid_table = add_one_hot_encoding(valid_table, 'GENDER')
valid_table = add_one_hot_encoding(valid_table, 'month_day')
valid_table = add_one_hot_encoding(valid_table, 'weekday')
valid_table = add_one_hot_encoding(valid_table, 'month')
valid_table = add_one_hot_encoding(valid_table, 'CUSTOMER_ID')

print(train_table.info())
print(valid_table.info())

#onehot_encoder = OneHotEncoder(sparse=True)
#label_encoder = LabelEncoder(sparse=True)


"""
scaler = MinMaxScaler()
cust_meal_weather.min_temper = scaler.fit_transform(cust_meal_weather.min_temper)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(customer['CUSTOMER_ID'].values.reshape(-1,1))
dataset = sparse.csr_matrix(onehot_encoder.fit_transform(customer['CUSTOMER_ID'].values.reshape(-1,1)))
for i in range(customer.ix[:, 1:].shape[1]):  # tf-idf matrix 앞에다 열 하나씩 붙이는데 거꾸로 붙이는 거라 뒤에서 붙어 붙였다. 그래서 -i
    dataset = np.insert(dataset, 1, customer.ix[:, -i].values, axis=1)

"""


""" backup code
import pandas as pd
df = inputs[0]
# date           실제론 기계학습에선 안씀. 결과 뱉을 때 쓸 것. test table에겐 쓸모 있음.
# CUSTOMER_ID    onehot으로 바꿔야할것.
# max_temper     사용
# min_temper     사용
# rainfall       사용
# snow_depth     사용
# GENDER         onehot으로 바꿔야할것.

# weekday        onehot으로 바꿔야할것.
# month          onehot으로 바꿔야할것.
# korean_age     사용
# month_day      onehot으로 바꿔야할것.

# CUSTOMER_ID_left는 컬럼명 충돌 대비해 나중에 처리
df.CUSTOMER_ID = df.CUSTOMER_ID.astype("category")
# date_left는 컬럼명 충돌 대비해 나중에 처리
df.drop(['date_right'], axis=1, inplace=True)
df.max_temper = df.max_temper.astype("float32")
df.min_temper = df.min_temper.astype("float32")
df.rainfall = df.rainfall.astype("float32")
df.snow_depth = df.snow_depth.astype("float32")
df.drop(['CUSTOMER_ID_right'], axis=1, inplace=True)
df.GENDER = df.GENDER.astype("category")
df.drop(['date'], axis=1, inplace=True)
df.rename(columns={"date_left": "date"}, inplace=True) #
df.drop(['CUSTOMER_ID'], axis=1, inplace=True)
df.rename(columns={"CUSTOMER_ID_left": "CUSTOMER_ID"}, inplace=True) # 
df.BRAND = df.BRAND.astype("category")
#df.drop(['MENU'], axis=1, inplace=True)
#df.drop(['PRICE'], axis=1, inplace=True)
#df.drop(['QUANTITY'], axis=1, inplace=True)
df.weekday = df.weekday.astype("int8")
df.month = df.month.astype("int8")
df.korean_age = df.korean_age.astype("int8")
df.month_day = df.month_day.astype("category")




date_cust_weather_meal = df

def add_one_hot_encoding(df, col):
    onehot = pd.get_dummies(df[col], sparse=True)
    # print(onehot.head())
    df.drop([col], axis=1, inplace=True)
    df = pd.concat([df, onehot], axis=1)
    return df
  
train_table = date_cust_weather_meal[date_cust_weather_meal['date'] < '2019-01-01']
print(train_table.shape)
valid_table = date_cust_weather_meal[date_cust_weather_meal['date'] >= '2019-01-01']
print(valid_table.shape)

train_table = add_one_hot_encoding(train_table, 'GENDER')
train_table = add_one_hot_encoding(train_table, 'month_day')
train_table = add_one_hot_encoding(train_table, 'weekday')
train_table = add_one_hot_encoding(train_table, 'month')
train_table = add_one_hot_encoding(train_table, 'CUSTOMER_ID')

valid_table = add_one_hot_encoding(valid_table, 'GENDER')
valid_table = add_one_hot_encoding(valid_table, 'month_day')
valid_table = add_one_hot_encoding(valid_table, 'weekday')
valid_table = add_one_hot_encoding(valid_table, 'month')
valid_table = add_one_hot_encoding(valid_table, 'CUSTOMER_ID')
"""