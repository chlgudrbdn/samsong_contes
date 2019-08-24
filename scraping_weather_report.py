from selenium import webdriver
import pandas as pd
from datetime import datetime, timedelta

crawl_start_time = datetime.now()
print("crawl_start_time : ", crawl_start_time)

global driver
driver = webdriver.Chrome()

driver.get('https://www.accuweather.com/ko/kr/songpa-gu/2330444/august-weather/2330444?monyr=8/1/2019&view=table')
# 8월 먼저. 어차피 제출일 생각하면 8월 이후 데이터만 사용할 수 있다.
df = pd.DataFrame()
# driver.find_element_by_xpath('/html/body/div/div[5]/div/div[1]/div/div[1]/div[2]/a[2]').click()
driver.find_element_by_class_name('btri-view-list').click()


tbody = driver.find_elements_by_class_name('calendar-list')[0]
i = 0
for row in tbody.find_elements_by_class_name('calendar-list-cl-tr'):
    date_wether = row.find_element_by_tag_name('th')
    date_weekday = date_wether.text  # 토 08-31
    if i == 0:
        begin_date = '2019_'+date_weekday.split()[1].split('-')[0]+"_"+date_weekday.split()[1].split('-')[1]
    df.loc[i, 'date'] = '2019-'+date_weekday.split()[1]
    # weekday_list = ['월', '화', '수', '목', '금', '토']
    # df.loc[i, 'weekday'] = date_weekday.split()[0]
    cell_list = row.find_elements_by_tag_name('td')

    temperature = cell_list[0].text.replace('°', '').split('/')
    df.loc[i, 'max_temper'] = temperature[0]  # 공시대상회사(종목명)
    df.loc[i, 'min_temper'] = temperature[1]  # 공시대상회사(종목명)
    df.loc[i, 'rainfall'] = cell_list[1].text.split()[0]  # 공시대상회사(종목명)
    df.loc[i, 'snow_depth'] = cell_list[2].text.split()[0]  # 공시대상회사(종목명)

    i += 1

driver.get('https://www.accuweather.com/ko/kr/songpa-gu/2330444/september-weather/2330444?monyr=9/1/2019&view=table')
tbody = driver.find_elements_by_class_name('calendar-list')[0]
for row in tbody.find_elements_by_class_name('calendar-list-cl-tr'):
    date_wether = row.find_element_by_tag_name('th')
    date_weekday = date_wether.text  # 토 08-31
    df.loc[i, 'date'] = '2019-'+date_weekday.split()[1]
    cell_list = row.find_elements_by_tag_name('td')

    temperature = cell_list[0].text.replace('°', '').split('/')
    df.loc[i, 'max_temper'] = int(temperature[0])
    df.loc[i, 'min_temper'] = int(temperature[1])
    df.loc[i, 'rainfall'] = int(cell_list[1].text.split()[0])
    df.loc[i, 'snow_depth'] = int(cell_list[2].text.split()[0])

    i += 1

print("take time : {}".format(datetime.now() - crawl_start_time))
# df.drop(df.index[0], inplace=True)
df.to_csv('weather_'+begin_date+'.csv', index=False)
print(df.dtypes)
# begin_date = '2019-08-11'
driver.quit()

