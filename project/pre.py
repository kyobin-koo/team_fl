import numpy as np
import pandas as pd
from nycflights13 import flights,planes
flights.info()
planes.info()

# merge 사용해서 flights와 planes 병합한 데이터로 각 데이터 변수 최소 하나씩 선택 후 분석
# 날짜&시간 전처리 코드
# 문자열 전처리 코드
# 시각화 종류 최소 3개



df = pd.merge(flights,planes,on='tailnum',how='inner')
df.drop(columns=['sched_dep_time','time_hour'])
df = df.rename(columns={'year_x':'year', 'year_y':'man_year', 'minute': 'sched_dep_minute', 'hour' :'sched_dep_hour'})
df.info()



## 남원정








# flights["carrier"].value_counts()
# planes["manufacturer"].nunique()
# df
# df.pivot_table(index = 'carrier', values='dep_delay')
# df.pivot_table(index = 'carrier', columns = 'manufacturer' ,values='dep_delay')
# df["type"].value_counts()
# df["model"].value_counts()



# 김예원

