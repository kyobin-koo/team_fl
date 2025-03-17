from nycflights13 import flights,planes
import pandas as pd 
# 설치 : conda install matplotlib
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns

flights.info()
planes.info()

planes.head()
flights.head()

# 주제 자유
""""
주제 : 항공사별 지연 원인 분석

# 주말, 주중, 시간대별 항공사의 지연 현황 파악

# 지연이 많이 되는 항공사가 많이 이용하는 기종 및 제조사가 무엇인지.

# 지연이 많이 되는 비행기 기종 / 엔진 종류 / 모델 파악

# 지연이 많이 되는 비행기 기종은 연식이 오래 됐을 것이다/ 엔진 양이 적을 것이다
   -> 오랜 거리 비행을 하지 않을 것이다.

"""


# merge 사용해서 데이터 합치기
# 각 변수 최소 하나씩 선택 후 분석
# 날짜&시간 전처리 코드 들어갈 것
# 문자열 전처리 코드 들어갈 것
# 시각화 종류 최소 3개 이상



### 결측치 처리
# speed 열 처리
flights.isna().sum()
planes.isna().sum()

planes=planes.drop("speed", axis=1)

# 도착 지연 채우기
f =flights.dropna()
f.isna().sum()

# year 채우기
p=planes.dropna()



###데이터 합치기
df=pd.merge(f,p,on="tailnum", how="inner")
df.isna().sum()
df.info()


### 필요없는 컬럼 제거
df=df.drop(columns=["sched_dep_time","time_hour"])
df
df.columns

### 컬럼명 제거
rename={"year_x":"year","year_y" : "man_year",
        "time":"sched_dep_time","hour":"sched_dep_hour",
        "minute":"sched_dep_minute"}
df=df.rename(columns=rename)
df.columns

df



### 소주제 1: 주말, 주중, 시간대별 항공사의 지연 현황 파악
# 데이트 추가
df["date"]=pd.to_datetime(df["year"].astype("str")+df["month"].astype("str")+df["day"].astype("str"),format="%Y%m%d")
df.info()
# 출발 지연이된 데이터 프레임
dep_delay_df=df.loc[df["dep_delay"]>0,:]
dep_delay_df.columns

dep_delay_df["day"]=dep_delay_df["date"].dt.day_name()
dep_delay_df['weekday_num'] = dep_delay_df['date'].dt.weekday 
dep_delay_weekday=dep_delay_df.loc[dep_delay_df['weekday_num']<5]

# 주중별 
dep_delay_weekday.groupby(["carrier","weekday_num"],as_index=False)[["sched_dep_hour","carrier"]].count()

dep_weekday_mean = pd.pivot_table(dep_delay_weekday,
                             index=["carrier","weekday_num"],
                             values="sched_dep_hour",
                             aggfunc="mean").reset_index()

dep_weekday_median = pd.pivot_table(dep_delay_weekday,
                             index=["carrier","weekday_num"],
                             values="sched_dep_hour",
                             aggfunc="median").reset_index()


dep_weekday_mean.sort_values(by="weekday_num",ascending=False)
dep_weekday_median.sort_values(by="weekday_num",ascending=False)












