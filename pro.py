import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nycflights13 import flights,planes
# del planes['speed']
from matplotlib import font_manager, rc



del planes['speed']

### 운영체제 확인 라이브러리
import platform

### 시각화 시 마이너스(-, 음수) 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

### OS별 한글처리
# - 윈도우 운영체게
if platform.system() == "Windows" :
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc("font", family = font_name)


flights.isna().sum()
planes.isna().sum()

#flights 결측치 싹다 제거 
f = flights.dropna()
f.isna().sum()
#planes 결측치 제거 
p = planes.dropna()
p.isna().sum()

#merge 및 중복열 삭제, 변수명 변경
df =pd.merge(f,p,on='tailnum',how='inner')
df.isna().sum()
df = df.drop(columns=['sched_dep_time','time_hour'])
df = df.rename(columns = {'year_x':'year','year_y':'man_year','hour':'sched_dep_hour','minute':'sched_dep_minute'})





########################################################################################################
# 데이터 현황 분석
########################################################################################################
df.info()
df.describe()
df["dep_delay"].describe()


# 출발 지연 시간 분포
#지수 비율 3:7인 경우, 출발 지연 된 변수들 중 dep_delay를 함 
plt.figure(figsize=(12, 6))
# sns.histplot(np.log1p(df["dep_delay"]), bins=50, kde=True, color="blue")
sns.histplot((df["dep_delay"]), bins=50, kde=True, color="blue")
plt.title("출발 지연 시간 분포")
plt.xlabel("출발 지연 시간 (분)")
plt.ylabel("항공편 수")
# plt.xlim(-10, 300)  # 극단적인 이상치는 제외하고 가시성 높이기
plt.grid()
plt.show()




# 이상치 확인
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["dep_delay"], color="red")
plt.title("출발 지연 시간 분포 (박스 플롯)")
plt.xlabel("출발 지연 시간 (분)")
plt.xlim(-10, 300)  # 이상치가 너무 많으면 가시성 좋게 제한
plt.grid()
plt.show()



########################################################################################################
# 일별 지연시간 분석
########################################################################################################


df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df['day_of_week'] = df['date'].dt.dayofweek

# 요일별 평균 출발 지연시간 계산
weekday_delays = df.groupby('day_of_week')['dep_delay'].mean().reset_index()

# 요일 이름 매핑
weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_delays['day_of_week'] = weekday_delays['day_of_week'].map(lambda x: weekday_labels[x])

# 시각화
plt.figure(figsize=(10, 5))
plt.bar(weekday_delays['day_of_week'], weekday_delays['dep_delay'], color='skyblue', alpha=0.8)
plt.xlabel('Day of the Week')
plt.ylabel('Average Departure Delay (minutes)')
plt.title('Average Departure Delay by Day of the Week')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


########################################################################################################
# 월별 지연 트렌드
########################################################################################################

mean_month=df.groupby("month")["dep_delay"].mean().reset_index()


# 첫 번째 그래프: 월별 평균 dep_delay
plt.figure(figsize=(10, 5))
plt.bar(mean_month["month"], mean_month["dep_delay"], color='skyblue')
plt.xlabel("Month")
plt.ylabel("Average Departure Delay")
plt.title("Average Departure Delay by Month")
plt.show()


df['carrier'].value_counts()
########################################################################################################
#항공사별 dep_delay 열의 평균/중앙값 파악
########################################################################################################


df.groupby('carrier')['dep_delay'].mean().sort_values(ascending=False)
df.groupby('carrier')['dep_delay'].median().sort_values(ascending=False)





########################################################################################################
# 데이터 현황 분석
########################################################################################################
df.info()
df.describe()
df["dep_delay"].describe()


# 출발 지연 시간 분포
plt.figure(figsize=(12, 6))
sns.histplot(df["dep_delay"], bins=50, kde=True, color="blue")
plt.title("출발 지연 시간 분포")
plt.xlabel("출발 지연 시간 (분)")
plt.ylabel("항공편 수")
plt.xlim(-10, 300)  # 극단적인 이상치는 제외하고 가시성 높이기
plt.grid()
plt.show()


# 이상치 확인
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["dep_delay"], color="red")
plt.title("출발 지연 시간 분포 (박스 플롯)")
plt.xlabel("출발 지연 시간 (분)")
plt.xlim(-10, 300)  # 이상치가 너무 많으면 가시성 좋게 제한
plt.grid()
plt.show()



########################################################################################################
# 일별 지연시간 분석
########################################################################################################


df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df['day_of_week'] = df['date'].dt.dayofweek

# 요일별 평균 출발 지연시간 계산
weekday_delays = df.groupby('day_of_week')['dep_delay'].mean().reset_index()

# 요일 이름 매핑
weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_delays['day_of_week'] = weekday_delays['day_of_week'].map(lambda x: weekday_labels[x])

# 시각화
plt.figure(figsize=(10, 5))
plt.bar(weekday_delays['day_of_week'], weekday_delays['dep_delay'], color='skyblue', alpha=0.8)
plt.xlabel('Day of the Week')
plt.ylabel('Average Departure Delay (minutes)')
plt.title('Average Departure Delay by Day of the Week')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


########################################################################################################
# 월별 지연 트렌드
########################################################################################################

mean_month=df.groupby("month")["dep_delay"].mean().reset_index()



########################################################################################################
#항공사별 dep_delay 열의 평균/중앙값 파악
########################################################################################################


df.groupby('carrier')['dep_delay'].mean().sort_values(ascending=False)
df.groupby('carrier')['dep_delay'].median().sort_values(ascending=False)

df.groupby('carrier')['dep_delay'].max().sort_values(ascending=False)
############################################################################################
delay_over_180 = df.loc[df['dep_delay'] >= 180]
df_count = df['carrier'].value_counts().reset_index()



# 180 이상
delay_over_180_count = delay_over_180['carrier'].value_counts().reset_index()
delay_over_180_count_merge = pd.merge(delay_over_180_count,df_count,on='carrier',how='outer')
delay_over_180_ratio = pd.DataFrame((delay_over_180_count_merge.iloc[:,1] / delay_over_180_count_merge.iloc[:,2]).sort_index(ascending=True))
ratio_over_180 = pd.concat([delay_over_180_count_merge,delay_over_180_ratio],axis=1)
ratio_over_180 = ratio_over_180.rename(columns = {'count_x':'dep_delay_count','count_y': 'total_count',0:'ratio'})
ratio_over_180.sort_values('ratio',ascending=False)

over_180_df = pd.merge(ratio_over_180,delay_over_180.groupby('carrier')['dep_delay'].mean(),on='carrier',how='outer')
over_180_df = over_180_df.fillna(0)

#180 이하
delay_under_180 = df.loc[(df['dep_delay'] > 0) & (df['dep_delay'] <180)]
delay_under_180_count = delay_under_180['carrier'].value_counts().reset_index()
delay_under_180_count_merge = pd.merge(delay_under_180_count,df_count,on='carrier',how='outer')
delay_under_180_ratio = pd.DataFrame((delay_under_180_count_merge.iloc[:,1] / delay_under_180_count_merge.iloc[:,2]).sort_index(ascending=True))
ratio_under_180 = pd.concat([delay_under_180_count_merge,delay_under_180_ratio],axis=1)
ratio_under_180 = ratio_under_180.rename(columns = {'count_x':'dep_delay_count','count_y': 'total_count',0:'ratio'})
ratio_under_180.sort_values('ratio',ascending=False)

under_180_df = pd.merge(ratio_under_180,delay_under_180.groupby('carrier')['dep_delay'].mean(),on='carrier',how='outer')

merged_df = pd.concat([over_180_df,under_180_df], axis=0).reset_index(drop=True)
merged_df['minmax_ratio'] = (merged_df['ratio'] - merged_df['ratio'].min()) / (merged_df['ratio'].max() - merged_df['ratio'].min())
merged_df['minmax_dep_delay'] = (merged_df['dep_delay'] - merged_df['dep_delay'].min()) / (merged_df['dep_delay'].max() - merged_df['dep_delay'].min())

over_180_df['minmax_ratio'] = (over_180_df['ratio'])/0.524012   #전체 ratio 중에서 가장 큰 값/작은값은 0
over_180_df['minmax_dep_delay'] = (over_180_df['dep_delay'])/743.5  #전체 dep_delay중에서 가장 큰 값/작은값은 0

under_180_df['minmax_ratio'] = (under_180_df['ratio'])/0.524012
under_180_df['minmax_dep_delay'] = (under_180_df['dep_delay'])/743.5

weight_total_score = ((over_180_df['minmax_ratio'] * 0.3 + over_180_df['minmax_dep_delay'] * 0.7) + 
 (under_180_df['minmax_ratio'] * 0.5 + under_180_df['minmax_dep_delay'] * 0.5))
weight_total_score_df = pd.concat([over_180_df['carrier'], weight_total_score], axis=1).sort_values(0,ascending=False)

weight_total_score_df.columns = ['carrier', 'weighted_score']
# HA, FL, WN, F9 의 순으로 높음

df.loc[(df['carrier'] == 'HA'),'distance'].mean()
df.loc[(df['carrier'] == 'FL'),'distance'].mean()
df.loc[(df['carrier'] == 'WN'),'distance'].mean()
df.loc[(df['carrier'] == 'F9'),'distance'].mean()

df['distance'].mean()



df['engine'].value_counts()
##!!!!!!!!!!!!!!!!표만들어!!!!!!!!!!!!!!!!!!!!!!##
df.loc[(df['carrier'] == 'HA'),'model'].value_counts()
df.loc[(df['carrier'] == 'HA'),'engines'].value_counts()
df.loc[(df['carrier'] == 'HA'),'engine'].value_counts()


df.loc[(df['carrier'] == 'FL'),'model'].value_counts()
df.loc[(df['carrier'] == 'FL'),'engines'].value_counts()
df.loc[(df['carrier'] == 'FL'),'engine'].value_counts()

df.loc[(df['carrier'] == 'WN'),'model'].value_counts()
df.loc[(df['carrier'] == 'WN'),'engines'].value_counts()
df.loc[(df['carrier'] == 'WN'),'engine'].value_counts()


df.loc[(df['carrier'] == 'F9'),'model'].value_counts()
df.loc[(df['carrier'] == 'F9'),'engines'].value_counts()
df.loc[(df['carrier'] == 'F9'),'engine'].value_counts()


#시각화 
df.loc[(df['carrier'] == 'HA') & (df['dep_delay']>0)]['month'].value_counts()
df.loc[(df['carrier'] == 'FL') & (df['dep_delay']>0)]['month'].value_counts()
df.loc[(df['carrier'] == 'WN') & (df['dep_delay']>0)]['month'].value_counts()
df.loc[(df['carrier'] == 'F9') & (df['dep_delay']>0)]['month'].value_counts()



df.loc[(df['carrier'] == 'HA') & (df['dep_delay']>0)]['day_of_week'].value_counts()
df.loc[(df['carrier'] == 'FL') & (df['dep_delay']>0)]['day_of_week'].value_counts()
df.loc[(df['carrier'] == 'WN') & (df['dep_delay']>0)]['day_of_week'].value_counts()
df.loc[(df['carrier'] == 'F9') & (df['dep_delay']>0)]['day_of_week'].value_counts()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='distance', y='dep_delay', alpha=0.5)
plt.title("비행 거리 vs. 출발 지연 시간")
plt.xlabel("비행 거리 (마일)")
plt.ylabel("출발 지연 시간 (분)")
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
top_4_carriers = ['HA', 'FL', 'WN', 'F9']
df_top_4 = df[df['carrier'].isin(top_4_carriers)]  # 상위 4개 항공사 데이터만 필터링

sns.scatterplot(data=df_top_4, x='distance', y='dep_delay', hue='carrier', alpha=0.5, palette='tab10')

plt.title("비행 거리 vs. 출발 지연 시간 (항공사별 비교)")
plt.xlabel("비행 거리 (마일)")
plt.ylabel("출발 지연 시간 (분)")
plt.legend(title="항공사")
plt.grid()
plt.show()





# 기존 출발 지연 시간 평균을 기준으로 한 순위
weight_total_score_df['rank'] = weight_total_score_df['weighted_score'].rank(ascending=False, method='dense')
# 항공사별 평균 지연 시간 계산
carrier_avg_delay = df.groupby('carrier')['dep_delay'].mean().reset_index()
carrier_avg_delay.columns = ['carrier', 'avg_dep_delay']
carrier_avg_delay['rank'] = carrier_avg_delay['avg_dep_delay'].rank(ascending=False, method='dense')

# 두 개의 순위를 비교하는 데이터프레임 생성
ranking_comparison = pd.merge(carrier_avg_delay[['carrier', 'rank']], 
                            weight_total_score_df[['carrier', 'rank']], 
                            on='carrier', 
                            suffixes=('_avg', '_weighted'))

# 그래프 시각화
plt.figure(figsize=(12, 6))
plt.bar(ranking_comparison['carrier'], ranking_comparison['rank_avg'], color='green', label='Avg Delay Rank')
plt.gca().invert_yaxis()  # Y축을 반전하여 아래에서 시작하도록 설정



plt.plot(ranking_comparison['carrier'], ranking_comparison['rank_weighted'], marker='s', linestyle='--', label='Weighted Score Rank')

plt.xlabel("Carrier")
plt.ylabel("Rank")
plt.title("Comparison of Rankings: Average Delay vs Weighted Score")
plt.xticks(rotation=90)
plt.legend()
plt.gca().invert_yaxis()  # 순위는 낮을수록 좋으므로 뒤집기
plt.grid(True)

plt.show()














# A330으로 시작하는 모든 모델을 사용하는 항공사들의 지연 시간 비교

# 'A330'으로 시작하는 모델을 사용하는 데이터 필터링
a330_models_df = df[df['model'].str.startswith('737', na=False)]

# 항공사별 평균 지연 시간 계산
a330_delays = a330_models_df.groupby('carrier')['dep_delay'].mean().reset_index()

# 시각화
plt.figure(figsize=(10, 6))
sns.barplot(data=a330_delays, x='carrier', y='dep_delay', palette='viridis')
plt.xlabel("Carrier")
plt.ylabel("Average Departure Delay (minutes)")
plt.title("Average Departure Delay for Airlines Using A330 Series Models")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



# Delta Airlines(DL) 항공사 데이터 필터링
dl_flights = df[df['carrier'] == 'WN']

# 기종별 평균 출발 지연 시간 계산
dl_avg_delay = dl_flights.groupby('model')['dep_delay'].mean().reset_index()

# 기종별 지연 횟수 (지연 시간이 0 이상인 경우)
dl_delay_count = dl_flights[dl_flights['dep_delay'] > 0].groupby('model').size().reset_index(name='delay_count')

# 두 데이터를 병합
dl_delay_stats = pd.merge(dl_avg_delay, dl_delay_count, on='model')

# 기종별 평균 지연 시간과 지연 빈도 비교
fig, ax1 = plt.subplots(figsize=(12, 6))

# 첫 번째 y축 (평균 지연 시간)
sns.barplot(data=dl_delay_stats, x='model', y='dep_delay', ax=ax1, color='skyblue', label='Avg Delay')
ax1.set_ylabel('Average Departure Delay (minutes)')
ax1.set_xlabel('Aircraft Model')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.legend(loc='upper left')

# 두 번째 y축 (지연 빈도)
ax2 = ax1.twinx()
sns.lineplot(data=dl_delay_stats, x='model', y='delay_count', ax=ax2, color='red', marker='o', label='Delay Count')
ax2.set_ylabel('Delay Count')
ax2.legend(loc='upper right')

plt.title('DL Airlines: Aircraft Model vs. Delay Statistics')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()








# 항공사별 지연 빈도 계산 (출발 지연이 0보다 큰 경우)
delay_counts = df[df['dep_delay'] > 0]['carrier'].value_counts().reset_index()
delay_counts.columns = ['carrier', 'delay_count']

# 전체 항공편 대비 지연 비율 계산
total_flights = df['carrier'].value_counts().reset_index()
total_flights.columns = ['carrier', 'total_flights']
delay_ratio = pd.merge(delay_counts, total_flights, on='carrier')
delay_ratio['delay_percentage'] = (delay_ratio['delay_count'] / delay_ratio['total_flights']) * 100

# 시각화
plt.figure(figsize=(12, 6))
sns.barplot(data=delay_ratio, x='carrier', y='delay_percentage', palette='viridis')
plt.xlabel("Carrier")
plt.ylabel("Delay Percentage (%)")
plt.title("Delay Frequency by Airline")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 항공사별 평균 지연 시간 계산
carrier_avg_delay = df.groupby('carrier')['dep_delay'].mean().reset_index()
carrier_avg_delay.columns = ['carrier', 'avg_dep_delay']

# 시각화
plt.figure(figsize=(12, 6))
sns.barplot(data=carrier_avg_delay, x='carrier', y='avg_dep_delay', palette='viridis')
plt.xlabel("Carrier")
plt.ylabel("Delay Percentage (%)")
plt.title("Delay Frequency by Airline")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()






import matplotlib.pyplot as plt
import seaborn as sns

# 항공사별 지연 횟수 (dep_delay > 0인 경우)
carrier_delay_count = df[df['dep_delay'] > 0].groupby('carrier').size().reset_index(name='delay_count')

# 항공사별 전체 운항 횟수
carrier_total_count = df.groupby('carrier').size().reset_index(name='total_count')

# 지연비율 계산: (지연 횟수) / (전체 운항 횟수)
carrier_delay_ratio = carrier_delay_count.merge(carrier_total_count, on='carrier')
carrier_delay_ratio['delay_ratio'] = carrier_delay_ratio['delay_count'] / carrier_delay_ratio['total_count']

# 정렬 (지연비율 높은 순)
carrier_delay_ratio = carrier_delay_ratio.sort_values(by='delay_ratio', ascending=False)

# 그래프 시각화
plt.figure(figsize=(12, 6))

# 첫 번째 y축 (지연 비율)
sns.barplot(data=carrier_delay_ratio, x='carrier', y='delay_ratio', color='blue', alpha=0.6)
plt.ylabel('Delay Ratio')
plt.xlabel('Carrier')
plt.title('Airlines: Delay Ratio by Carrier')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()




import matplotlib.pyplot as plt
import pandas as pd

# 기존 출발 지연 시간 평균을 기준으로 한 순위
weight_total_score_df['rank'] = weight_total_score_df['weighted_score'].rank(ascending=False, method='dense')

# 항공사별 평균 지연 시간 계산
carrier_avg_delay = df.groupby('carrier')['dep_delay'].mean().reset_index()
carrier_avg_delay.columns = ['carrier', 'avg_dep_delay']
carrier_avg_delay['rank'] = carrier_avg_delay['avg_dep_delay'].rank(ascending=False, method='dense')

# 두 개의 순위를 비교하는 데이터프레임 생성
ranking_comparison = pd.merge(carrier_avg_delay[['carrier', 'rank']], 
                              weight_total_score_df[['carrier', 'rank']], 
                              on='carrier', 
                              suffixes=('_avg', '_weighted'))

# 순위 반전: 1이 가장 위로 오도록 조정
max_rank = ranking_comparison['rank_avg'].max()
ranking_comparison['rank_avg_reversed'] = max_rank - ranking_comparison['rank_avg'] + 1

# 그래프 시각화
plt.figure(figsize=(12, 6))

# 막대그래프: 기존 데이터 그대로 사용 (1위가 위로 가도록 반전 적용)
plt.bar(ranking_comparison['carrier'], ranking_comparison['rank_avg_reversed'], 
        color='green', label='Avg Delay Rank')

# 점선 그래프 (기존 값 그대로 유지)
plt.plot(ranking_comparison['carrier'], ranking_comparison['rank_weighted'], 
         marker='s', linestyle='--', label='Weighted Score Rank')

plt.xlabel("Carrier")
plt.ylabel("Rank")
plt.title("Comparison of Rankings: Average Delay vs Weighted Score")
plt.legend()

# Y축 눈금 값을 1이 위로 오도록 설정
plt.yticks(ticks=ranking_comparison['rank_avg_reversed'], 
           labels=ranking_comparison['rank_avg'])

plt.show()








import matplotlib.pyplot as plt
import pandas as pd

# 기존 출발 지연 시간 평균을 기준으로 한 순위
weight_total_score_df['rank'] = weight_total_score_df['weighted_score'].rank(ascending=False, method='dense')

# 항공사별 평균 지연 시간 계산
carrier_avg_delay = df.groupby('carrier')['dep_delay'].mean().reset_index()
carrier_avg_delay.columns = ['carrier', 'avg_dep_delay']
carrier_avg_delay['rank'] = carrier_avg_delay['avg_dep_delay'].rank(ascending=False, method='dense')

# 두 개의 순위를 비교하는 데이터프레임 생성
ranking_comparison = pd.merge(carrier_avg_delay[['carrier', 'rank']], 
                              weight_total_score_df[['carrier', 'rank']], 
                              on='carrier', 
                              suffixes=('_avg', '_weighted'))

# 순위 반전: 1이 가장 위로 오도록 조정
max_rank_avg = ranking_comparison['rank_avg'].max()
max_rank_weighted = ranking_comparison['rank_weighted'].max()

ranking_comparison['rank_avg_reversed'] = max_rank_avg - ranking_comparison['rank_avg'] + 1
ranking_comparison['rank_weighted_reversed'] = max_rank_weighted - ranking_comparison['rank_weighted'] + 1

# 상위 4개 항공사 선정
top_n = 4
top_avg_carriers = ranking_comparison.nsmallest(top_n, 'rank_avg')['carrier'].tolist()
top_weighted_carriers = ranking_comparison.nsmallest(top_n, 'rank_weighted')['carrier'].tolist()

plt.figure(figsize=(12, 6))

# 막대그래프: 상위 4개 항공사는 빨간색, 나머지는 초록색
colors = ['gold' if carrier in top_avg_carriers else 'gray' for carrier in ranking_comparison['carrier']]
plt.bar(ranking_comparison['carrier'], ranking_comparison['rank_avg_reversed'], 
        color=colors, label='Avg Delay Rank')

# 점선 그래프: 기존 그래프 형태 유지, Top 4는 빨간색으로 하이라이트
plt.plot(ranking_comparison['carrier'], ranking_comparison['rank_weighted_reversed'], 
         marker='s', linestyle='--', color='royalblue', label='Weighted Score Rank')

# 개별 점 하이라이트 (Top 4 항공사는 빨간색 점선)
for i, carrier in enumerate(ranking_comparison['carrier']):
    if carrier in top_weighted_carriers:
        plt.plot(carrier, ranking_comparison.loc[i, 'rank_weighted_reversed'], 
                 marker='s', linestyle='--', color='red')

plt.xlabel("Carrier")
plt.ylabel("Rank")
plt.title("Comparison of Rankings: Average Delay vs Weighted Score (Top 4 Highlighted)")
plt.legend()


# Y축 눈금 값을 1이 위로 오도록 설정
plt.yticks(ticks=ranking_comparison['rank_avg_reversed'], 
           labels=ranking_comparison['rank_avg'])

plt.show()
