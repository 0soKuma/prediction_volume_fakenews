




def minmaxavg(df):

  stats = df.groupby('dates')['NumArticles'].agg(['mean', 'max', 'min','count'])
  stats.columns = ['mean_NumArticles', 'max_NumArticles', 'min_NumArticles','NumberEvents']


  stats2 = df.groupby('dates')['GoldsteinScale'].agg(['mean', 'max', 'min',])
  stats2.columns = ['mean_GoldsteinScale', 'max_GoldsteinScale', 'min_GoldsteinScale',]

  stats = stats2.merge(stats, left_on = 'dates', right_index=True, how = 'left')
  
  stats3 = df.groupby('dates')['AvgTone'].agg(['mean', 'max', 'min'])
  stats3.columns = ['mean_AvgTone', 'max_AvgTone', 'min_AvgTone']
  
  stats = stats3.merge(stats, left_on = 'dates', right_index=True, how = 'left')
  

 
  stats4 = df.groupby('dates')['IsRootEvent'].agg(['mean', 'max', 'min','count'])
  stats4.columns = ['mean_IsRootEvent', 'max_IsRootEvent', 'min_IsRootEvent',"count_IsRootEvent"]
  stats = stats4.merge(stats, left_on = 'dates', right_index=True, how = 'left')
  
  stats4 = df.groupby('dates')['positive'].agg(['mean', 'max', 'min','count'])
  stats4.columns = ['mean_positive', 'max_positive', 'min_positive',"count_positive"]
  stats = stats4.merge(stats, left_on = 'dates', right_index=True, how = 'left')

  stats5 = df.groupby('dates')['negative'].agg(['mean', 'max', 'min','count'])
  stats5.columns = ['mean_negative', 'max_negative', 'min_negative',"count_negative"]
  stats = stats5.merge(stats, left_on = 'dates', right_index=True, how = 'left')

  stats6 = df.groupby('dates')['neutral'].agg(['mean', 'max', 'min','count'])
  stats6.columns = ['mean_neutral', 'max_neutral', 'min_neutral',"count_neutral"]
  stats = stats6.merge(stats, left_on = 'dates', right_index=True, how = 'left')

  lone_stats = stats
  gold = lone_stats['mean_NumArticles'].tolist()
  tone = lone_stats['mean_AvgTone'].tolist()
  
  res = []
  for i,j in zip(gold,tone):
    try:
      result = i / j
    except ZeroDivisionError:
      result = 0
   
    res.append(result)

  lone_stats = lone_stats.reset_index(level=0)
  lone_stats['articles_ratio_tone'] = res
 
  list_dates = lone_stats['dates'].tolist()
  last_week = []
  for day in list_dates:
    last_week.append(len(lone_stats[lone_stats['dates'] > day - pd.to_timedelta("7day")]))
  
  lone_stats['last_week'] = last_week
  last_week = []
  for day in list_dates:
    last_week.append(len(lone_stats[lone_stats['dates'] > day - pd.to_timedelta("14day")]))

  lone_stats['last_2weeks'] = last_week

  last_week = []
  for day in list_dates:
    last_week.append(len(lone_stats[lone_stats['dates'] > day - pd.to_timedelta("30day")]))

  lone_stats['last_month'] = last_week

 

  
  return stats,lone_stats


def count_sentiment(df):
  counting_df = pd.DataFrame()
  list_dates = df['dates'].tolist()
  topic_list = set(df['feelings'].tolist())
  
  listt = []
  point = 0
  for i in range(3):
    listt.append([])


 
  for day in list_dates:
    daily = df.loc[df['dates'] == day]
   
    for i in [0,1,2]:
   
      partial = daily.loc[daily['feelings'] == i]
    
      listt[i].append(len(partial))
    point += 1

  for i  in range(len(listt)):
    counting_df["feelings"+str(i)] = listt[i]

  counting_df["dates"] = list_dates
  return counting_df

def get_topics(df):

  cats = ['Social',"Política",'Migración / Racismo','Salud','Otros']
  cat = df['categories'].tolist()
  print(cat)
  per = []
  topics = []
  for dic in cat:

    res = list(eval(dic))
    per.append(res)

  list_topics = []
  for a in per:
    try:

      list_topics.append(a[0]['name'])
    except:
      list_topics.append("")
  return list_topics

def process_list(ls):
  final = []
  for l in ls:
    if 'Pol' in l:
      final.append("Politica")
    elif 'Salud' in l:
      final.append("Salud")
    elif 'Migración' in l:
       final.append("Migracion")
    elif 'Social' in l:
       final.append("Social")
    else:
      final.append("otros")
  return final


def group_by_topic(df):
  dfs = []
  topics = set(df['lista_topics'].tolist())
  for top in topics:
    dfs.append(df[df['lista_topics'] == top])
  return dfs

def group_titles_by_day(df):
  days = df['dates'].tolist()
  days = set(days)
  titles_matrix = []
  for day in days:

    partial = df[df['dates'] == day]
    titles_matrix.append(partial['titles'].tolist())
  return titles_matrix 


def group_titles_by_day(dates,df):
  days = set(dates)
  titles_matrix = []
  for day in days:
    partial = df[df['dates'] == day]
    titles_matrix.append(partial['titles'].tolist())
  return titles_matrix 
  
def calculate_rolling(df):
  keys = df.keys()
  days = [3,5,7,10,14,30,60,90]

  for key in keys:
      try:
        for day in days:
          key_df = key+"_"+str(day)
          df[key_df] = df[key].rolling(day).mean()
      except:
        continue
  return df

def count_fakes_in_day(df,df_maldita):
  dates = df['dates'].tolist()
  len_dates = []
  for date in dates:
  
    len_dates.append(len(df_maldita[df_maldita['just_date'] == str(date).split(" ")[0] ]))
  
  df['number_fake_news'] = len_dates

  return df



def create_dataset(df,days,flag):
  
  list_dates = df['dates'].tolist()
  quantity_fakes = df['number_fake_news'].tolist()
  reached_first = flag
  pointer = 0
  target_values = []
  dataset = []
  for date in list_dates:
    if reached_first == 1:
      d = date - timedelta(days=days)
      str_d = str(d).split(" ")[0]
      partial_df = df[(df['dates'] <= date) & (df['dates'] >= str_d)]
      get_val = df[(df['dates'] == date)]
      target_size = days + 1
      partial_df = partial_df.drop(columns=['dates','number_fake_news','just_dates'])
      if len(partial_df.values.tolist()) == target_size:
        dataset.append(partial_df.values.tolist())
        target_values.append(get_val['number_fake_news'].tolist()[0])
    else:
      if quantity_fakes[pointer] > 1000:
        reached_first = 1
      pointer += 1

  return dataset,target_values,list_dates





def prepare_data_consistency(dataset,target_values,scaler):
  
  train = dataset
  train_X = np.asarray(train).astype(np.float32)
  train_y = np.asarray(target_values).astype(np.float32)

  return train_X,train_y

def prepare_data(dataset,target_values,list_days,percentage):

  train = dataset
  train_X = np.asarray(train).astype(np.float32)
  train_y = np.asarray(target_values).astype(np.float32)
  diff_days = abs(len(train_X)-len(list_days))
  list_days = list_days[:len(list_days)-diff_days]
  X_test = train_X[int(len(train_X)*percentage):] 
  X_train= train_X[:int(len(train_X)*percentage)]
  y_test = train_y[int(len(train_y)*percentage):]
  y_train = train_y[:int(len(train_y)*percentage)]


  return X_test,X_train,y_test,y_train,scaler



def count_fakes_in_day(df,df_maldita):
  dates = df['dates'].tolist()
  len_dates = []
  for date in dates:
  
    len_dates.append(len(df_maldita[df_maldita['just_date'] == str(date).split(" ")[0] ]))
  
  
  df['number_fake_news'] = len_dates

  return df
  
def count_fakes_in_next_days(df,df_maldita,days):
  dates = df['dates'].tolist()
  len_dates = []
  for date in dates:    
    d = date + timedelta(days=days)

    str_d = str(d).split(" ")[0]
    original_date = str(date).split(" ")[0]

    testing_week = df_maldita[(df_maldita['just_date'] >= original_date) & (df_maldita['just_date'] < str_d)]

    len_dates.append(len(testing_week))
  
  df['number_fake_news'] = len_dates

  return df

def group_titles_by_day(dates,df):
  days = set(dates)
  
  titles_matrix = []
  for day in days:
    str_day = str(day).split(" ")[0]
    
    partial = df[df['dates'] == str_day]
    titles_matrix.append(partial['titles'].tolist())
  return titles_matrix 



def create_dataset_target_days(df,days):

  
  list_dates = df['dates'].tolist()
  quantity_fakes = df['number_fake_news'].tolist()

  reached_first = 1
  pointer = 0
  target_values = []
  dataset = []

  for date in list_dates:

    if reached_first == 1:
      

      d = date - timedelta(days=days)
      str_d = str(d).split(" ")[0]
      partial_df = df[(df['dates'] <= date) & (df['dates'] >= str_d)]
      get_val = df[(df['dates'] == date)]
      
      target_size = days + 1
      
      partial_df = partial_df.drop(columns=['dates','number_fake_news'])
      
      if len(partial_df.values.tolist()) == target_size:
        dataset.append(partial_df.values.tolist())
        target_values.append(get_val['number_fake_news'].tolist()[0])

    else:
      if quantity_fakes[pointer] > 1000:
        reached_first = 1
      pointer += 1


  return dataset,target_values,list_dates


training_gdelt = read_dataframe("/xxxx.csv")
training_maldita = read_dataframe("xxxxx.csv")
gtopL = get_topics(training_maldita)
gtopL = process_list(gtopL)
training_maldita['lista_topics'] = gtopL
training_gdelt = training_gdelt.drop_duplicates(subset=["titles"], keep=False)
training_gdelt = training_gdelt.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'GLOBALEVENTID',
      'FractionDate', 'Actor1Code',
       'Actor1CountryCode', 'Actor1KnownGroupCode', 'Actor1EthnicCode',
       'Actor1Religion1Code', 'Actor1Religion2Code', 'Actor1Type1Code',
       'Actor1Type2Code', 'Actor1Type3Code', 'Actor2Type1Code',
       'Actor2Type2Code', 'Actor2Type3Code', 
       'Actor1Geo_FeatureID', 'Actor2Geo_Type', 'Actor2Geo_FullName',
       'Actor2Geo_CountryCode', 'Actor2Geo_ADM1Code', 'Actor2Geo_Lat',
       'Actor2Geo_Long', 'Actor2Geo_FeatureID', 'ActionGeo_Type',
       'ActionGeo_FullName', 'ActionGeo_CountryCode', 'ActionGeo_ADM1Code',
       'ActionGeo_Lat', 'ActionGeo_Long', 'ActionGeo_FeatureID', 
       'SOURCEURL', 'Actor1Geo_ADM2Code', 'Actor2_ADM1Code', 'Actor3Geo_Type',
       'Actor3Geo_FullName', 'Randomplace'])

dates = training_gdelt['DATEADDED'].tolist()
new_dates = []
for date in dates:
  new_dates.append(str(date)[0:4]+"-"+str(date)[4:6]+"-"+str(date)[6:8])
training_gdelt['dates'] = new_dates
training_gdelt['dates'] = pd.to_datetime(training_gdelt['dates'],  format='%Y-%m-%d')
training_gdelt = training_gdelt.sort_values(by="dates")
get_titles = training_gdelt[training_gdelt['dates']>= "2019-09-23"]
training_gdelt = training_gdelt.sort_values(by="dates")
get_titles = get_titles[get_titles['dates']<= "2022-07-14"]
listdays = try_df['dates'].tolist()
texts_in_dataset = group_titles_by_day(listdays,training_gdelt)
texts_by_day = []
for textday in texts_in_dataset:
  sub_arra = []
  i = 0
  while i < 10:
    if len(textday) < 10:
      for x in range(len(textday)):
        sub_arra.append(textday[x].replace("_"," "))
        i += 1
      
      howmany = abs(len(sub_arra)-10)
      for y in range(howmany):
        sub_arra.append("")
    else:
      sub_arra.append(textday[i].replace("_"," "))
      i += 1
  
  texts_by_day.append(sub_arra)
     