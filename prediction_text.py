from utils import *
from text_prepare_data *
from models import *
import pandas as pd
import numpy as np
import keras
import os
import statistics
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import movecolumn as mc
from sklearn.preprocessing import MinMaxScaler
import pickle
import os



def main():

    training_gdelt = pd.read_dataframe("xxxx.csv")
    training_maldita = pd.read_dataframe("xxxxx.csv")
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
    make_linear = []
    for group in texts_by_day:
        for phrase in group:
            make_linear.append(phrase)

    vectorize_layer_for_model = keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=64)

    vectorize_layer_for_model.adapt(make_linear)
    for percent in [0.6,0.9]:
        for j in [1,2,4,7]:
            days_to_predict = j
            epochs = 3000
            batch_size = 512
            texts_to_input = 1
            percentage_training = percent
            consistency = 0
            test_name = "xxx" +str(days_to_predict)+"_"+str(epochs)+"_"+str(percentage_training)
            try_df = count_fakes_in_next_days(to_train1,result,days_to_predict)
            try_df = mc.MoveToLast(try_df,'number_fake_news')
            try_df = try_df[try_df['dates']>= "2019-10-01"]
            try_df = try_df[try_df['dates']<= "2022-07-14"]
            dates_df = try_df['dates']
            try_df = try_df.drop(columns=["just_dates","dates"])


            train = try_df
            scalers={}
            for i in try_df.columns:
                
                scaler = MinMaxScaler(feature_range=(0,1))
                s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
                s_s=np.reshape(s_s,len(s_s))
                scalers['scaler_'+ i] = scaler
                train[i]=s_s

            train['dates'] = dates_df
            if consistency:
                
                testing_consistency = train[train['dates']>= "2022-02-01"]
                testing_consistency = testing_consistency[testing_consistency['dates']<= "2022-02-28"]
                dates_df = testing_consistency['dates']
                testing_consistency = testing_consistency.drop(columns=["dates"])
                
            for i in testing_consistency.columns:
                
                    scaler = scalers['scaler_'+ i]
                    s_s = scaler.fit_transform(testing_consistency[i].values.reshape(-1,1))
                    s_s=np.reshape(s_s,len(s_s))
                    testing_consistency[i]=s_s
            testing_consistency['dates'] = dates_df
            train = train[~train.dates.isin(testing_consistency.dates)]
            testing_consistency = mc.MoveToLast(testing_consistency,'number_fake_news')
            for i in [1,2,3,4,5,6,7,8,9,10]:
                err = []
                errrmse = []
                errmse = []
                errmae = []
                errr2= []
                days = i
                errrpear = []
                errors = []
                
            for i in range(5):
                lowest = 9999
                results_experiments = []
                
                try_df = mc.MoveToLast(try_df,'number_fake_news')
                listdays = try_df['dates'].tolist()
                texts_in_dataset = group_titles_by_day(listdays,training_gdelt)
                texts_by_day = []
                
                path = "xxxx"+test_name+"/"+test_name+str(days)+"/"
                if not os.path.exists("xxx"+test_name+"/"):
                    os.mkdir("xxx"+test_name+"/")
                if not os.path.exists("xxx"+test_name+"/"+test_name+str(days)+"/"):
                    os.mkdir("xxx"+test_name+"/"+test_name+str(days)+"/")

                for textday in texts_in_dataset:
                    sub_arra = []
                i = 0
                while i < 10:
                    if len(textday) < 10:
                        for x in range(len(textday)):
                            sub_arra += textday[x]
                            i += 1
                    
                    howmany = abs(len(sub_arra)-10)
                    for y in range(howmany):
                        sub_arra.append("")
                    else:
                        sub_arra.append(textday[i])
                    i += 1
                
                texts_by_day.append(sub_arra)
                target_days = i
                dataset,target_values,list_days,aa = create_dataset_text(try_df,target_days,1,texts_by_day)

                aa = aa[:len(aa)-target_days]

                dataset = np.array(dataset)
                target_values = np.array(target_values)
                X_test,X_train,y_test,y_train,scaler,test_texts,train_texts = prepare_data_with_texts(dataset,target_values,list_days,aa,percentage_training)
                train_text_new = np.array(train_texts)
                test_text_new = np.array(test_texts)


                train_1 = []
                train_2 = []
                train_3 = []
                train_4 = []
                train_5 = []
                train_6 = []
                train_7 = []
                train_8 = []
                train_9 = []
                train_10 = []

                test_1 = []
                test_2 = []
                test_3 = []
                test_4 = []
                test_5 = []
                test_6 = []
                test_7 = []
                test_8 = []
                test_9 = []
                test_10 = []

                for group in train_text_new:
                    train_1.append(group[0])
                    train_2.append(group[1])
                    train_3.append(group[2])
                    train_4.append(group[3])
                    train_5.append(group[4])
                    train_6.append(group[5])
                    train_7.append(group[6])
                    train_8.append(group[7])
                    train_9.append(group[8])
                    train_10.append(group[9])

                for group in test_text_new:
                    test_1.append(group[0])
                    test_2.append(group[1])
                    test_3.append(group[2])
                    test_4.append(group[3])
                    test_5.append(group[4])
                    test_6.append(group[5])
                    test_7.append(group[6])
                    test_8.append(group[7])
                    test_9.append(group[8])
                    test_10.append(group[9])

                    train_1 = np.array(train_1)
                    train_2 = np.array(train_2)
                    train_3 = np.array(train_3)
                    train_4 = np.array(train_4)
                    train_5 = np.array(train_5)
                    train_6 = np.array(train_6)
                    train_7 = np.array(train_7)
                    train_8 = np.array(train_8)
                    train_9 = np.array(train_9)
                    train_10 = np.array(train_10)


                    test_1 = np.array(test_1)
                    test_2 = np.array(test_2)
                    test_3 = np.array(test_3)
                    test_4 = np.array(test_4)
                    test_5 = np.array(test_5)
                    test_6 = np.array(test_6)
                    test_7 = np.array(test_7)
                    test_8 = np.array(test_8)
                    test_9 = np.array(test_9)
                    test_10 = np.array(test_10)

                early_stopping_monitor = EarlyStopping(
                    monitor='val_loss',
                    min_delta=0,
                    patience=1000,
                    verbose=0,
                    mode='auto',
                    baseline=None,
                    restore_best_weights=True
                )




                model1 = create_model_text(X_train,vectorize_layer_for_model)
                history = model1.fit([X_train,train_1,train_2,train_3,train_4,train_5,train_6,train_7,train_8,train_9,train_10], y_train, epochs=3000, batch_size=batch_size,callbacks=[early_stopping_monitor],validation_data=([X_test,test_1,test_2,test_3,test_4,test_5,test_6,test_7,test_8,test_9,test_10], y_test), verbose=0, shuffle=True)
                yhat = model1.predict([X_test,test_1,test_2,test_3,test_4,test_5,test_6,test_7,test_8,test_9,test_10])
                yhat = yhat.flatten()
                y_testw = y_test.flatten()
                y_testw = scalers['scaler_number_fake_news'].inverse_transform(y_test.reshape(-1, 1)).flatten()
                yhat = scalers['scaler_number_fake_news'].inverse_transform(yhat.reshape(-1, 1)).flatten()
                yhat[yhat < 0] = 0
                rmse = sqrt(mean_squared_error(y_testw,yhat))
                mse = mean_squared_error(y_testw,yhat)
                mae = np.mean(np.abs(y_testw - yhat))
                r2 = r2_score(y_testw, yhat)
                cor = median_abs_deviation(y_testw - yhat) 
                with open("xxxx"+test_name+"/"+test_name+str(days)+'/'+str(i)+"savingrest.pkl",'wb') as f:
                    pickle.dump(yhat, f)
                with open("xxxx"+test_name+"/"+test_name+str(days)+'/'+str(i)+"other.pkl",'wb') as f:
                    pickle.dump(y_testw, f) 
                errrmse.append(rmse)
                errmae.append(mae)
                errmse.append(mse)
                errrpear.append(cor)
                errr2.append(r2)

                np.savetxt("xxxx"+test_name+"/"+test_name+str(days)+"/"+test_name+"round"+str(i)+".csv", [rmse,mae,mse,cor,r2], delimiter=",")

            mediaremse = statistics.mean(errrmse)
            mediamae = statistics.mean(errmae)
            mediamse = statistics.mean(errmse)
            mediamr2 = statistics.mean(errr2)
            meadipear = statistics.mean(errrpear)
            errors.append([mediaremse,mediamae,mediamse,mediamr2,meadipear])
            arra = np.array(errors)
            np.savetxt("xxxx"+test_name+"/"+test_name+str(days)+"/"+test_name+"total.csv", arra, delimiter=",")
        

    
    
     
        

if __name__ == "__main__":
    main()
