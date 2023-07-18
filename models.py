import keras

def autoencoder(train_X,train_y):

    enc_inputs = keras.Input(shape=(train_X.shape[1], train_X.shape[2]), name='inputs')

    lstm = keras.layers.LSTM(units=64,return_sequences=True)(enc_inputs)
    lstm = keras.layers.LSTM(units=8,return_sequences=False)(lstm)
    gruencode = keras.layers.RepeatVector(train_X.shape[1], name='encoder_decoder')(lstm)
    lstm = keras.layers.LSTM(units=8,return_sequences=True)(gruencode)
    lstm = keras.layers.LSTM(units=64,return_sequences=False)(lstm)
    dense = keras.layers.Dense(units=32, activation='linear')(lstm)
    dense = keras.layers.Dropout(0.1)(dense)
    dense = keras.layers.Dense(units=16, activation='linear')(dense)
    dense = keras.layers.Dropout(0.1)(dense)
    dense = keras.layers.Dense(units=8, activation='linear')(dense)
    dense = keras.layers.Dropout(0.1)(dense)
    dense = keras.layers.Dense(units=len(train_y[0]), activation='linear')(dense)
    
    model =keras.Model(enc_inputs, dense, name='encoder')
    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse'])

    return model



def create_basic_model_encoder_attention(train_X,train_y):
    n_hidden = 128
    enc_inputs = keras.Input(shape=(train_X.shape[1], train_X.shape[2]), name='inputs')
    encoder_last_h1 = keras.layers.LSTM(
    n_hidden, dropout=0.2, recurrent_dropout=0.2, 
    return_sequences=True, return_state=False)(enc_inputs)
    x = Attention(units=64)(encoder_last_h1)
    mlp = keras.layers.Dense(64)(x)
    mlp = keras.layers.Dense(32)(mlp)
    mlp = keras.layers.Dense(16)(mlp)
    mlp = keras.layers.Dense(8)(mlp)
    mlp = keras.layers.Dense(len(train_y[0]))(mlp)
    model = keras.Model(enc_inputs, mlp, name='encoder')
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model
     





def create_model_text(X_train,vectorize_layersfinal,train_y):
    embedding_dim = 128 
    enc_inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]), name='inputs')
      
    

    texts_input1 =  keras.Input(shape=(1,), dtype=tf.string)
    texts_input2 =  keras.Input(shape=(1,), dtype=tf.string)
    texts_input3 =  keras.Input(shape=(1,), dtype=tf.string)
    texts_input4 =  keras.Input(shape=(1,), dtype=tf.string)
    texts_input5 =  keras.Input(shape=(1,), dtype=tf.string)
    texts_input6 =  keras.Input(shape=(1,), dtype=tf.string)
    texts_input7 =  keras.Input(shape=(1,), dtype=tf.string)
    texts_input8 =  keras.Input(shape=(1,), dtype=tf.string)
    texts_input9 =  keras.Input(shape=(1,), dtype=tf.string)
    texts_input10 =  keras.Input(shape=(1,), dtype=tf.string)

    vect1 = vectorize_layersfinal(texts_input1)
    vect2 = vectorize_layersfinal(texts_input2)
    vect3 = vectorize_layersfinal(texts_input3)
    vect4 = vectorize_layersfinal(texts_input4)
    vect5 = vectorize_layersfinal(texts_input5)
    vect6 = vectorize_layersfinal(texts_input6)
    vect7 = vectorize_layersfinal(texts_input7)
    vect8 = vectorize_layersfinal(texts_input8)
    vect9 = vectorize_layersfinal(texts_input9)
    vect10 = vectorize_layersfinal(texts_input10)


    test_emb = keras.layers.Embedding(10000, embedding_dim, name = 'embedding1')
    emb1 = test_emb(vect1)  
    emb2 = test_emb(vect2)  
    emb3 = test_emb(vect3)  
    emb4 = test_emb(vect4)  
    emb5 = test_emb(vect5)  
    emb6 = test_emb(vect6)  
    emb7 = test_emb(vect7)  
    emb8 = test_emb(vect8)  
    emb9 = test_emb(vect9)  
    emb10 = test_emb(vect10)  

    msg_out1 = keras.layers.LSTM(32, kernel_regularizer = keras.regularizers.L1L2(l1=0.001, l2=0.01),dropout=0.3,return_sequences=False)(emb1)    
    msg_out2 = keras.layers.LSTM(32, kernel_regularizer = keras.regularizers.L1L2(l1=0.001, l2=0.01),dropout=0.3,return_sequences=False)(emb2)    
    msg_out3 = keras.layers.LSTM(32, kernel_regularizer = keras.regularizers.L1L2(l1=0.001, l2=0.01),dropout=0.3,return_sequences=False)(emb3)    
    msg_out4 = keras.layers.LSTM(32, kernel_regularizer = keras.regularizers.L1L2(l1=0.001, l2=0.01),dropout=0.3,return_sequences=False)(emb4)    
    msg_out5 = keras.layers.LSTM(32, kernel_regularizer = keras.regularizers.L1L2(l1=0.001, l2=0.01),dropout=0.3,return_sequences=False)(emb5)   
    msg_out6 = keras.layers.LSTM(32, kernel_regularizer = keras.regularizers.L1L2(l1=0.001, l2=0.01),dropout=0.3,return_sequences=False)(emb6)    
    msg_out7 = keras.layers.LSTM(32, kernel_regularizer = keras.regularizers.L1L2(l1=0.001, l2=0.01),dropout=0.3,return_sequences=False)(emb7)    
    msg_out8 = keras.layers.LSTM(32, kernel_regularizer = keras.regularizers.L1L2(l1=0.001, l2=0.01),dropout=0.3,return_sequences=False)(emb8)    
    msg_out9 = keras.layers.LSTM(32, kernel_regularizer = keras.regularizers.L1L2(l1=0.001, l2=0.01),dropout=0.3,return_sequences=False)(emb9)    
    msg_out10 = keras.layers.LSTM(32, kernel_regularizer = keras.regularizers.L1L2(l1=0.001, l2=0.01),dropout=0.3,return_sequences=False)(emb10)     

    gruencode = keras.layers.LSTM(256 ,return_sequences=False)(enc_inputs)
    x = keras.layers.concatenate([msg_out1,msg_out2,msg_out3,msg_out4,msg_out5,msg_out6,msg_out7,msg_out8,msg_out9,msg_out10, gruencode]) 
    mlp = keras.layers.Dense(1024, activation='linear')(x)
    mlp = keras.layers.Dropout(0.5)(mlp)
    mlp = keras.layers.Dense(512, activation='linear')(mlp)
    mlp = keras.layers.Dropout(0.3)(mlp)
    mlp = keras.layers.Dense(256, activation='linear')(mlp)
    mlp = keras.layers.Dropout(0.2)(mlp)
    mlp = keras.layers.Dense(128, activation='linear')(mlp)
    mlp = keras.layers.Dropout(0.2)(mlp)
    mlp = keras.layers.Dense(64, activation='linear')(mlp)
    mlp = keras.layers.Dropout(0.1)(mlp)
    mlp = keras.layers.Dense(32, activation='linear')(mlp)
    mlp = keras.layers.Dropout(0.1)(mlp)
    mlp = keras.layers.Dense(16, activation='linear')(mlp)
    mlp = keras.layers.Dropout(0.1)(mlp)
    mlp = keras.layers.Dense(len(train_y[0]), activation='linear',kernel_constraint=keras.constraints.non_neg())(mlp)

    model = keras.Model([enc_inputs,texts_input1,texts_input2,texts_input3,texts_input4,texts_input5,texts_input6,texts_input7,texts_input8,texts_input9,texts_input10], mlp, name='encoder')
    model.compile(loss='mse', optimizer='adam')

    return model
