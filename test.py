import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM,Conv1D,MaxPooling1D,Flatten,TimeDistributed #the two main layers of the model
from tensorflow.keras.optimizers import Adam#for the training of the model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from targets_plot_generator import generate_plot
# from preprocessing import preprocessing
twoexp_nodes_number_layer_1 = 7
twoexp_nodes_number_layer_2 = 10
twoexp_nodes_number_layer_3 = 7
twoexp_nodes_number_layer_4 = 6
twoexp_nodes_number_layer_5 = 0
df = pd.read_csv("TSLA_data.csv")
df.set_index('DCP', inplace=True) 
features = df[['DNCP', 'OPCP', 'HPCP', 'LPCP', 'CPCP', 'ACPCP', 'VTCP']]
target = df['MPN5P']

scaler_features = MinMaxScaler().fit(features)                 #scaling
scaler_target = MinMaxScaler().fit(target.values.reshape(-1, 1))
features_scaled = scaler_features.transform(features)
target_scaled = scaler_target.transform(target.values.reshape(-1, 1))
def building_data_sequences(data_X,data_Y, timesteps): #timesteps means how many days we consider for each block
    print("inside building sequence ")
    X=[]
    y_MPNxP = []
    for i in range(len(data_X)-timesteps+1):  #how it works: every timesteps (e.g. 10 days) a block is constituted and for each block data and true values are stored
        X.append(data_X[i:(i+timesteps),:])
        y_MPNxP.append(data_Y[i+timesteps-1])
    return np.array(X), np.array(y_MPNxP)
def custom_loss_function(attenuated_padding_value):
    print("inside the custom loss sequence")
    
    def padding_loss_function(y_true, y_pred):
        print("inside the padding loss sequence")
        attenuated_padding_value_tensor = tf.convert_to_tensor(attenuated_padding_value, dtype=tf.float32)
        batch_size = tf.shape(y_pred)[0]
        attenuated_padding_value_broadcasted = tf.broadcast_to(attenuated_padding_value_tensor[:batch_size], tf.shape(y_pred))
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.multiply(y_pred, attenuated_padding_value_broadcasted)
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(squared_difference, axis=-1)  #mse
    return padding_loss_function
def prediction_model_002(input_shape,optimizer,attenuated_padding_value):
    tf.keras.backend.clear_session()
    model = Sequential([
        LSTM(2**twoexp_nodes_number_layer_1,input_shape=input_shape,return_sequences=True),
        LSTM(2**twoexp_nodes_number_layer_2,return_sequences=True),
        LSTM(2**twoexp_nodes_number_layer_3),
        Dense(2**twoexp_nodes_number_layer_4),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss=custom_loss_function(attenuated_padding_value))
    return model
timesteps = 10  # Number of days to look back for each sequence
X, y = building_data_sequences(features_scaled, target_scaled, timesteps)
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
optimizer = Adam(learning_rate=0.001)
attenuated_padding_value = np.ones_like(y) 
input_shape = (X.shape[1], X.shape[2])
model = prediction_model_002(input_shape, optimizer, attenuated_padding_value)
split_index = int(0.8 * len(X))  # 80% train, 20% test
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
predictions = model.predict(X_test)
predictions = scaler_target.inverse_transform(predictions) 

dates = df.index[-len(predictions):]  
y_true = scaler_target.inverse_transform(y_val)
target_name = "Stock Price"

plot_results = generate_plot(predictions, y_true, dates, target_name)

print("Plot")
for plot_type, metrics in plot_results.items():
    if isinstance(metrics, dict):
        print(f"{plot_type}: {metrics}")
    else:
        print(f"{plot_type} saved at: {metrics}")
