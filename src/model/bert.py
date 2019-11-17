from util import get_dataset
from sklearn.metrics import classification_report 

from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def build_model(input_dim=768):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_dim,), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', f1_m])
    return model
    

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = get_dataset('bert')

    # print(train_x.shape, train_y.shape)

    '''
        build neural network model
    '''
    model = build_model(input_dim=train_x.shape[1])
    model.summary()

    model.fit(x=train_x, y=train_y, batch_size=32, validation_split=0.1, shuffle=True, epochs=100)

    # print(model.evaluate(test_x, test_y))