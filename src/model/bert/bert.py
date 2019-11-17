from util import get_dataset
from sklearn.metrics import classification_report 

import numpy as np  
from keras.layers import Dense, Input, concatenate
from keras.models import Model
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

def build_model(bert_dim=768, profile_dim=32):
    '''
        bert network
    '''
    bert_input = Input(shape=(bert_dim,))
    bert_output = Dense(256, activation='relu')(bert_input)
    bert_output = Dense(256, activation='relu')(bert_output)
    bert_output = Dense(256, activation='relu')(bert_output)
    bert_output = Dense(32, activation='relu')(bert_output)

    '''
        input for profile network
    '''
    profile_input = Input(shape=(profile_dim,))

    '''
        model for combined features
    '''
    x = concatenate([profile_input, bert_output])
    output = Dense(32, activation='relu')(x)
    output = Dense(16, activation='relu')(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[profile_input, bert_input], outputs=[output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', f1_m])
    return model
    

if __name__ == "__main__":
    cross_val = None

    for i in range(5):
        '''
            get data with bert embeddings
        '''
        train_x, train_y, test_x, test_y = get_dataset('bert')

        '''
            build neural network model
        '''
        model = build_model()
        model.summary()

        train_split = np.hsplit(train_x, np.array([32, 800]))[:2]
        test_split = np.hsplit(test_x, np.array([32, 800]))[:2]
        model.fit(x=train_split, y=train_y, batch_size=32, shuffle=True, epochs=100)

        if cross_val == None:
            cross_val = model.evaluate(test_split, test_y)
        else:
            cross_val += model.evaluate(test_split, test_y)
    
    print([metric/5 for metric in cross_val])
    model.save('bert_sent_parallel_cross_val.h5')
