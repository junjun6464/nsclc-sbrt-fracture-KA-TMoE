import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Multiply, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import backend as K

class AUCStopping3(Callback):
    def __init__(self, target_auc=0.8, training_data=None, min_epochs=3):
        super().__init__()
        self.target_auc = target_auc
        self.training_data = training_data
        self.min_epochs = min_epochs
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        X_train, y_train = self.training_data
        y_pred = self.model.predict(X_train, verbose=0)
        
        overall_auc = roc_auc_score(y_train, y_pred)
        print(f"Epoch {epoch + 1} - Overall Training AUC: {overall_auc:.4f}")
        if overall_auc >= self.target_auc and self.epoch_count >= self.min_epochs:
            print(f"\nStopping training as Overall Training AUC has reached {overall_auc:.2f} at epoch {epoch + 1}")
            self.model.stop_training = True

class TimeMoEWithKAN:
    def __init__(self, num_radiomics, knowledge_dim):
        self.num_radiomics = num_radiomics
        self.knowledge_dim = knowledge_dim
        self.model = self._build_model()

    def _build_model(self):
        input_1 = Input(shape=(self.num_radiomics,), name='time_1_input')
        input_2 = Input(shape=(self.num_radiomics,), name='time_2_input')
        input_3 = Input(shape=(self.num_radiomics,), name='time_3_input')

        knowledge_input_1 = Input(shape=(self.knowledge_dim,), name='knowledge_input_1')
        knowledge_input_2 = Input(shape=(self.knowledge_dim,), name='knowledge_input_2')
        knowledge_input_3 = Input(shape=(self.knowledge_dim,), name='knowledge_input_3')

        combined_input = Concatenate()([input_1, input_2, input_3, knowledge_input_1, knowledge_input_2, knowledge_input_3])
        gate_layer = Dense(64)(combined_input)
        gate_layer = BatchNormalization()(gate_layer)
        gate_layer = tf.keras.layers.ReLU()(gate_layer)
        gate_layer = Dropout(0.7)(gate_layer)
        gate_output = Dense(3, activation='softmax', name='gate_output')(gate_layer)

        expert_1 = Dense(64, activation='relu')(input_1)
        expert_2 = Dense(64, activation='relu')(input_2)
        expert_3 = Dense(64, activation='relu')(input_3)

        weighted_expert_1 = Multiply(name='weighted_expert_1')([tf.expand_dims(gate_output[:, 0], axis=-1), expert_1])
        weighted_expert_2 = Multiply(name='weighted_expert_2')([tf.expand_dims(gate_output[:, 1], axis=-1), expert_2])
        weighted_expert_3 = Multiply(name='weighted_expert_3')([tf.expand_dims(gate_output[:, 2], axis=-1), expert_3])

        final_expert_output = Concatenate(name='final_expert_concat')([weighted_expert_1, weighted_expert_2, weighted_expert_3])
        final_layer = Dense(64, activation='relu', name='final_dense')(final_expert_output)
        final_layer = Dropout(0.4)(final_layer)
        output = Dense(1, activation='sigmoid', name='output')(final_layer)

        model = Model(inputs=[input_1, input_2, input_3, knowledge_input_1, knowledge_input_2, knowledge_input_3], outputs=[output])
        optimizer = Adam(learning_rate=1e-5, clipvalue=1)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X1, y1, epochs=50, batch_size=32, class_weight=None):
        auc_stopping = AUCStopping3(target_auc=0.81, training_data=(X1, y1))
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weight, callbacks=[auc_stopping])
        return history

    def evaluate(self, X_test, y_test):
        val_loss, val_accuracy = self.model.evaluate(X_test, y_test)
        print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
        return val_loss, val_accuracy

