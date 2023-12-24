import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

# Reload the model from the saved file
loaded_model = load_model('best_model_mnist.keras')

data = pd.read_csv('mnist_test.csv')
data = data.dropna()

features = data.drop(columns=['label'], axis =1)
features = features/255  
targets = data['label']

features = np.array(features)
features = features.reshape((features.shape[0], -1))

targets = to_categorical(targets, 10)

y_pred = loaded_model.predict(features)
y_predict = np.argmax(y_pred, axis=1)
accuracy = np.mean(y_predict == np.argmax(targets, axis =1))

print("Accuracy for Model:", accuracy)