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

data = pd.read_csv('mnist_train.csv')
print(data.describe())
data = data.dropna()

# classes 0 - 9
# 784 (28x28)feature 1 target

features = data.drop(columns=['label'], axis =1)

# Normalization the data by dividing it by 255
features = features/255  
targets = data['label']

# Resize images to dimensions of 28 by 28
images = []
for row in features.values:
    temp = row.reshape(28, 28).tolist()
    images.append(temp)
    

# Visualize some images to verify the correctness of the reshaping process
num_images_to_visualize = 5
plt.figure(figsize=(10, 4))
for i in range(num_images_to_visualize):
    plt.subplot(2, num_images_to_visualize, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f'resized {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

#spliting the data into train and validation sets
x_train , x_validation , y_train , y_validation = train_test_split(features, targets, test_size=0.2, random_state=42)

x_train = np.array(x_train)
x_validation = np.array(x_validation)

x_train = x_train.reshape((x_train.shape[0], -1))
x_validation = x_validation.reshape((x_validation.shape[0], -1))

y_train = np.array(y_train)
y_validation = np.array(y_validation)

y_train = to_categorical(y_train, 10)
y_validation = to_categorical(y_validation, 10)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Intial Experiment: 
# KNN using grid search technique
KNN = KNeighborsClassifier()

# Create the parameter grid based on the results of random search
grid = {'n_neighbors': [2,4,6,8] }
GridSearch = GridSearchCV(KNN, grid, cv=2, scoring ='accuracy')
GridSearch.fit(x_train, y_train)
print ('Best Param', GridSearch.best_params_)

# Get the best model
best_knn_model = GridSearch.best_estimator_

# Evaluate the best model on the test set
accuracy = best_knn_model.score(x_train, y_train)
print("Test Accuracy:", accuracy)
#-----------------------------------------------------------------------------------------------------------------------------------------------
# Subsequent Experiment :
# Build the first model
model_1 = Sequential()
model_1.add(Dense(128, activation='relu', input_shape=(784,)))
model_1.add(Dense(64, activation='relu'))
model_1.add(Dense(10, activation='softmax'))
optimizer_1 = Adam(lr=0.001)
model_1.compile(optimizer=optimizer_1, loss='categorical_crossentropy', metrics=['accuracy'])
model_1.fit(x_train, y_train, epochs=10, batch_size=64,validation_data=(x_validation, y_validation))

# Build the second model
model_2 = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
optimizer_2 = Adam(lr=0.01)
model_2.compile(optimizer=optimizer_2, loss='categorical_crossentropy', metrics=['accuracy'])
model_2.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_validation, y_validation))


# Evaluate models on the test set
y_pred_1 = model_1.predict(x_validation)
y_pred_2 = model_2.predict(x_validation)

y_predict_1 = np.argmax(y_pred_1, axis=1)
y_predict_2 = np.argmax(y_pred_2, axis=1)

accuracy_1 = np.mean(y_predict_1 == np.argmax(y_validation, axis=1))
accuracy_2 = np.mean(y_predict_2 == np.argmax(y_validation, axis=1))

print("Accuracy for Model 1:", accuracy_1)
print("Accuracy for Model 2:", accuracy_2)

# Save the best model
best_model = model_1 if accuracy_1 > accuracy_2 else model_2
best_model.save('best_model_mnist.keras')

# Get the confusion matrix of the best model
conf_matrix_test = confusion_matrix(y_validation.argmax(axis=1), y_predict_1)
print("Confusion Matrix on Testing Set for model 1 :\n", conf_matrix_test)

conf_matrix_test = confusion_matrix(y_validation.argmax(axis=1), y_predict_2)
print("Confusion Matrix on Testing Set for model 2 :\n", conf_matrix_test)


