import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


data = pd.read_csv('mnist_train.csv')
# print(data.describe())

data = data.dropna()

# classes 0 - 9
# 784 (28x28)feature 1 target

features = data.drop(columns=['label'], axis =1)
#print(features)

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
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Intial Experiment ----> KNN using grid search technique
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
