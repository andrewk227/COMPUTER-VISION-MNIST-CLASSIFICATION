import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

data = pd.read_csv('mnist_train.csv')

print(data.describe())

data = data.dropna()

# classes 0 - 9
# 784 (28x28)feature 1 target

features = data.drop(columns=['label'], axis =1)
#print(features)

features = features/255  # Normalization
targets = data['label']

# Reshape images to (28, 28, 1) for compatibility with convolutional layers
# train_images = features.reshape((features.shape[0], 28, 28, 1))


#splitin the data into train an validation sets
x_train , x_validation , y_train , y_validation = train_test_split(features, targets, test_size=0.2, random_state=42)

KNNclassifier = KNeighborsClassifier(n_neighbors=4)
KNNclassifier.fit(x_train, y_train)
y_pred = KNNclassifier.predict(y_train)

print(accuracy_score(y_train, y_pred))
