import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow

data = pd.read_csv('mnist_train.csv')

print(data.describe())

data = data.dropna()

# classes 0 - 9
# 784 (28x28)feature 1 target

features = data.drop(columns=['label'], axis =1)
print(features)

features = features/255  # Normalization
targets = data['label']

# reshaping the image before the split into 28 x 28
# train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))

x_train , x_validation , y_train , y_validation = train_test_split(features, targets, test_size=0.2, random_state=42)

