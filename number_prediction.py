import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#read csv file
mnist_train=pd.read_csv('mnist_train.csv', header=None)
mnist_test=pd.read_csv('mnist_test.csv', header=None)

cols=['label']


for i in range(784):
    cols.append('px_'+str(i+1))

mnist_test.columns=cols
mnist_train.columns=cols

#image = 28*28 pixels
image_size=28

#take values from "label'
train_label=mnist_train['label'].values
test_label=mnist_test['label'].values

train_image=mnist_train.values[:,1:]
test_image=mnist_test.values[:,1:]


train_image=train_image.reshape(60000,28,28)

test_image=test_image.reshape(10000,28,28)


knn_classifier=KNeighborsClassifier(n_jobs=-1)
knn_classifier=knn_classifier.fit(train_image.reshape(60000,784), train_label)
image_id=740
prediction=knn_classifier.predict(test_image[image_id].reshape(1,784))
print(prediction)
all_predictions=knn_classifier.predict(test_image.reshape(10000,784))
print(accuracy_score(test_label,all_predictions)*100)
