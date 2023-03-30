# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# load dataset ربط ملف قاعدة البيانات 
df = pd.read_csv('dataset.csv')

#print(df.info())
null_values = df.isnull().sum()

df['ammonia'] = df['ammonia'].apply(lambda x: float(x.split()[0].replace('#NUM!', 'NAN')))
df['is_safe'] = df['is_safe'].apply(lambda x: float(x.split()[0].replace('#NUM!', 'NAN')))

df['ammonia'] = df['ammonia'].astype(float)
df['is_safe'] = df['is_safe'].astype(float)

# Preprocessing
# Remove objects with missing values
df = df.dropna()

# split dataset into features and target variable
y = df['is_safe']
X = df.drop(['is_safe'], axis=1)

# normalize the dataset
sc = MinMaxScaler()
X = sc.fit_transform(X) 


# Split dataset into 75% Training and 25% Test
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, y,test_size=0.25, random_state=42)

# Classification using Logistic Regression
model1 = LogisticRegression(random_state= 42)

# Train the Logistic Regression model
model1.fit(Xtrain,Ytrain)

# Make Predictions
prediction1 = model1.predict(Xtest)

# calculate the performance of the Logistic Regression model
accuracy_lr = np.round(accuracy_score(prediction1,Ytest)*100, 2)
recall_lr = np.round(recall_score(prediction1,Ytest, average='weighted')*100, 2)


# print the performance of the Logistic Regression model
print('LR Accuracy= ' + str(accuracy_lr))
print('LR Recall= ' + str(recall_lr))


print('')
#-----------------------------------------------------------------#
# Classification using K-Nearest Neighbors (KNN)
model2 = KNeighborsClassifier(n_neighbors = 3)

# Train the KNN model
model2.fit(Xtrain,Ytrain)

# Make Predictions
prediction2 = model2.predict(Xtest)

# calculate the performance of the KNN model
accuracy_knn = np.round(accuracy_score(prediction2,Ytest)*100, 2)
recall_knn = np.round(recall_score(prediction2,Ytest, average='weighted')*100, 2)

# print the performance of the KNN model
print('KNN Accuracy= ' + str(accuracy_knn))
print('KNN Recall= ' + str(recall_knn))


# Clustering using k-means algorithm
print('\nClusteringg using K-means algorithm')
sse = {}
for k in range(2, 10):
    # Create k-means model
    kmeans = KMeans(n_clusters=k, max_iter=1000)
    # Train the model using the dataset
    kmeans.fit(X)   
    # Evaluate the model
    labels = kmeans.predict(X)
    # Calculate silhoutee score for each number of clusters
    score = silhouette_score(X, labels, metric='euclidean')
    print('K: '+str(k)+'  Silhouette Score: %.3f' % score)
    # Calculate SSE for each number of clusters    
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

# Plot SSE with the number of clusters
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()
