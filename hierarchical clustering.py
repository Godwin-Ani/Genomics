import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


df = pd.read_csv('hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv')
data = df.iloc[:,[3,4]].values
#dend = sch.dendrogram(sch.linkage(data, method = 'ward'))
#plt.show()
model = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
model.fit_predict(data)
plt.scatter(data[:,0],data[:,1], c=model.labels_, cmap='rainbow')
plt.show()