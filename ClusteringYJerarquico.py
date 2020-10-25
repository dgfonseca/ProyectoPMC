import pandas as pd
from csv import reader
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import numpy as np

df = pd.read_csv(r'C:\Users\david\Desktop\Clases Universidad\Semestre 7\PMC\ProyectoPMC\proyecto.csv', encoding = "ISO-8859-1")
print(df.head())
#Estadisticas
jd = df
cluster_included = df

#K means
X = df.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]].values
wcss=[]

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('Metodo del codo')
plt.xlabel('Numero de clusters')
plt.ylabel('Suma distancia centroide')
plt.show()

kmeans = KMeans(n_clusters=5, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
y_kmeans = kmeans.fit_predict(X)
arreglo = kmeans.predict(X)

clusters = {}
n=0
for item in arreglo:
    if item in clusters:
        clusters[item].append(df.to_numpy()[n])
    else:
        clusters[item] = [df.to_numpy()[n]]
        n+=1
for item in clusters:
    print ("Cluster ", item)
    for i in clusters[item]:
        print (i)

plt.scatter(X[y_kmeans==0, 19], X[y_kmeans==0, 20], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 19], X[y_kmeans==1, 20], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 19], X[y_kmeans==2, 20], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 19], X[y_kmeans==3, 20], s=100, c='black', label ='Cluster 4')
plt.scatter(X[y_kmeans==4, 19], X[y_kmeans==4, 20], s=100, c='yellow', label ='Cluster 5')
plt.scatter(X[y_kmeans==5, 19], X[y_kmeans==5, 20], s=100, c='orange', label ='Cluster 6')


plt.scatter(kmeans.cluster_centers_[:, 19], kmeans.cluster_centers_[:, 20], s=300, c='yellow', label = 'Centroids')
plt.title('Comparación puntaje global vs Proposito en la vivienda')
plt.xlabel('Puntaje Global')
plt.ylabel('Proposito universitario')
plt.show()

plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 20], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 20], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 20], s=100, c='green', label ='Cluster 3')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 20], s=300, c='yellow', label = 'Centroids')
plt.title('Comparación puntaje global vs Cantidad compañeros')
plt.xlabel('Puntaje Global')
plt.ylabel('Cantida compañeros')
plt.show()



print(df.describe().transpose())
