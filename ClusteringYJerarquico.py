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

kmeans = KMeans(n_clusters=3, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
y_kmeans = kmeans.fit_predict(X)
arreglo = kmeans.predict(X)


clusters = {}
n=0
for item in arreglo:
    if item in clusters:
        clusters[item].append(df.to_numpy()[n])
        n+=1
    else:
        clusters[item] = [df.to_numpy()[n]]
        n+=1
        
        
f = open("analisis.txt","w")
for item in clusters:
    print ("Cluster ", item)
    f.write("Cluster"+str(item)+"\n")
    for i in clusters[item]:
        print (i[0])
        f.write(str(i[0])+"\n")
f.close()
print(n)


print(df.describe().transpose())
