import pandas as pd
from csv import reader
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv(r'C:\Users\david\Desktop\Clases Universidad\Semestre 7\BI\laboratorio_3a_Mall_Customers_Ultima_Version.csv')
#Estadisticas
jd = df
contadorFem = 0
contadorMasc = 0
with open(r'C:\Users\david\Desktop\Clases Universidad\Semestre 7\BI\laboratorio_3a_Mall_Customers_Ultima_Version.csv') as read_obj:
    csv_reader = reader(read_obj)
    contadorFem = 0
    contadorMasc = 0
    for row in csv_reader:
        if str(row[1])==str("Male"):
            contadorMasc+=1
        else:
            contadorFem+=1

#K means
X = df.iloc[:,[3,4]].values
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



plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='magenta', label ='Cluster 4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='black', label ='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
plt.title('Cluster de clientes')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100')
plt.show()

#Hierarchical cluster
data = jd.iloc[:, 3:5].values
dend = shc.dendrogram(shc.linkage(data, method="ward"))
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)
plt.figure(figsize=(10, 7))
plt.title("Dendrograma de los clientes")
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100')

plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')

print(df.describe().transpose())
print("Promedio Masculino: " + str(contadorMasc/(contadorFem+contadorMasc)))
print("Promedio Femenino: " + str(contadorFem/(contadorFem+contadorMasc)))