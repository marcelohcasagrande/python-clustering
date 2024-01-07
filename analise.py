# Importando pacotes.
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt


    #                                   # 
    # Lendo banco de dados e tratativas #
    #                                   # 

# Lendo banco de dados.
df = pd.read_csv( 'datasets/CC GENERAL.csv' )
df.drop( columns=[ 'CUST_ID', 'TENURE' ], inplace = True )
df.head()

# Caçando eventuais missings.
missing = df.isna().sum()
print( missing )

# Preenchendo valores missings com a mediana de cada variável.
df.fillna( df.median(), inplace = True )
missing = df.isna().sum()
print( missing )

# Normalizando dados para não ter interferência de escala.
values = Normalizer().fit_transform( df.values ) # Fitando e transformando.
print( values )

# Aplicando KMeans com 5 clusters.
kmeans = KMeans( n_clusters = 5, n_init = 10, max_iter = 300 )
y_pred = kmeans.fit_predict( values )

# Pegando os labels atribuídos e calculando Silhouette Score.
# O valor resultante do cálculo está entre -1 e 1, sendo que, quando mais próximo de 
# 1 significa que os clusters são densos e bem separados, 
# quando está próximo de 0 significa que os clusters estão sobrepostos e 
# finalmente valores negativos significam que a clusterização está incorreta.
labels = kmeans.labels_
silhouette = metrics.silhouette_score( values, labels, metric = 'euclidean' ) # baseado na distância euclidiana.
print( silhouette ) 

# Calculando Davies Bouldin Score. Quanto menor, melhor.
dbs = metrics.davies_bouldin_score( values, labels )
print( dbs ) # Q

# Calculando Calinski Harabasz Score. # Quanto maior, melhor.
calinski = metrics.calinski_harabasz_score( values, labels )
print( calinski )


    #                                    #
    # Criando função para criar métricas #
    #                                    #

# Função.
def clustering_algorithm( n_clusters, dataset ):
    kmeans = KMeans( n_clusters = n_clusters, n_init = 10, max_iter = 300 )
    labels = kmeans.fit_predict( dataset )
    s = metrics.silhouette_score( dataset, labels, metric = 'euclidean') 
    dbs = metrics.davies_bouldin_score( dataset, labels )
    calinski = metrics.calinski_harabasz_score( dataset, labels )
    return round( s, 2 ), round( dbs, 2 ), round( calinski, 2 ) 


# Vendo resultado para 3 clusters.
s1, dbs1, calinski1 = clustering_algorithm( 3, values )
print( s1, dbs1, calinski1 )

# Vendo resultado para 5 clusters.
s2, dbs2, calinski2 = clustering_algorithm( 5, values )
print( s2, dbs2, calinski2 )

# Vendo resultado para 50 clusters.
s3, dbs3, calinski3 = clustering_algorithm( 50, values )
print( s3, dbs3, calinski3 )


    #              #  
    # Graficamente #
    #              #

# Graficamente os rótulos olhando para 2 variáveis.    
plt.scatter( df[ 'PURCHASES' ], df[ 'PAYMENTS' ], c = labels, s = 5, cmap = 'rainbow' )
plt.xlabel( 'Valor total pago' )
plt.ylabel( 'Valor total gasto' )
plt.show()    

# Olhando descritiva das variáveis para cada cluster.
df[ 'cluster' ] = labels
df.groupby( 'cluster' ).describe()


    #                                                     #
    # Filtrando as variáveis mais relevantes nos clusters #
    #                                                     #
    
# Pegando os centróides de cada cluster para cada variável.
centroids = kmeans.cluster_centers_
print( centroids )

# Pegando a quantidade máxima e analisando a variabilidade dos centróides para cada variável.
max = len( centroids[ 0 ] )
for i in range( max ):
    print( df.columns.values[ i ], '\n{:.4f}'.format( centroids[:, i].var() ) )

# Filtrando variáveis que tiveram maior variabilidade nos centróides e criando visão de médias e quantidade de registros.
description = df.groupby( 'cluster' )[ [ 'BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS' ] ]
n_clients = description.size()
description = description.mean()
description[ 'n_clients' ] = n_clients
print( description )


    #                      #  
    # Tentando interpretar #
    #                      # 

# Rótulos dos clusters:
# CLUSTER 1: Clientes que gastam pouco. Clientes com o maior limite. Bons pagadores. Maior número de clientes.
# CLUSTER 3: Clientes que mais gastam. O foco deles é o saque. Piores pagadores. Boa quantidade de clientes.
# CLUSTER 2: Clientes que gastam muito com compras. Melhores pagadores.
# CLUSTER 0: Clientes que gastam muito com saques. Pagam às vezes.
# CLUSTER 4: Clientes com o menor limite. Não são bons pagadores. Menor quantidade de clientes.
    