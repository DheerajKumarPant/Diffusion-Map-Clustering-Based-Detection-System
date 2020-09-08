import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from numpy import linalg as LA
from sklearn.utils import shuffle
import pickle



def gaussian(v1,v2,sigma):
    return np.exp(-(LA.norm(v1-v2)**2)/(2*sigma**2))


def kmatrix(data,sigma = 1):
    data = np.asarray(data)
    rows,cols = data.shape
    
    
    mat = np.zeros(shape=(rows,rows))

    i = 0
    for v1 in data:
        j=0
        for v2 in data:
            mat[i][j] = gaussian(v1.T,v2.T,sigma)
            j+=1
        i+=1
    
    return mat



data = pd.read_excel(r'./HW_TESLA.xlt')

data = data.set_index("STATIC")

data0 = data.drop("1", axis=0)
data1 = data.drop("0", axis=0)

data0 = data0.reset_index()
data1 = data1.reset_index()


frames = [data0, data1]
training_data = pd.concat(frames)
training_data = shuffle(training_data)

training_data = training_data.to_numpy()

training_labels = training_data[:,0:1]

training_data = training_data[:,1:]

sigma = 5
t_parameter = 20

K = pickle.load(open('kmatrix', 'rb'))

D = np.sum(K, axis=1)
print(D)
D_inv = 1/D

D_half = np.sqrt(D)
	
D_inv_half = np.sqrt(D_inv)

D_inv = np.diag(D_inv)
D_half = np.diag(D_half)
D_inv_half = np.diag(D_inv_half)

P = D_inv.dot(K)

# P_t = P

# P_t = LA.matrix_power(P, t_parameter)

P_t = pickle.load(open('P_t', 'rb'))

P_prime_t = D_half.dot(P_t.dot(D_inv_half))

P_prime_eigen_vals, P_prime_eigen_vacs = LA.eig(P_prime_t)

idx = P_prime_eigen_vals.argsort()[::-1]
P_prime_eigen_vals = np.diag(P_prime_eigen_vals[idx].real)
P_prime_eigen_vacs = P_prime_eigen_vacs[:, idx].real

Q = D_inv_half.dot(P_prime_eigen_vacs)
Q_inv = P_prime_eigen_vacs.T.dot(D_half)


def Pclus(kmeans,n=2):
    print("k means clusters are")

    tmp = kmeans.labels_
    for i in range(n):
        kt = tmp[tmp == i]
        print(kt.shape)
        

P_new = Q.dot(P_prime_eigen_vals)
print(P_new.shape)
kmeans = KMeans(n_clusters=2).fit(P_new.T)
Pclus(kmeans)


kmeans = KMeans(n_clusters=2).fit(P_t.T)
Pclus(kmeans,2)

P_projected = Q[:,:3].dot(P_prime_eigen_vals[:3,:3])
print(P_projected.shape)

kmeans = KMeans(n_clusters=2).fit(P_projected)
Pclus(kmeans,2)

kmeans = KMeans(n_clusters=2).fit(training_data)
Pclus(kmeans,2)
print(training_data.shape)

U, S, v_t = np.linalg.svd(P_t, full_matrices=False)
S = np.diag(S)
P_projected_svd = U[:,:3].dot(S[:3,:3])
print(P_projected_svd.shape)

kmeans = KMeans(n_clusters=2).fit(P_projected_svd)
Pclus(kmeans,2)

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection="3d")

P_projected = P_projected

x_points = P_projected[:,0]
y_points = P_projected[:,1]
z_points = P_projected[:,2]


color = []

for i in training_labels:
    
    if i==0:
        color.append('b')
       
    else:
        color.append('r')
        
for i in range(len(P_projected)):
    ax.scatter3D(x_points[i], y_points[i],z_points[i], c=color[i] ,cmap='viridis')



print(training_labels.shape)
training_labels=training_labels.flatten()



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(30, 60)
ax.scatter3D(x_points, y_points, z_points, c=training_labels, label=training_labels)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

