import pandas
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from numpy import linalg as LA
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib
import time
import pickle

'''
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
'''


def data_handler(filename, parse_label=True, random_sort=False):
    data = pandas.read_excel(filename)

    data = data.as_matrix()

    if random_sort:
        random.shuffle(data)

    if parse_label:
        label = data[:, 0]
        data = data[:, 1:]

        return data, label

    return data


def scree_plot(no_comp, vals):
    """
	# This function is created to observe eigen values of a matrix
	# no_comp variable takes number of eigen of eigen values to be visualize
	# vals is a list or numpy array containing eigen values
	"""

    fig = plt.figure(figsize=(8, 5))

    comp = np.arange(no_comp) + 1
    plt.plot(comp, vals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigen Values')
    leg = plt.legend(['Eigen Values'], loc='best', borderpad=0.3,
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)
    plt.show()


def load_cached_matrices(K=False, P=False, P_t=False, P_prime_decomposition=False, P_prime_svd=False):
    """
	# This function returns pre calculated values
	# The sole purpose of this function is to reduce time.
	# These precalculated matrices are for t=10 and sigma=5
	# If P_prime_decomposition is true it will return a list containing eigen values and eigen vectors of P_prime matrix.
	"""
    K_mat = None
    P_mat = None
    P_t_mat = None
    P_prime_eig = []
    P_prime_svd_m = []

    if K:
        K_mat = pickle.load(open('K_mat', 'rb'))
    if P:
        P_mat = pickle.load(open('P_mat', 'rb'))
    if P_t:
        P_t_mat = pickle.load(open('P_t_mat', 'rb'))
    if P_prime_decomposition:
        P_prime_eig.append(pickle.load(open('Eig_vec', 'rb')))
        P_prime_eig.append(pickle.load(open('Eig_vals', 'rb')))
    if P_prime_svd:
        P_prime_svd_m.append(pickle.load(open('U_mat', 'rb')))
        P_prime_svd_m.append(pickle.load(open('S_mat', 'rb')))
        P_prime_svd_m.append(pickle.load(open('v_t_mat', 'rb')))

    return K_mat, P_mat, P_t_mat, P_prime_eig, P_prime_svd


def plot3D(data_mat, label_mat):
    x_points = data_mat[:, 0]
    y_points = data_mat[:, 1]
    z_points = data_mat[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color = []
    for i in label_mat:
        if i == 0:
            color.append('b')

        else:
            color.append('r')

    ax.scatter3D(x_points, y_points, z_points, c=color, label=label_mat, cmap='viridis')
    plt.show()


def Pclus(kmeans, n=2):
    print("k means clusters are")

    tmp = kmeans.labels_
    for i in range(n):
        kt = tmp[tmp == i]
        print(kt.shape)


def apply_kmeans(data_mat, clusters=2, message='null'):
    if message != 'null':
        print(message)

    kmeans = KMeans(n_clusters=clusters).fit(data_mat)
    Pclus(kmeans, clusters)


def kmatrix(data_mat, sigma):
    return rbf_kernel(data_mat, data_mat, 1 / (2 * (sigma ** 2)))


def calculate_matrices(data_mat, sigma, t_parameter, comp, cached=True, dump=False):
    """
	# This function is used to calculate all matrices and values
	# cached variable if true precalculated values are loaded
	# dump variable if true cache all variables calcutated here, note that dump can be true only if cached is false 
	"""

    if not cached:
        K = kmatrix(training_data, sigma)

        D = np.sum(K, axis=1)

        D_inv = 1 / D

        D_half = np.sqrt(D)
        D_inv_half = np.sqrt(D_inv)

        D_inv = np.diag(D_inv)
        D_half = np.diag(D_half)
        D_inv_half = np.diag(D_inv_half)

        P = D_inv.dot(K)

        P_t = LA.matrix_power(P, t_parameter)

        P_prime_t = D_half.dot(P_t.dot(D_inv_half))
        P_prime_eigen_vals, P_prime_eigen_vacs = LA.eig(P_prime_t)

        if dump:
	        pickle.dump(P_t, open('P_t_mat', 'wb'))
	        pickle.dump(P, open('P_mat', 'wb'))
	        pickle.dump(P_prime_eigen_vacs, open('Eig_vec', 'wb'))
	        pickle.dump(P_prime_eigen_vals, open('Eig_vals', 'wb'))
	        pickle.dump(K, open('K_mat', 'wb'))
        return K, P, D_half, D_inv_half, P_t, P_prime_t, P_prime_eigen_vals, P_prime_eigen_vacs

    else:
        K, P, P_t, [P_prime_eigen_vacs, P_prime_eigen_vals], _ = load_cached_matrices(K=True, P=True, P_t=True,
                                                                                      P_prime_decomposition=True)

        D = np.sum(K, axis=1)
        D_inv = 1 / D

        D_half = np.sqrt(D)
        D_inv_half = np.sqrt(D_inv)

        D_inv = np.diag(D_inv)
        D_half = np.diag(D_half)
        D_inv_half = np.diag(D_inv_half)

        P_prime_t = D_half.dot(P_t.dot(D_inv_half))

        return K, P, D_half, D_inv_half, P_t, P_prime_t, P_prime_eigen_vals, P_prime_eigen_vacs


if __name__ == "__main__":
    sigma = 5
    t_parameter = 10
    comp = 3

    training_data, training_labels = data_handler('HW_TESLA.xlt', random_sort=True)

    K, P, D_half, D_inv_half, P_t, P_prime_t, P_prime_eigen_vals, P_prime_eigen_vacs = calculate_matrices(training_data,
                                                                                                          sigma,
                                                                                                          t_parameter,
                                                                                                          comp,cached=False,dump=False)

    idx = P_prime_eigen_vals.argsort()[::-1]
    P_prime_eigen_vals = P_prime_eigen_vals[idx].real
    scree_plot(10, P_prime_eigen_vals[0:10])

    P_prime_eigen_vals = np.diag(P_prime_eigen_vals)

    P_prime_eigen_vacs = P_prime_eigen_vacs[:, idx].real

    Q = D_inv_half.dot(P_prime_eigen_vacs)
    Q_inv = P_prime_eigen_vacs.T.dot(D_half)

    # P_new = Q.dot(P_prime_eigen_vals)
    # apply_kmeans(P_new.T)

    apply_kmeans(training_data, message='Applying KMeans on original data...')

    apply_kmeans(P_t.T, message='Applying KMeans on Diffusion Matrix after t steps...')

    P_projected = Q[:, :comp].dot(P_prime_eigen_vals[:comp, :comp])

    apply_kmeans(P_projected, message='Applying KMeans on Diffusion Matrix after dimensionality reduction...')
    
    # ########################SINGULAR VALUE DECOMPOSITION####################################
    # U, S, v_t = np.linalg.svd(P_t, full_matrices=False)
    # S = np.diag(S)
    # P_projected = U[:, :comp].dot(S[:comp, :comp])

    # apply_kmeans(P_projected,
    #              message='Applying KMeans on Diffusion Matrix after dimensionality reduction using SVD...')

    plot3D(P_projected, training_labels)
