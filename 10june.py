import numpy as np
import random
import matplotlib.pyplot as plt
from  numpy  import zeros_like
from numpy import linalg as LA 
from scipy.linalg import block_diag
import scipy.linalg as sla
from scipy import optimize
from numpy import matrix, rank
float_formatter = lambda x: "%.1f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
import networkx as nx
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
sns.set()
# np.random.seed(0)

def grad_f(xx,kk):
    aa = np.asarray(np.dot(C[kk],xx)).reshape(-1)
    bb = np.asarray(b[kk]).reshape(-1)
    #print(aa.shape, bb.shape) 
    g_temp = aa - bb
    g_f = np.dot(C[kk].T,g_temp)
    #print(aa.shape, bb.shape, g_temp.shape,g_f.shape)
    return g_f
def func(xx,C,b):
    f = 0.5*((LA.norm(np.dot(np.asmatrix(C),xx) - b))**2) + theta_f*0.5*((LA.norm(xx))**2)
    return f
def flambda_max(B):
    zna, vek = np.linalg.eigh(B)
    lambda_max  = max(zna) 
    return lambda_max

def flambda_min(B):
    zna, vek = np.linalg.eigh(B)
    lambda_min  = min(zna) 
    return lambda_min

def flambda_min_plus(B):#наименьшее положительное соб знач
    zna, vek = np.linalg.eigh(B)
    flambda_min_plus = min([n for n in zna if n>1e-6])
    return flambda_min_plus 
def alg1(number_agent ,iter_a, A, eta_x, alfa_x,  beta_x,  tao_x,  sigma_x):
    x = np.asarray(A[0,:]).reshape(-1)
    
    x_f =  np.asarray(A[0,:]).reshape(-1)
    
    zna_func = np.zeros(number_agent)
    sum_iter = np.zeros(iter_a)
    C_st = np.vstack(C)
    b_st = np.hstack(b)
    x_s = sla.lstsq(C_st,b_st)[0]
    #print(sla.norm(C_st.dot(x_s)-b_st)**2/2)
    f_star = 1/(number_agent*0.5)*((LA.norm(np.dot(C_st,x_s) - b_st))**2)
    #print(f_star)
    for ii in range(iter_a):
        for j in range (number_agent):
            #здесь считаем функцию(ii-ю сумму), общую для всех number_agent для ii-й итерации
            x_g = tao_x*x  + (1-tao_x)*x_f     
            xprev = x
            zz =  np.asarray(np.dot(A,x)).reshape(-1)
            zzz =  np.asarray(np.dot(A.T,zz)).reshape(-1)
            x = xprev + eta_x*alfa_x*(x_g - x) - eta_x*beta_x*zzz - eta_x*grad_f(x_g,j) 
            x_f  = x_g + sigma_x*(x - xprev)
            zna_func[j] = func(x_f,C[j],b[j,:])
        sum_iter[ii] = sum(zna_func)   
        
    return  sum_iter - f_star  

n = 3
gg = 0.8
G = nx.random_graphs.erdos_renyi_graph(9, 1,   directed=False)
W = nx.adjacency_matrix(G)
D = np.diag(np.sum(np.array(W.todense()), axis=1))
L = D - W
C = torch.randint(high =3, size = (18,18,18)) # kk<=17
b = np.random.rand(18,18)#use in func 
s = 18
m = s
cm = np.matrix(np.random.randint(10, size=(9, 1)))
B = np.dot(cm,cm.T)
beta_min_plus = flambda_min_plus(B)
w_min_plus = flambda_min_plus(L)
gamma_x = beta_min_plus/w_min_plus
gW = gamma_x*L
A  = np.bmat([B,gW])
C = torch.randint(high =3, size = (s,s,s))
theta_f = 0.9
lambda_max = np.zeros((m), dtype = np.int8)
zz =  np.zeros((m,m), dtype = np.int8)

for i in range(m):                #получили список  из 18 lambda_max
    zz = np.dot(C[i].T,C[i])
    zna, vek = np.linalg.eigh(zz)
    lambda_max [i] = max(zna) 

lambda_min =  np.zeros((m), dtype = np.int8)
zz1 =  np.zeros((m,m), dtype = np.int8)
for i in range(m):                #получили список  из 18 lambda_max
    zz1 = np.dot(C[i].T,C[i])
    znamin, vek = np.linalg.eigh(zz1)
    lambda_min [i] = min(znamin) 
y_star = np.zeros(18)
k_mnk = np.zeros((18,18),dtype = np.int8)
b_mnk = np.zeros((18,18),dtype = np.int8)
L_x = max(lambda_max) + theta_f
mu_x = min(lambda_min) + theta_f
alfa_x = mu_x
sigma_x = np.sqrt(mu_x/(2*L_x))
mu_xy = np.sqrt(flambda_min_plus(np.dot(A,A.T)))
L_xy = np.sqrt(flambda_max(np.dot(A.T,A)))
delta = np.sqrt((mu_xy*mu_xy)/(2*mu_x*L_x))
eta_x = 0.0004#min((1/(4*(mu_x + L_x*sigma_x))), delta/(4*L_xy))
tao_x = 1/(sigma_x + 0.5)
beta_x = np.minimum(1/(2*L_x), 1/(2*eta_x*L_xy*L_xy))
iterac = 36
plot_x_f =  np.arange(0,iterac,1,dtype = np.int8)
plx_f =  np.arange(0,iterac,1,dtype = np.int8)
plot_x_f  = alg1( 18,iterac,A, eta_x, alfa_x,  beta_x,  tao_x,  sigma_x)
#should find f_star (минимизируем сумму f(i)) 
fig, ax = plt.subplots()

 
print(' condition number  = ',(LA.cond(W .todense())))
ax.grid()

ax.set_xlabel('number of iterations ' )
ax.set_ylabel('f - f*' )
 
 
ax.plot(plx_f, plot_x_f)
ax.set_yscale('log')
