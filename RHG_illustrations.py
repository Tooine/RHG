################################################################################
#####                         RHG -- ILLUSTRATIONS                         #####
################################################################################

import numpy as np
import numpy.random as rd
import pylab as plt
plt.ion(), plt.show()
from mpl_toolkits.mplot3d import Axes3D
import time

import os
main_folder = #TODO
os.chdir(main_folder + '/python')
from RHG_functions import *

dir_fig = main_folder + '/figures'
if not os.path.exists(dir_fig): os.makedirs(dir_fig)
os.chdir(dir_fig)

## fig 2_1

alpha = 0.7
n = 500
nu = 1

R = 2*np.log(n/nu)

V, V_cart, E = genere_G(n, alpha, nu)

plt.figure(figsize=(9, 9)), plt.title("RHG, (alpha, nu, n) = ({}, {}, {})".format(alpha, nu, n))
plot_graph_with_E(V_cart, E, R)
plot_circles_and_center(R)
plt.axis('square')
plt.savefig('fig_2_1.eps', format='eps', bbox_inches='tight', dpi=1200)

## fig 2_2

ALPHA = [0.5, 0.7, 1.2]
NU = [0.5, 1, 5]
n = 500
NAME_FIG = ['fig_2_2_a.eps', 'fig_2_2_b.eps', 'fig_2_2_c.eps', 'fig_2_2_d.eps', 'fig_2_2_e.eps', 'fig_2_2_f.eps', 'fig_2_2_g.eps', 'fig_2_2_h.eps', 'fig_2_2_i.eps']
lim = 15
i = 0

for nu in NU:
    for alpha in ALPHA:
        R = 2*np.log(n/nu)
        V, V_cart, E = genere_G(n, alpha, nu)
        plt.figure(figsize=(9, 9)), plt.title("RHG, (alpha, nu, n) = ({}, {}, {})".format(alpha, nu, n))
        plt.axis('square'), plt.xlim(-lim, lim), plt.ylim(-lim, lim)
        plot_graph_with_E(V_cart, E, R)
        plot_circles_and_center(R)
        plt.savefig(NAME_FIG[i], format='eps', bbox_inches='tight', dpi=1200)
        i += 1

## fig 2_2: density (not in the report)

def f(r, alpha, R):
    """ cumulative distribution function of the radial coordinate """
    
    if r <= 0: return 0
    if r > R: return 0
    return alpha * np.sinh(alpha * r) / ((np.cosh(alpha * R) - 1) * 2*np.pi)

def np_f(r, alpha, R):
    r = np.asarray(r)
    lst_f = np.copy(r)
    
    if len(r.shape) == 1:
        l = r.shape[0]
        for i in range(l):
            lst_f[i] = f(r[i], alpha, R)
    if len(r.shape) == 2:
        l, m = r.shape
        for i in range(l):
            for j in range(m):
                lst_f[i, j] = f(r[i, j], alpha, R)

    return lst_f

#
ALPHA = [0.5, 0.7, 1.2]
nu = 0.5
n = 500
lim = 15
NAME_FIG = ['fig_2_2_j.eps', 'fig_2_2_k.eps', 'fig_2_2_l.eps']

#
R = 2 * np.log(n/nu)

x = np.linspace(-lim, lim, 100)
y = np.linspace(-lim, lim, 100)
X, Y = np.meshgrid(x, y)

for (i, alpha) in enumerate(ALPHA):
    z = np_f(np.sqrt(X**2 + Y**2), alpha, R)

    plt.figure()
    plt.axis('square'), plt.xlim(-lim, lim), plt.ylim(-lim, lim), plt.title("radial density function (alpha, nu, n) = ({}, {}, {})".format(alpha, nu, n))
    plt.pcolor(x, y, z)
    plt.colorbar()
    plt.savefig(NAME_FIG[i], format='eps', bbox_inches='tight', dpi=1200)

## fig 3_1

# balls

R = 10
theta = 0

lst_r = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NAME_FIG = ['fig_3_1_r1.eps', 'fig_3_1_r2.eps', 'fig_3_1_r3.eps', 'fig_3_1_r4.eps', 'fig_3_1_r5.eps', 'fig_3_1_r6.eps', 'fig_3_1_r7.eps', 'fig_3_1_r8.eps', 'fig_3_1_r9.eps']

i = 0
for r in lst_r:
    plt.figure(figsize=(9, 9)), plt.axis('square'), plt.xlim(-11, 21), plt.ylim(-11, 11), plt.title("r = {}".format(r))
    plot_circles_and_center(R, col='k')
    plot_ball(r, theta, R, colB='g', center=True, colC='y')
    plt.savefig(NAME_FIG[i], format='eps', bbox_inches='tight', dpi=1200)
    i += 1

## fig_3_2: balls around two vertices, with neighbors highlighted

alpha = 0.7
n = 1000
nu = 1

R = 2*np.log(n/nu)

V, V_cart, E = genere_G(n, alpha, nu)

plt.figure(figsize=(9, 9)), plt.title("RHG, (alpha, nu, n) = ({}, {}, {})".format(alpha, nu, n))
plot_graph_with_E(V_cart, E, R)
plot_circles_and_center(R)
plt.axis('square')

# research of a point near R/2
i_1 = 0
while V[i_1, 0] > 9*R/16 or V[i_1, 0] < 7*R/16:
    i_1 +=1
# research of a point near 3R/4
i_2 = 0
while V[i_2, 0] > 13*R/16 or V[i_2, 0] < 11*R/16:
    i_2 +=1
plot_ball_in_graph(V, V_cart, E, i_1, R, edges=True, colB='g', colC='y', colE='g')
plot_ball_in_graph(V, V_cart, E, i_2, R, edges=True, colB='r', colC='y', colE='r')

plt.savefig('fig_3_2.eps', format='eps', bbox_inches='tight', dpi=1200)

## fig 3_3_a: requests on the upper bands

R = 10
theta = 0
r = 8.25

lst_bands = np.arange(5, 11)
lst_bands = 0.5 * np.arange(10, 21)

plt.figure(figsize=(9, 9)), plt.axis('square'), plt.xlim(-11, 11), plt.ylim(-11, 11)
plot_circles_and_center(R, col='k')
plot_ball(r, theta, R, colB='g', center=True, colC='y')

plot_circle(lst_bands[0], col='k', lin='--')
d_theta_band = []
for i in range(len(lst_bands)-1):
    r_min = lst_bands[i]
    r_max = lst_bands[i+1]
    plot_circle(r_max, col='k', lin='--')
    theta = d_theta_max(r, r_min, R)
    d_theta_band.append(theta)
    plot_request(r_min, r_max, -theta, theta)

plt.savefig('fig_3_3_a.eps', format='eps', bbox_inches='tight', dpi=1200)
    
## fig 3_3_b: vertices + request

alpha = 0.8
n = 1000
R = 10
nu = np.exp(np.log(n) - R/2)

V, V_cart, E = genere_G(n, alpha, nu)

plt.figure(figsize=(9, 9)), plt.axis('square'), plt.xlim(-11, 11), plt.ylim(-11, 11)
plot_graph_with_E(V_cart, E, R, edges=False, colV='k')
plt.plot([0], [0], color='k', marker='+', markersize=10)
plot_circle(R)

theta = 0
r = 6.5

lst_bands = np.arange(8, 10)

plot_ball(r, theta, R, colB='g', center=True, colC='y')

d_theta_band = []
r_min = lst_bands[0]
r_max = lst_bands[1]
plot_circle(r_min, col='k', lin='--')
plot_circle(r_max, col='k', lin='--')
theta = d_theta_max(r, r_min, R)
d_theta_band.append(theta)
plot_request(r_min, r_max, -theta, theta)

theta_max = d_theta_max(r, r_min, R)
for i in range(n):
    r, theta = V[i]
    if r_min <= r and r < r_max and (theta <= theta_max or 2*np.pi - theta <= theta_max):
        plt.plot(V_cart[i, 0], V_cart[i, 1], color='r', marker='o', markersize=3, linestyle="None")

plt.savefig('fig_3_3_b.eps', format='eps', bbox_inches='tight', dpi=1200)

## fig 3_4: execution time of the two algorithms

nu = 1
alpha_1 = 1
lst_n_1 = np.array(10**(np.arange(10, 17)/5), dtype=int)
lst_n_2 = np.array(10**(np.arange(10, 21)/5), dtype=int)
ALPHA_2 = [0.5, 0.6, 0.7, 0.8, 0.9]

# algo 1

lst_time_1 = []
for n in lst_n_1:
    R = 2*np.log(n/nu)    
    debut = time.time()
    V, V_cart = genere_V(n, alpha_1, R)
    E = genere_E(V, n, R)
    fin = time.time()
    lst_time_1.append(fin-debut)
    print("algo 1 : ", n)

# algo 2

lst_time_2 = []
for alpha in ALPHA_2:
    for n in lst_n_2:
        R = 2*np.log(n/nu)
        debut = time.time()
        V, V_cart, E = genere_G(n, alpha, nu)
        fin = time.time()
        lst_time_2.append(fin-debut)
        print(alpha, n)
lst_time_2 = np.asarray(lst_time_2).reshape((len(ALPHA_2), lst_n_2.shape[0]))

# plot
plt.figure(figsize=(12, 6)), plt.title("Generation time"), plt.xlabel("n"), plt.ylabel("Time")
plt.loglog(lst_n_1, lst_time_1, marker='o', markersize=5, linestyle='None', label="Algo 1")
for (i, alpha) in enumerate(ALPHA_2):
    plt.loglog(lst_n_2, lst_time_2[i], marker='o', markersize=5, linestyle='None', label="alpha = {}".format(alpha))
L = np.polyfit(np.log10(lst_n_1), np.log10(lst_time_1), 1)
plt.loglog(lst_n_1, lst_n_1**L[0]*10**L[1], color = 'b', label='linear: a={:2.3f}'.format(L[0]))
L = np.polyfit(np.log10(lst_n_2), np.log10(lst_time_2[0]), 1)
plt.loglog(lst_n_2, lst_n_2**L[0]*10**L[1], color = 'g', label='linear: a={:2.3f}'.format(L[0]))
L = np.polyfit(np.log10(lst_n_2), np.log10(lst_time_2[-1]), 1)
plt.loglog(lst_n_2, lst_n_2**L[0]*10**L[1], color = 'yellow', label='linear: a={:2.3f}'.format(L[0]))
plt.legend(loc='best')

plt.savefig('fig_3_4.eps', format='eps', bbox_inches='tight', dpi=1200)

## fig 3_5: power-law, one graph per parameter

ALPHA = [0.3, 0.5, 0.7, 0.9, 1.1]
nu = 1
NN = [500, 1000, 5000]
lim = 20

NAME_FIG = ['fig_3_4_1_a.eps', 'fig_3_4_1_b.eps', 'fig_3_4_1_c.eps', 'fig_3_4_1_d.eps', 'fig_3_4_1_e.eps', 'fig_3_4_2_a.eps', 'fig_3_4_2_b.eps', 'fig_3_4_2_c.eps', 'fig_3_4_2_d.eps', 'fig_3_4_2_e.eps', 'fig_3_4_3_a.eps', 'fig_3_4_3_b.eps', 'fig_3_4_3_c.eps', 'fig_3_4_3_d.eps', 'fig_3_4_3_e.eps']

i = 0
for n in NN:
    R = 2*np.log(n/nu)
    for alpha in ALPHA:
        V, V_cart, E = genere_G(n, alpha, nu)
        
        plt.figure(i, figsize=(16, 9)), plt.xlabel('Degree'), plt.ylabel('number of vertices')
        N = genere_N_from_E(n, E)
        hist_degrees = genere_hist_degrees(N, alpha=alpha)
        
        plt.pause(1)
        i_moy = int(input("i_moy: "))
        i_max = int(input("i_max: "))
        
        plt.figure(i, figsize=(16, 9)), plt.clf(), plt.xlabel('Degree'), plt.ylabel('number of vertices')
        hist_degrees = genere_hist_degrees(N, alpha=alpha, i_moy=i_moy, i_max=i_max)
        
        plt.title("Degree distribution, (alpha, nu, n) = ({}, {}, {}) - D_max = {}".format(alpha, nu, n, len(hist_degrees)))
        plt.savefig(NAME_FIG[i], format='eps', bbox_inches='tight', dpi=1200)
        i += 1

## fig 3_6: power-law, ten graphs per parameter

ALPHA = [0.3, 0.5, 0.7, 0.9, 1.1]
nu = 1
NN = [500, 1000, 5000]
lim = 20
nb_graphs = 10

NAME_FIG = ['fig_3_5_1_a.eps', 'fig_3_5_1_b.eps', 'fig_3_5_1_c.eps', 'fig_3_5_1_d.eps', 'fig_3_5_1_e.eps', 'fig_3_5_2_a.eps', 'fig_3_5_2_b.eps', 'fig_3_5_2_c.eps', 'fig_3_5_2_d.eps', 'fig_3_5_2_e.eps', 'fig_3_5_3_a.eps', 'fig_3_5_3_b.eps', 'fig_3_5_3_c.eps', 'fig_3_5_3_d.eps', 'fig_3_5_3_e.eps']

NN = [1000]
ALPHA = [0.3]

i = 5
for n in NN:
    R = 2*np.log(n/nu)
    for alpha in ALPHA:
        N = []
        for graph in range(nb_graphs):
            V, V_cart, E = genere_G(n, alpha, nu)
            N += genere_N_from_E(n, E)
        
        plt.figure(i, figsize=(16, 9)), plt.xlabel('Degree'), plt.ylabel('number of vertices')
        hist_degrees = genere_hist_degrees(N, alpha=alpha)
        
        plt.pause(1)
        i_moy = int(input("i_moy: "))
        i_max = int(input("i_max: "))
        
        plt.figure(i, figsize=(16, 9)), plt.clf(), plt.xlabel('Degree'), plt.ylabel('number of vertices')
        plt.title("Degree distribution ({} graphs), (alpha, nu, n) = ({}, {}, {}) - D_max = {}".format(nb_graphs, alpha, nu, n, len(hist_degrees)))
        hist_degrees = genere_hist_degrees(N, alpha=alpha, i_moy=i_moy, i_max=i_max)
        plt.savefig(NAME_FIG[i], format='eps', bbox_inches='tight', dpi=1200)
        i += 1

## fig 3_7: power-law, role of nu

alpha = 0.5
n = 1000
NU = [0.5, 1, 5]
lim = 20

NAME_FIG = ['fig_3_6_a.eps', 'fig_3_6_b.eps', 'fig_3_6_c.eps']

i = 0
for nu in NU:
    R = 2*np.log(n/nu)
    V, V_cart, E = genere_G(n, alpha, nu)
    
    plt.figure(i, figsize=(16, 9)), plt.xlabel('Degree'), plt.ylabel('number of vertices')
    N = genere_N_from_E(n, E)
    hist_degrees = genere_hist_degrees(N, alpha=alpha)
    
    plt.pause(1)
    i_moy = int(input("i_moy: "))
    i_max = int(input("i_max: "))
    
    plt.figure(i, figsize=(16, 9)), plt.clf(), plt.xlabel('Degree'), plt.ylabel('number of vertices')
    hist_degrees = genere_hist_degrees(N, alpha=alpha, i_moy=i_moy, i_max=i_max)
    
    plt.title("Degree distribution, (alpha, nu, n) = ({}, {}, {}) - D_max = {}".format(alpha, nu, n, len(hist_degrees)))
    plt.savefig(NAME_FIG[i], format='eps', bbox_inches='tight', dpi=1200)
    i += 1

NAME_FIG = ['fig_3_6_d.eps', 'fig_3_6_e.eps', 'fig_3_6_f.eps']

i = 0
for nu in NU:
    R = 2*np.log(n/nu)
    V, V_cart, E = genere_G(n, alpha, nu)
    
    plt.figure(figsize=(9, 9)), plt.title("RHG, (alpha, nu, n) = ({}, {}, {})".format(alpha, nu, n))
    plot_graph_with_E(V_cart, E, R)
    plot_circles_and_center(R)
    plt.axis('square'), plt.xlim(-lim, lim), plt.ylim(-lim, lim)
    plt.savefig(NAME_FIG[i], format='eps', bbox_inches='tight', dpi=1200)
    i += 1

## fig 3_8: size of the giant component

ALPHA = [0.9, 1.1]
nu = 1
NN = [1000, 2000, 3000]
lim = 20

NAME_FIG = ['fig_3_7_a.eps', 'fig_3_7_b.eps', 'fig_3_7_c.eps', 'fig_3_7_d.eps', 'fig_3_7_e.eps', 'fig_3_7_f.eps']

i = 0
for alpha in ALPHA:
    for n in NN:
        R = 2*np.log(n/nu)
        V, V_cart, E = genere_G(n, alpha, nu)
        N = genere_N_from_E(n, E)
        connected_compos = genere_connected_compos(N)
        plt.figure(figsize=(9, 9)), plt.title("RHG, (alpha, nu, n) = ({}, {}, {}) - |C_1| = {}".format(alpha, nu, n, len(connected_compos[0])))
        plt.axis('square'), plt.xlim(-lim, lim), plt.ylim(-lim, lim)
        plot_graph_with_E(V_cart, E, R)
        plot_compo_with_E(V_cart, E, connected_compos, R)
        plot_circles_and_center(R)
        plt.savefig(NAME_FIG[i], format='eps', bbox_inches='tight', dpi=1200)
        print(alpha, n, len(connected_compos[0]))
        i += 1

alpha = 1
NU = [np.pi/16, 40 * np.pi]
NN = [1000, 5000, 10000]
lim = 20

NAME_FIG = ['fig_3_7_g.eps', 'fig_3_7_h.eps', 'fig_3_7_i.eps', 'fig_3_7_j.eps', 'fig_3_7_k.eps', 'fig_3_7_l.eps']

i = 0
for nu in NU:
    for n in NN:
        R = 2*np.log(n/nu)
        V, V_cart, E = genere_G(n, alpha, nu)
        N = genere_N_from_E(n, E)
        connected_compos = genere_connected_compos(N)
        #plt.figure(figsize=(9, 9)), plt.title("RHG, (alpha, nu, n) = ({}, {}, {}) - |C_1| = {}".format(alpha, nu, n, len(connected_compos[0])))
        #plt.axis('square'), plt.xlim(-lim, lim), plt.ylim(-lim, lim)
        #plot_graph_with_E(R, V_cart, E)
        #plot_compo_with_E(R, V_cart, E, connected_compos)
        #plot_circles_and_center(R)
        #plt.savefig(NAME_FIG[i], format='eps', bbox_inches='tight', dpi=1200)
        print(nu, n, len(connected_compos[0]))
        i += 1

## fig 3_9: average size of the giant component (10 graphs)

ALPHA = [0.9, 1.1]
nu = 1
NN = [1000, 5000, 10000]
nb_graphs = 10

avg_size = []

for alpha in ALPHA:
    for n in NN:
        R = 2*np.log(n/nu)
        size = 0
        for i in range(nb_graphs):
            V, V_cart, E = genere_G(n, alpha, nu)
            N = genere_N_from_E(n, E)
            connected_compos = genere_connected_compos(N)
            size += len(connected_compos[0])
            #print(len(connected_compos[0]))
        avg_size.append(size / nb_graphs)

## fig 3_10, 3_11 & 3_12: push & pull

# parameters

alpha = 0.9
nu = 1
n = 1000
infected_vertice = -1  # -1 : choose a random vertice in the central component
plot_all_graph = False
d_time = 0.1
legend = False
lim = 15

# program

R = 2 * np.log(n/nu)

V, V_cart, E = genere_G(n, alpha, nu)
N = genere_N_from_E(n, E)
connected_compos = genere_connected_compos(N)

if legend: plt.figure(figsize=(16, 9)), plt.axis('equal')
else: plt.figure(figsize=(9, 9)), plt.axis('square'), plt.xlim(-lim, lim), plt.ylim(-lim, lim)
if plot_all_graph: plot_graph_with_E(V_cart, E, R)
else: plot_compo_with_E(V_cart, E, connected_compos, R, num=0, vertices=True, edges=True, colV='b', colE='gray')
plot_circles_and_center(R)

T, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t = push_pull_with_plot(V, V_cart, N, connected_compos, n, alpha, nu, R, first_informed=-1, break_time=d_time, save=True, name_fig="fig_3_10_", legend=False)

plt.figure(figsize=(16, 9)), plt.title("(alpha, nu, n, n_real) = ({}, {}, {}, {})".format(alpha, nu, n, n_real))
plot_stat_push_pull(T, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t)
plt.savefig('fig_3_10_graphic.eps', format='eps', bbox_inches='tight', dpi=1200)

## fig 3_13: average time of propagation

# parameters

ALPHA = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
nu = 1
lst_n = np.array(10**(np.arange(10, 21)/5), dtype=int)
NAME_FIG = ['fig_3_13_a.eps', 'fig_3_13_b.eps', 'fig_3_13_c.eps', 'fig_3_13_d.eps', 'fig_3_13_e.eps', 'fig_3_13_f.eps']
infected_vertice = -1  # -1 : choose a random vertice in the giant component
nb_exp = 10
nb_graph = 20

# simulation program

def addtext(text, name):
    file = open(name,'a')
    file.write(text)
    file.close()

lst_R = 2 * np.log(lst_n/nu)
lst_paths = []
for alpha in ALPHA:
    dir = main_folder + '/simulations/push_pull_evolution/' + str(alpha)
    if alpha == int(alpha): dir += '_' 
    while os.path.exists(dir): dir += '0'
    lst_paths.append(dir)
    os.makedirs(dir)
    os.chdir(dir)
    
    file = open('stats.txt','w')
    file.close()
    
    for (n, R) in zip(lst_n, lst_R):
        print("n = {}".format(n))
        addtext("{}".format(n), 'stats.txt')
        for num_graph in range(nb_graph):
            print("    GRAPH {}".format(num_graph))
            V, V_cart, E = genere_G(n, alpha, nu)
            N = genere_N_from_E(n, E)
            connected_compos = genere_connected_compos(N)
            
            for num_exp in range(nb_exp):
                print("        {}".format(num_exp))
                simul = push_pull_without_plot(V, V_cart, N, connected_compos, n, alpha, nu, R, first_informed=-1)
                addtext(" {}".format(simul[0]), 'stats.txt')
        addtext("\n", 'stats.txt')

# plot of statistics

for (i, (alpha, dir)) in enumerate(zip(ALPHA, lst_paths)):
    os.chdir(dir)
    data = np.loadtxt('stats.txt')
    
    AVG = []
    SD = []
    for (j, n) in enumerate(lst_n):
        AVG.append(np.average(data[j, 1:]))
        SD.append(np.std(data[j, 1:]))
    AVG = np.asarray(AVG)
    SD = np.asarray(SD)
    TH = lst_n**(alpha - 1/2) 
    TH *= AVG[-1] / TH[-1]
    
    plt.figure(figsize=(12, 8)), plt.title("Average time - alpha = {}".format(alpha)), plt.xlabel("n"), plt.ylabel("Time")
    plt.semilogx(lst_n, TH, marker='o', markersize=8, linestyle="None", color='k', label="Cn^{alpha - 1/2}")
    plt.semilogx(lst_n, AVG, marker='o', markersize=8, linestyle="None", color='b', label="average")
    plt.semilogx(lst_n, AVG + SD, marker='+', markersize=5, linestyle="None", color='r', label="average +/- SD")
    plt.semilogx(lst_n, AVG - SD, marker='+', markersize=5, linestyle="None", color='r')
    plt.legend(loc='best')
    
    os.chdir(dir_fig)
    plt.savefig(NAME_FIG[i], format='eps', bbox_inches='tight', dpi=1200)

## fig 3_14: one layer graph propagation

def neighbor_choice(i, N):
    """ randomly choose of a neighbor of i """
    return N[i][rd.randint(0, len(N[i]))]

def genere_modified_graph(Nk, Dk, m):
    card_group = Nk // m
    Nk = m * card_group
    
    E = []
    for group in range(m-1):
        for i in range(group*card_group, (group+1)*card_group):
            for j in range(i+1, (group+1)*card_group): E.append((i, j))
            for j in range((group+1)*card_group, (group+2)*card_group): E.append((i, j))
    group = m - 1
    for i in range(group*card_group, (group+1)*card_group):
        for j in range(i+1, (group+1)*card_group): E.append((i, j))
        for j in range(0, card_group): E.append((i, j))
    
    N = genere_N_from_E(Nk, E)
    for i in range(len(N)):
        while len(N[i]) < Dk: N[i].append(i)
    
    return E, N

def push_auxiliary_graph(Nk, N, m, first_informed=-1, save=False, break_time=0.5, fig_title="", file_title=""):    
    t = 0
    states = np.zeros(Nk, dtype=int)
    informed_per_group = np.zeros(m)
    #if first_informed == -1: first_informed = rd.randint(0, Nk)
    if first_informed == -1: first_informed = int((Nk / m) / 2)
    informed_per_group[first_informed//(Nk//m)] = 1
    states[first_informed] = 4
    informed_at_t = [[first_informed]]
    nb_push_at_t, nb_pull_at_t, nb_both_at_t = [0], [0], [0]
    nb_infected = 1
    infected_by_push = np.zeros(Nk, dtype=object)
    infected_by_pull = np.zeros(Nk, dtype=object)
    for i in range(Nk):
        infected_by_push[i], infected_by_pull[i] = [], []
    
    for (i, nb) in enumerate(informed_per_group):
        plt.plot([i, i], [0, nb], color='r', linewidth=5)
    plt.title(fig_title + "t = {}".format(t)), plt.pause(break_time)
    
    if save: plt.savefig(file_title + 't{}'.format(t) + '.eps', format='eps', bbox_inches='tight', dpi=1200)
    
    while nb_infected != Nk:
        t += 1
        informed_at_t += [[]]
        new_infected_by_push = np.zeros(Nk, dtype=object)
        new_infected_by_pull = np.zeros(Nk, dtype=object)
        for i in range(Nk): 
            new_infected_by_push[i], new_infected_by_pull[i] = [], []
        for i in range(Nk):
            j = neighbor_choice(i, N)
            if states[i] != 0 and states[j] == 0:   # push
                new_infected_by_push[j].append(i)
                infected_by_push[j].append(i)
            if states[i] == 0 and states[j] != 0:   # pull
                new_infected_by_pull[i].append(j)
                infected_by_pull[i].append(j)
        
        nb_push, nb_pull, nb_both = 0, 0, 0
        for i in range(Nk):
            if len(new_infected_by_push[i]) + len(new_infected_by_pull[i]) > 0:
                states[i] = 1
                informed_per_group[i//(Nk//m)] += 1
                informed_at_t[-1].append(i)
                if len(new_infected_by_push[i]) > 0 and len(new_infected_by_pull[i]) == 0:
                    nb_push += 1
                    states[i] = 1
                elif len(new_infected_by_pull[i]) > 0 and len(new_infected_by_push[i]) == 0:
                    nb_pull += 1
                    states[i] = 2
                else:
                    nb_both += 1
                    states[i] = 3
        nb_push_at_t.append(nb_push_at_t[-1] + nb_push), nb_pull_at_t.append(nb_pull_at_t[-1] +  nb_pull), nb_both_at_t.append(nb_both_at_t[-1] + nb_both)
        nb_infected += len(informed_at_t[-1])
        
        for (i, nb) in enumerate(informed_per_group):
            plt.plot([i, i], [0, nb], color='r', linewidth=5)
        plt.title(fig_title + "t = {}".format(t)), plt.pause(break_time)
        
        if save: plt.savefig(file_title + 't{}'.format(t) + '.eps', format='eps', bbox_inches='tight', dpi=1200)
    return t, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t, infected_by_push, infected_by_pull

# parameters

Nk = 5000
Dk = 1000
m = 50

# program

E, N = genere_modified_graph(Nk, Dk, m)

plt.figure(figsize=(16, 9)), plt.xlim([-0.5, m-1+0.5]), plt.ylim([0, Nk // m+1]), plt.title("(Nk, Dk, m) = ({}, {}, {}) - t = {}".format(Nk, Dk, m, 0))
plt.xlabel("group")
plt.ylabel("number of informed vertices")

t, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t, infected_by_push, infected_by_pull = push_auxiliary_graph(Nk, N, m, first_informed=Nk//2, save=True, fig_title="(Nk, Dk, m) = ({}, {}, {}) - ".format(Nk, Dk, m), file_title="fig_3_14_")
