################################################################################
#####                          RHG -- SIMULATIONS                          #####
################################################################################

import numpy as np
import numpy.random as rd
import pylab as plt
plt.ion(), plt.show()
import time

import os
main_folder = #TODO
os.chdir(main_folder + '/python')
from RHG_functions import *

##########                 simple generation and plot                 ##########

# parameters

alpha = 0.9
nu = 1
n = 1000

# program

R = 2 * np.log(n/nu)

V, V_cart = genere_V(n, alpha, R)
E, N = genere_E_N(V, n, R)
connected_compos = genere_connected_compos(N)

plt.figure(figsize=(9, 9)), plt.axis('equal'), plt.title("RHG, (alpha, nu, n) = ({}, {}, {})".format(alpha, nu, n))
plot_graph_with_E(V_cart, E, R)
plot_compo_with_E(V_cart, E, connected_compos, R)
plot_circles_and_center(R)

plt.figure(figsize=(16, 9)), plt.title("Degree distribution")
hist_degrees = genere_hist_degrees(N)

##########                  push and pull simulation                  ##########

# parameters

alpha = 0.8
nu = 0.5
n = 500
infected_vertice = -1  # -1 : choose a random vertice in the central component
plot_all_graph = False
d_time = 0.1
save = False

# program

if save:
    dir = main_folder + '/simulations/RHG_push_pull'
    if not os.path.exists(dir): os.makedirs(dir)
    os.chdir(dir)
    
    if os.path.isfile(dir + '/histo.txt'):
        histo = open('histo.txt', 'r')
        nb_exp = int(histo.read().split('\n')[-1][:6]) + 1
        histo.close()
    else:
        histo = open('histo.txt','a')
        histo.write("nb_exp alpha nu    n        T")
        nb_exp = 1
        histo.close()
    histo = open('histo.txt','a')
    histo.write('\n{:6d} {:1.3f} {:1.3f} '.format(nb_exp, alpha, nu) + '{:1.2e}'.format(n))
    histo.close()
    folder = '{:6d}_{:1.3f}_{:1.3f}_{}'.format(nb_exp, alpha, nu, n)
    i = 0
    for i in range(len(folder)):
        if folder[i] == ' ': folder = folder[:i] + '0' + folder[i+1:]
    os.mkdir(folder), os.chdir(folder)

R = 2 * np.log(n/nu)

V, V_cart = genere_V(n, alpha, R)
E, N = genere_E_N(V, n, R)
connected_compos = genere_connected_compos(N)

plt.figure(figsize=(16, 9)), plt.axis('equal')
if plot_all_graph: plot_graph_with_E(V_cart, E, R)
else: plot_compo_with_E(V_cart, E, connected_compos, R, num=0, vertices=True, edges=True, colV='b', colE='gray')
plot_circles_and_center(R)

T, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t = push_pull_with_plot(V, V_cart, N, connected_compos, n, alpha, nu, R, first_informed=-1, break_time=d_time, save=save)

plt.figure(figsize=(16, 9)), plt.title("(alpha, nu, n) = ({}, {}, {})".format(alpha, nu, n))
plot_stat_push_pull(T, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t)
if save:
    plt.savefig('graphic.eps', format='eps', bbox_inches='tight', dpi=1200)

    os.chdir(dir)
    histo = open('histo.txt','a')
    histo.write(' {:1.2e}'.format(T))
    histo.close()