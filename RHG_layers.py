################################################################################
#####                      RHG LAYERS -- SIMULATIONS                       #####
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

#dir = '/Users/abarrier/Documents/Scolaire/ENS/M1/Stage/LaTeX'
#os.chdir(dir)

##########                 simple generation and plot                 ##########

# parameters

alpha = 0.85
nu = 2
n = 10000
#
R = 2 * np.log(n/nu)
#
C = 0.5

radius_layers = range(1, int(R) + 1)
#radius_layers = [R/2 + C, R/2 + np.log(np.log(n))]
radius_layers = [8, 10, 12, 14]

# program

radius_layers_round = []
for r in radius_layers: radius_layers_round.append(round(r, 2))

V, V_cart, nb_points_layers, nb_points_cumules_layers = genere_V_layers(radius_layers, n, alpha, R)
n_real = len(V)
N = genere_N_layers(V, n, alpha, R, nb_points_layers, nb_points_cumules_layers)
connected_compos = genere_connected_compos(N)

plt.figure(figsize=(9, 9)), plt.axis('equal'), plt.title("(alpha, nu, n, n_real) = ({:.2f}, {:.2f}, {}, {}), radius = {}".format(alpha, nu, n, n_real, radius_layers_round))
plot_graph_with_N(V_cart, N, R)
plot_compo_with_N(V_cart, N, connected_compos, R)
plot_circles_and_center(R)

plt.figure(figsize=(16, 9)), plt.title("Degree distribution")
hist_degrees = genere_hist_degrees(N)

##########             push and pull simulation with save             ##########

# parameters

alpha = 0.7
nu = 1
n = 100000
#   
R = 2 * np.log(n/nu)
#
radius_layers = [R/2 + 0.5, R/2 + np.log(np.log(n))]
radius_text = "[R/2 + 0.5, R/2 + log(log(n))]"

infected_vertice = -1  # -1 : choose a random vertice in the main component
plot_all_graph = False
plot_all_figures = False
d_time = 0.1
save = True

# program

radius_layers_round = []
for r in radius_layers: radius_layers_round.append(round(r, 2))

if save:
    dir = main_folder + '/simulations/2_layers_push_pull'
    if not os.path.exists(dir): os.makedirs(dir)
    os.chdir(dir)
    
    if os.path.isfile(dir + '/histo.txt'):
        histo = open('histo.txt', 'r')
        nb_exp = int(histo.read().split('\n')[-1][:6]) + 1
        histo.close()
    else:
        histo = open('histo.txt','a')
        histo.write("nb_exp alpha nu    n        T        radius")
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

V, V_cart, nb_points_layers, nb_points_cumules_layers = genere_V_layers(radius_layers, n, alpha, R)
n_real = len(V)
N = genere_N_layers(V, n, alpha, R, nb_points_layers, nb_points_cumules_layers)
connected_compos = genere_connected_compos(N)

if plot_all_figures:
    plt.figure(figsize=(16, 9)), plt.axis('equal')
    if plot_all_graph: plot_graph_with_N(V_cart, N, R)
    else: plot_compo_with_N(V_cart, N, connected_compos, R, num=0, vertices=True, edges=True, colV='b', colN='gray')
    plot_circles_and_center(R)

if plot_all_figures:
    T, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t = push_pull_with_plot(V, V_cart, N, connected_compos, n, alpha, nu, R, first_informed=-1, break_time=d_time, save=save)
else:
    T, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t, infected_by_push, infected_by_pull = push_pull_without_plot2(V, V_cart, N, connected_compos, n, alpha, nu, R, first_informed=-1)

plt.figure(figsize=(16, 9)), plt.title("(alpha, nu, n, n_real) = ({}, {}, {}, {})".format(alpha, nu, n, n_real))
plot_stat_push_pull(T, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t)
if save:
    plt.savefig('graphic.eps', format='eps', bbox_inches='tight', dpi=1200)

plt.figure(figsize=(16, 9)), plt.axis('equal'), plt.title("(alpha, nu, n, n_real) = ({:.2f}, {:.2f}, {}, {}), radius = {}".format(alpha, nu, n, n_real, radius_layers_round))
plot_stat_push_pull2(radius_layers, R, V, informed_at_t, infected_by_push, infected_by_pull, nb_points_layers, nb_points_cumules_layers)
if save:
    plt.savefig('propagation.eps', format='eps', bbox_inches='tight', dpi=1200)

    os.chdir(dir)
    histo = open('histo.txt','a')
    histo.write(' {:1.2e} {}'.format(T, radius_layers_round))
    histo.close()