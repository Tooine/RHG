################################################################################
#####                           RHG -- FUNCTIONS                           #####
################################################################################

import numpy as np
import numpy.random as rd
import pylab as plt
import time

########## BASIC FUNCTIONS ##########

def F(r, alpha, R):
    """ cumulative distribution function of the radial coordinate """
    
    if r <= 0: return 0
    if r >= R: return 1
    return (np.cosh(alpha * r) - 1) / (np.cosh(alpha * R) - 1)

def inverse_F(u, alpha, R, epsilon=10**-8):
    """ dichotomic research of t such that F(t) = u """
    
    if u == 0: return 0
    if u == 1: return R
    a, b = 0, R
    while b-a > epsilon:
        t = (a+b) / 2
        if F(t, alpha, R) > u: b = t
        else : a = t
    return (a+b) / 2

def cosh_dh(V1, V2):
    r_1, theta_1 = V1
    r_2, theta_2 = V2
    return np.cosh(r_1) * np.cosh(r_2) - np.sinh(r_1) * np.sinh(r_2) * np.cos(theta_1 - theta_2)

def d_theta_max(r_1, r_2, R):
    """ return the theta_max between two points at fixed radius for being neighbors """
    
    if r_1+r_2 < R: return np.pi
    return np.arccos((np.cosh(r_1) * np.cosh(r_2) - np.cosh(R)) / (np.sinh(r_1) * np.sinh(r_2)))

def search_band(r, limits):
    """ search b such that limits[b] <= r < limits[b+1]"""
    b = 1
    while limits[b] <= r: b += 1
    return b - 1

def theta_mod_2pi(theta):
    theta = theta % (2*np.pi)
    if theta > np.pi: theta -= 2*np.pi
    return theta

def neighbor_choice(N, i):
    """ random choice of a neighbor of i """
    return N[i][rd.randint(0, len(N[i]))]

########## GRAPH GENERATOR ##########

# greedy algorithm

def genere_V(n, alpha, R):
    """ generate n vertices of a RHG(n, alpha, nu) and return theirs
    polar coordinates in V and cartesian coordinates in V_cart """
    
    V, V_cart = np.zeros((n, 2)), np.zeros((n, 2))
    V[:, 1] = 2*np.pi * rd.rand(n)
    U = rd.rand(n)
    for i in range(n):
        V[i, 0] = inverse_F(U[i], alpha, R)
        V_cart[i] = [V[i, 0] * np.cos(V[i, 1]), V[i, 0] * np.sin(V[i, 1])]
    return V, V_cart

def genere_E(V, n, R):
    """ giving V, generate the edge set of the graph """
    
    E = []
    threshold = np.cosh(R)
    for i in range(n):
        for j in range(i+1, n):
            if cosh_dh(V[i], V[j]) <= threshold: E.append((i, j))
    return E

def genere_N(V, n, R):
    """ giving V, generate the list of neighbors of each point """
    
    N = []
    for i in range(n): N.append([])
    threshold = np.cosh(R)
    for i in range(n):
        for j in range(i+1, n):
            if cosh_dh(V[i], V[j]) <= threshold:
                N[i].append(j)
                N[j].append(i)
    return N

def genere_E_N(V, n, R):
    """ giving V, generate the list of neighbors of each point and the edge set """
    
    E, N = [], []
    for i in range(n): N.append([])
    threshold = np.cosh(R)
    for i in range(n):
        for j in range(i+1, n):
            if cosh_dh(V[i], V[j]) <= threshold:
                N[i].append(j)
                N[j].append(i)
                E.append((i, j))
    return E, N

# fast generator

def genere_G(n, alpha, nu, beta = 0.9):
    """ fast generator of G """
    
    R = 2 * np.log(n/nu)
    nbBands = max(2, int(beta * R) + 1)
    c = R/2/(nbBands - 1)
    limits = [0]
    for i in range(nbBands): limits.append(R/2 + i*c)
    
    bands_V, bands_R = [], []
    for i in range(len(limits)-1): bands_V.append([]), bands_R.append([])
    V, V_cart, E = [], [], []
    
    lst_theta = 2*np.pi * rd.rand(n)
    for i in range(n):
        r = inverse_F(rd.rand(), alpha, R)
        b = search_band(r, limits)
        theta = lst_theta[i]
        bands_V[b].append([i, r, theta])
        V.append([r, theta])
        V_cart.append([r * np.cos(theta), r * np.sin(theta)])
        for band in range(max(1, b), nbBands):
            theta_min_req = theta - d_theta_max(r, limits[band], R)
            theta_max_req = theta + d_theta_max(r, limits[band], R)
            if theta_min_req < 0:
                bands_R[band].append([i, b, theta_min_req+2*np.pi, 2*np.pi])
                bands_R[band].append([i, b, 0, theta_max_req])
            elif theta_max_req > 2*np.pi:
                bands_R[band].append([i, b, theta_min_req, 2*np.pi])
                bands_R[band].append([i, b, 0, theta_max_req - 2*np.pi])
            else: bands_R[band].append([i, b, theta_min_req, theta_max_req])
    V, V_cart = np.asarray(V), np.asarray(V_cart)
    
    for (i, r_i, theta_i) in bands_V[0]:
        for (j, r_j, theta_j) in bands_V[0]:
            if i < j: E.append((i, j))
    
    for band in range(1, nbBands):
        bands_V[band].sort(key=lambda point: point[2])
        bands_R[band].sort(key=lambda req: req[2])
        candidates_R = []
        for (i_pt, r_pt, theta_pt) in bands_V[band]:
            new_E = []
            for req in candidates_R:
                if theta_pt > req[3]: candidates_R.remove(req)
            test_new_req = True
            while test_new_req and len(bands_R[band]) > 0:
                req = bands_R[band][0]
                if req[2] <= theta_pt:
                    bands_R[band] = bands_R[band][1:]
                    if theta_pt <= req[3]: candidates_R.append(req)
                else: test_new_req = False
            for (i_req, b_req, theta_min_req, theta_max_req) in candidates_R:
                r_req, theta_req = V[i_req]
                candidate = (min(i_req, i_pt), max(i_req, i_pt))
                if cosh_dh([r_req, theta_req], [r_pt, theta_pt]) <= np.cosh(R) and candidate not in new_E and i_req != i_pt and ((b_req == band and i_req < i_pt) or b_req != band): new_E.append(candidate)
            E += new_E
    
    return V, V_cart, E

def genere_N_from_E(n, E):
    N = []
    for i in range(n): N.append([])
    for (i, j) in E: N[i].append(j), N[j].append(i)    
    return N

# layer graphs

def genere_layer(i, n, alpha, R):
    """ create points on layer i """
    
    nb_points = int(n * np.exp(-alpha*(R-i)) * (1 - np.exp(-alpha)))
    if nb_points == 0 or i >= R: return [], [], 0   # crée erreur dans genere_N_layers() si i >= R
    V_layer, V_cart_layer = [], []
    theta = 2*np.pi/nb_points
    for j in range(nb_points):
        theta_j = j * theta
        V_layer.append([i, theta_j])
        V_cart_layer.append([i * np.cos(theta_j), i * np.sin(theta_j)])
    return V_layer, V_cart_layer, nb_points

def genere_V_layers(radius_layers, n, alpha, R):
    """ generate vertices of a graph at given radii. At each radius, the number 
    of points generated is the expected number of points at the layer, they have
    a regular disposal compared to theta """ 
    
    V, V_cart = [], []
    nb_points_layers, nb_points_cumul_layers = [0], [0]
    for r in radius_layers:
        V_layer, V_cart_layer, nb_points = genere_layer(r, n, alpha, R)
        V += V_layer
        V_cart += V_cart_layer
        nb_points_layers.append(nb_points)
        nb_points_cumul_layers.append(nb_points_cumul_layers[-1] + nb_points)
    return np.asarray(V), np.asarray(V_cart), nb_points_layers, nb_points_cumul_layers

def genere_N_layers(V, n, alpha, R , nb_points_layers, nb_points_cumul_layers):        # à reprendre
    """ generate the list of neighbors of each point, in the case of a layer 
    disposal """
    
    N = []
    for a in range(len(V)): N.append([])
    nb_couches = len(nb_points_layers)-1
    for num_couche_1 in range(nb_couches):
        r_1, theta_1 = V[nb_points_cumul_layers[num_couche_1]+1]
        # voisins sur la couche r_1:
        d_theta = d_theta_max(r_1, r_1, R)
        for i in range(nb_points_layers[num_couche_1+1]):
            ind_theta_min = int(((i*theta_1 - d_theta) % (2*np.pi)) // theta_1)
            ind_theta_max = int(((i*theta_1 + d_theta) % (2*np.pi)) // theta_1)
            if d_theta == np.pi:
                liste_voisins_i = list(range(0, i)) + list(range(i+1, nb_points_layers[num_couche_1+1]))
            elif ind_theta_min <= ind_theta_max:
                liste_voisins_i = list(range(ind_theta_min+1, i)) + list(range(i+1, ind_theta_max+1))
            elif 2*i / nb_points_layers[num_couche_1+1] < 1:
                liste_voisins_i = list(range(0, i)) + list(range(i+1, ind_theta_max+1)) + list(range(ind_theta_min+1, nb_points_layers[num_couche_1+1]))
            else:
                liste_voisins_i = list(range(0, ind_theta_max+1)) + list(range(ind_theta_min+1, i)) + list(range(i+1,  nb_points_layers[num_couche_1+1]))
            N[nb_points_cumul_layers[num_couche_1] + i] += list(nb_points_cumul_layers[num_couche_1] + np.asarray(liste_voisins_i))
        
        # voisins sur les couches supérieures:
        for num_couche_2 in range(num_couche_1+1, len(nb_points_cumul_layers)-1):
            r_2, theta_2 = V[nb_points_cumul_layers[num_couche_2]+1]
            d_theta = d_theta_max(r_1, r_2, R)
            
            # ajout des voisins de la couche 1:
            for i in range(nb_points_layers[num_couche_1+1]):
                ind_theta_min = int(((i*theta_1 - d_theta) % (2*np.pi)) // theta_2)
                ind_theta_max = int(((i*theta_1 + d_theta) % (2*np.pi)) // theta_2)
                if ind_theta_min <= ind_theta_max:
                    liste_voisins_i = list(range(ind_theta_min+1, ind_theta_max+1))
                else:
                    liste_voisins_i = list(range(0, ind_theta_max+1)) + list(range(ind_theta_min+1, nb_points_layers[num_couche_2+1]))
                N[nb_points_cumul_layers[num_couche_1] + i] += list(nb_points_cumul_layers[num_couche_2] + np.asarray(liste_voisins_i))
            
            # ajout des voisins de la couche 2:
            for j in range(nb_points_layers[num_couche_2+1]):
                #ind_theta_min = int(((j*theta_2 - d_theta) % (2*np.pi)) // theta_1 + 1)
                ind_theta_min = int(((j*theta_2 - d_theta) % (2*np.pi)) // theta_1)
                ind_theta_max = int(((j*theta_2 + d_theta) % (2*np.pi)) // theta_1)
                if ind_theta_min <= ind_theta_max:
                    liste_voisins_j = list(range(ind_theta_min+1, ind_theta_max+1))
                else:
                    liste_voisins_j = list(range(0, ind_theta_max+1)) + list(range(ind_theta_min+1, nb_points_layers[num_couche_1+1]))
                N[nb_points_cumul_layers[num_couche_2] + j] += list(nb_points_cumul_layers[num_couche_1] + np.asarray(liste_voisins_j))

        expected_degree = int(2 * alpha / (np.pi * (alpha - 1/2)) * n * np.exp(-r_1/2)) + 1
        for i in range(nb_points_layers[num_couche_1+1]):
            while len(N[nb_points_cumul_layers[num_couche_1] + i]) < expected_degree:
                N[nb_points_cumul_layers[num_couche_1] + i] += [nb_points_cumul_layers[num_couche_1] + i]
    return N

########## HISTOGRAM ##########

def genere_hist_degrees(N, figure=True, alpha=0, i_moy=10, i_max=0):
    """ plot the histogram of the degrees of a RHG """
    
    L = []
    for i in range(len(N)): L.append(len(N[i]))
    hist_degrees = np.zeros(int(max(L))+1)
    for l in L: hist_degrees[l] += 1
    if alpha > 0:
        C = np.zeros(len(hist_degrees))
        C[0] = 1
        if alpha>=1/2:
            for i in range(1, len(hist_degrees)): C[i] = i**-(2*alpha+1)
        else:
            for i in range(1, len(hist_degrees)): C[i] = i**-(2)
        C *= hist_degrees[i_moy]/C[i_moy]
    if figure: 
        if alpha > 0: plt.plot(np.arange(len(hist_degrees)) + 0.5, C, color='r', linewidth=3)
        plt.hist(L, bins=int(max(L)))
        if i_max != 0: plt.xlim(0, i_max)
        plt.ylim(0, max(hist_degrees))
    return hist_degrees

########## CONNECTED COMPONENTS ##########

def genere_connected_compo(N, i):
    """ return the connected component containing i """
    
    connected_compo, pile = [], [i]
    while pile != []:
        j = pile[-1]
        pile = pile[:-1]
        connected_compo.append(j)
        for v in N[j]:
            if v not in connected_compo and v not in pile: pile.append(v)
    return connected_compo

def genere_connected_compos(N):
    """ return the list of connected components of the graph """
    
    flag_points = np.zeros(len(N), dtype=bool)  # 0 while i not in a compo, 1 if i in a compo
    connected_compos = []
    for i in range(len(N)):
        if not flag_points[i]:
            connected_compo = genere_connected_compo(N, i)
            connected_compo.sort()
            connected_compos.append(connected_compo)
            for c in connected_compo: flag_points[c] = True
    connected_compos.sort(key=len, reverse=True)
    return connected_compos

########## PLOTS ##########

def plot_circle(r, c=[0, 0], col='k', lin='-'):
    """ plot a circle of radius r and center c """
    theta = np.linspace(0, 2*np.pi, 100)
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    plt.plot(X, Y, color=col, linestyle=lin)

def plot_arc_circle(r, theta_min=0, theta_max=2*np.pi, c=[0, 0], col='k', lin='-', linewidth=3):
    theta = np.linspace(-theta_max, theta_max, 100)
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    plt.plot(X, Y, color=col, linestyle=lin, linewidth=linewidth)

def plot_circles_and_center(r, col='k'):
    """ plot origin and circles at r and r/2 """
    plt.plot([0], [0], color=col, marker='+', markersize=10)
    plot_circle(r, col=col)
    plot_circle(r/2, col=col, lin='--')

def plot_graph_with_E(V_cart, E, R, vertices=True, edges=True, colV='b', colE='gray'):
    """ plot RHG with V_cart and E """
    
    if edges:
        for (i, j) in E: plt.plot([V_cart[i, 0], V_cart[j, 0]], [V_cart[i, 1], V_cart[j, 1]], color=colE)
    if vertices: plt.plot(V_cart[:, 0], V_cart[:, 1], color=colV, marker='o', markersize=3, linestyle="None")

def plot_graph_with_N(V_cart, N, R, vertices=True, edges=True, colV='b', colN='gray'):
    """ plot RHG with V_cart and N """
    
    if edges:
        for i in range(len(N)):
            for j in N[i]:
                if i < j: plt.plot([V_cart[i, 0], V_cart[j, 0]], [V_cart[i, 1], V_cart[j, 1]], color=colN)
    if vertices: plt.plot(V_cart[:, 0], V_cart[:, 1], color=colV, marker='o', markersize=3, linestyle="None")

def plot_compo_with_E(V_cart, E, connected_compos, R, num=0, vertices=True, edges=True, colV='r', colE='m'):
    """ plot compo number *num* in size (starting by 0 for the giant component)
    with V_cart and E """
    connected_compo = connected_compos[num]
    
    if edges:
        for (i, j) in E:
            if i in connected_compo: plt.plot([V_cart[i, 0], V_cart[j, 0]], [V_cart[i, 1], V_cart[j, 1]], color=colE)
    if vertices:
        for i in connected_compo: plt.plot([V_cart[i, 0]], [V_cart[i, 1]], color=colV, marker='o', markersize=3, linestyle='None')

def plot_compo_with_N(V_cart, N, connected_compos, R, num=0, vertices=True, edges=True, colV='r', colN='m'):
    """ plot compo number *num* in size (starting by 0 for the center component)
    with V_cart and N """
    
    connected_compo = connected_compos[num]
    
    if edges:
        for i in connected_compo:
            for j in N[i]:
                if j > i: plt.plot([V_cart[i, 0], V_cart[j, 0]], [V_cart[i, 1], V_cart[j, 1]], color=colN)
    if vertices:
        for i in connected_compo: plt.plot([V_cart[i, 0]], [V_cart[i, 1]], color=colV, marker='o', markersize=3, linestyle='None')

def plot_ball(r, theta, R, colB='g', center=False, colC='y'):
    """ plot the ball of radius R centered at point (r, theta) """
    r_max = r + R
    range_R = np.linspace(0, r_max, r_max * 200 + 1)
    lst_theta_plus, lst_theta_moins = [], []
    for (i, r_test) in enumerate(range_R[:-1]):
        d_theta = d_theta_max(r, r_test, R)
        if d_theta == 0 or d_theta == np.pi:
            lst_theta_plus += [[]]
            lst_theta_moins += [[]]
            i_min = i
        else:
            lst_theta_plus.append(theta + d_theta)
            lst_theta_moins.append(theta - d_theta)
    lst_theta_plus.append(theta), lst_theta_moins.append(theta)
    lst_theta_plus[i_min], lst_theta_moins[i_min] = theta + np.pi, theta + np.pi
    
    x, y = r * np.cos(theta), r * np.sin(theta)
    lst_plus, lst_moins = [], []
    for (r_test, theta_plus, theta_moins) in zip(range_R, lst_theta_plus, lst_theta_moins):
        if not theta_plus == []:
            lst_plus.append([r_test * np.cos(theta_plus), r_test * np.sin(theta_plus)])
            lst_moins.append([r_test * np.cos(theta_moins), r_test * np.sin(theta_moins)])
    lst_plus = np.asarray(lst_plus)
    lst_moins = np.asarray(lst_moins)
    
    plt.plot(lst_plus[:, 0], lst_plus[:, 1], color=colB, linewidth=3)
    plt.plot(lst_moins[:, 0], lst_moins[:, 1], color=colB, linewidth=3)
    if center: plt.plot([x], [y], marker='o', color=colC, markersize=5)

def plot_ball_in_graph(V, V_cart, E, i, R, colB='g', colC='y', edges=True, colE='g'):
    r, theta = V[i]
    x, y = V_cart[i]
    plot_ball(r, theta, R, colB=colB, center=False, colC=colC)
    if edges:
        for (j, k) in E:
            if j == i: plt.plot([x, V_cart[k, 0]], [y, V_cart[k, 1]], color=colE)
            if k == i: plt.plot([V_cart[j, 0], x], [V_cart[j, 1], y], color=colE)
        for (j, k) in E:
            if j == i: plt.plot([V_cart[k, 0]], [V_cart[k, 1]], color=colE, marker='o', markersize=3, linestyle='None')
            if k == i: plt.plot([V_cart[j, 0]], [V_cart[j, 1]], color=colE, marker='o', markersize=3, linestyle='None')
    plt.plot([x], [y], marker='o', color=colC, markersize=5)

def plot_request(r_min, r_max, theta_min, theta_max, colR='r', linewidth=3):
    x_1, x_2, y_1, y_2 = r_min * np.cos(theta_min), r_max * np.cos(theta_min), r_min * np.sin(theta_min), r_max * np.sin(theta_min)
    plt.plot([x_1, x_2], [y_1, y_2], color='r', linewidth=linewidth)
    x_1, x_2, y_1, y_2 = r_min * np.cos(theta_max), r_max * np.cos(theta_max), r_min * np.sin(theta_max), r_max * np.sin(theta_max)
    plt.plot([x_1, x_2], [y_1, y_2], color='r', linewidth=linewidth)
    plot_arc_circle(r_min, theta_min, theta_max, c=[0, 0], col='r', linewidth=linewidth)
    plot_arc_circle(r_max, theta_min, theta_max, c=[0, 0], col='r', linewidth=linewidth)

########## PUSH & PULL ##########

def push_pull_with_plot(V, V_cart, N, connected_compos, n, alpha, nu, R, first_informed=-1, break_time=1, save=False, name_fig="", legend=True):
    """ simulation of a push & pull propagation in the central component
    plot the propagation at each time
    if first_informed = -1 the informer is chosen uniformly in the giant component """
    
    t = 0
    compo = connected_compos[0]
    states = np.zeros(n, dtype=int)
    if first_informed == -1: first_informed = compo[rd.randint(0, len(compo))]
    states[first_informed] = 4
    informed_at_t = [[first_informed]]
    nb_push_at_t, nb_pull_at_t, nb_both_at_t = [0], [0], [0]
    nb_infected = 1
    COL, SIZE = ['b', 'r', 'g', 'k', 'y'], [3, 5, 5, 5, 7]
    
    plt.title("(alpha, nu, n) = ({}, {}, {}) - t = {}, {}/{} informed".format(alpha, nu, n, t, nb_infected, len(compo)))
    
    plt.plot([V_cart[first_informed][0]], [V_cart[first_informed][1]], color=COL[4], marker='o', markersize=SIZE[4], linestyle='None')
    
    plt.plot([], [], color=COL[4], marker='o', markersize=5, linestyle='None', label="informer")
    plt.plot([], [], color=COL[1], marker='o', markersize=5, linestyle='None', label="received by push")
    plt.plot([], [], color=COL[2], marker='o', markersize=5, linestyle='None', label="received by pull")
    plt.plot([], [], color=COL[3], marker='o', markersize=5, linestyle='None', label="received by both")
    plt.plot([], [], color='tomato', label="transmit by push")
    plt.plot([], [], color='lawngreen', label="transmit by pull")
    plt.plot([], [], color=COL[3], label="transmit by both")
    if legend: plt.legend()
    if save: plt.savefig(name_fig + 't{}.eps'.format(t), format='eps', bbox_inches='tight', dpi=1200)
    plt.pause(break_time)
    
    while nb_infected != len(compo):
        t += 1
        informed_at_t += [[]]
        new_infected_by_push, new_infected_by_pull = np.zeros(n, dtype=list), np.zeros(n, dtype=list)
        for i in range(n): new_infected_by_push[i], new_infected_by_pull[i] = [], []
        
        for i in compo:
            j = neighbor_choice(N, i)
            if states[i] != 0 and states[j] == 0: new_infected_by_push[j].append(i)     # push
            if states[i] == 0 and states[j] != 0: new_infected_by_pull[i].append(j)     # pull
        
        nb_push, nb_pull, nb_both = 0, 0, 0
        for i in compo:
            if len(new_infected_by_push[i]) + len(new_infected_by_pull[i]) > 0:
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
        nb_push_at_t.append(nb_push_at_t[-1] + nb_push), nb_pull_at_t.append(nb_pull_at_t[-1] + nb_pull), nb_both_at_t.append(nb_both_at_t[-1] + nb_both)
        nb_infected += len(informed_at_t[-1])
        
        plt.title("(alpha, nu, n) = ({}, {}, {}) - t = {}, {}/{} informed".format(alpha, nu, n, t, nb_infected, len(compo)))
        for infected in informed_at_t[-1]:
            for i in new_infected_by_push[infected]:
                plt.plot([V_cart[infected, 0], V_cart[i, 0]], [V_cart[infected, 1], V_cart[i, 1]], color='tomato')
                plt.plot([V_cart[i, 0]], [V_cart[i, 1]], color=COL[states[i]], marker='o', markersize=SIZE[states[i]])
            for i in new_infected_by_pull[infected]:
                plt.plot([V_cart[infected, 0], V_cart[i, 0]], [V_cart[infected, 1], V_cart[i, 1]], color='lawngreen')
                plt.plot([V_cart[i, 0]], [V_cart[i, 1]], color=COL[states[i]], marker='o', markersize=SIZE[states[i]])
            plt.plot([V_cart[infected, 0]], [V_cart[infected,1]], color=COL[states[infected]], marker='o', markersize=SIZE[states[infected]], linestyle='None')
        
        if save: plt.savefig(name_fig + 't{}.eps'.format(t), format='eps', bbox_inches='tight', dpi=1200)
        plt.pause(break_time)
    
    for i in compo: plt.plot([V_cart[i, 0]], [V_cart[i, 1]], color=COL[states[i]], marker='o', markersize=SIZE[states[i]], linestyle='None')
    plt.plot([V_cart[first_informed][0]], [V_cart[first_informed][1]], color='yellow', marker='o', markersize=7, linestyle='None')
    if save: plt.savefig(name_fig + 't{}.eps'.format(t), format='eps', bbox_inches='tight', dpi=1200)
    
    return t, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t

def push_pull_without_plot(V, V_cart, N, connected_compos, n, alpha, nu, R, first_informed=-1):
    """ simulation of a push_pull propagation of information in the central component
    if first_informed = -1, the informer is chosen uniformly in the central component """
    
    t = 0
    compo = connected_compos[0]
    states = np.zeros(n, dtype=int)
    if first_informed == -1: first_informed = compo[rd.randint(0, len(compo))]
    states[first_informed] = 4
    informed_at_t = [[first_informed]]
    nb_push_at_t, nb_pull_at_t, nb_both_at_t = [0], [0], [0]
    nb_infected = 1
    
    while nb_infected != len(compo):
        t += 1
        informed_at_t += [[]]
        new_infected_by_push, new_infected_by_pull = np.zeros(n, dtype=list), np.zeros(n, dtype=list)
        for i in range(n): new_infected_by_push[i], new_infected_by_pull[i] = [], []
        
        for i in compo:
            j = neighbor_choice(N, i)
            if states[i] != 0 and states[j] == 0: new_infected_by_push[j].append(i)     # push
            if states[i] == 0 and states[j] != 0: new_infected_by_pull[i].append(j)     # pull
        
        nb_push, nb_pull, nb_both = 0, 0, 0
        for i in compo:
            if len(new_infected_by_push[i]) + len(new_infected_by_pull[i]) > 0:
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
        nb_push_at_t.append(nb_push_at_t[-1] + nb_push), nb_pull_at_t.append(nb_pull_at_t[-1] + nb_pull), nb_both_at_t.append(nb_both_at_t[-1] + nb_both)
        nb_infected += len(informed_at_t[-1])
        
    return t, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t

def push_pull_without_plot2(V, V_cart, N, connected_compos, n, alpha, nu, R, first_informed=-1):
    """ simulation of a push_pull propagation of information in the central component
    if first_informed = -1, the informer is chosen uniformly in the central component """
    
    t = 0
    compo = connected_compos[0]
    states = np.zeros(n, dtype=int)
    if first_informed == -1: first_informed = compo[rd.randint(0, len(compo))]
    states[first_informed] = 4
    informed_at_t = [[first_informed]]
    nb_push_at_t, nb_pull_at_t, nb_both_at_t = [0], [0], [0]
    nb_infected = 1
    infected_by_push, infected_by_pull = np.zeros(n, dtype=object), np.zeros(n, dtype=object)
    for i in range(n):
        infected_by_push[i], infected_by_pull[i] = [], []
    
    while nb_infected != len(compo):
        t += 1
        informed_at_t += [[]]
        new_infected_by_push, new_infected_by_pull = np.zeros(n, dtype=object), np.zeros(n, dtype=object)
        for i in range(n): new_infected_by_push[i], new_infected_by_pull[i] = [], []
        
        for i in compo:
            j = neighbor_choice(N, i)
            if states[i] != 0 and states[j] == 0:   # push
                new_infected_by_push[j].append(i)
                infected_by_push[j].append(i)
            if states[i] == 0 and states[j] != 0:   # pull
                new_infected_by_pull[i].append(j)
                infected_by_pull[i].append(j)
        
        nb_push, nb_pull, nb_both = 0, 0, 0
        for i in compo:
            if len(new_infected_by_push[i]) + len(new_infected_by_pull[i]) > 0:
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
    
    return t, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t, infected_by_push, infected_by_pull

def plot_stat_push_pull(T, informed_at_t, nb_push_at_t, nb_pull_at_t, nb_both_at_t):
    """ plot the graphic with some stats about the push and pull simulation """
    nb_informed_at_t = 1 + np.asarray(nb_push_at_t) + np.asarray(nb_pull_at_t) + np.asarray(nb_both_at_t)
    
    t = np.linspace(0, T, T+1)
    plt.xlabel("t"), plt.xlim(0, T)
    plt.plot(t, nb_informed_at_t, color='b', label="total informed")
    plt.plot(t, nb_push_at_t, color='r', label="informed by push")
    plt.plot(t, nb_pull_at_t, color='g', label="informed by pull")
    plt.plot(t, nb_both_at_t, color='k', label="informed by both")
    plt.legend(loc='best')

def plot_stat_push_pull2(radius_layers, R, V, informed_at_t, infected_by_push, infected_by_pull, nb_points_layers, nb_points_cumul_layers):
    
    T = len(informed_at_t)-1
    
    theta_ref = V[informed_at_t[0][0], 1]
    d_theta_max_m, d_theta_max_p, d_theta_max_layer_1_m, d_theta_max_layer_1_p, d_theta_max_layer_2_m, d_theta_max_layer_2_p = [0], [0], [0], [0], [0], [0]
    
    push_layer_1_by_layer_1, push_layer_1_by_layer_2, push_layer_2_by_layer_1, push_layer_2_by_layer_2 = [], [], [], []
    pull_layer_1_by_layer_1, pull_layer_1_by_layer_2, pull_layer_2_by_layer_1, pull_layer_2_by_layer_2 = [], [], [], []
    
    lim_layers = nb_points_layers[1]
    
    for (t, list) in enumerate(informed_at_t[1:]):
        record_layer_1_m, record_layer_2_m = d_theta_max_layer_1_m[-1], d_theta_max_layer_2_m[-1]
        record_layer_1_p, record_layer_2_p = d_theta_max_layer_1_p[-1], d_theta_max_layer_2_p[-1]
        for i in list:
            theta_i = V[i, 1]
            
            #courbes d_theta_max
            d_theta_i = theta_mod_2pi(theta_i - theta_ref)
            if d_theta_i < 0:
                if i < lim_layers:
                    record_layer_1_m = min(record_layer_1_m, d_theta_i)
                else:
                    record_layer_2_m = min(record_layer_2_m, d_theta_i)
            else:
                if i < lim_layers:
                    record_layer_1_p = max(record_layer_1_p, d_theta_i)
                else:
                    record_layer_2_p = max(record_layer_2_p, d_theta_i)
            
            # push/pull
            for j in infected_by_push[i]:
                theta_j = V[j, 1]
                d_theta_j = theta_mod_2pi(theta_j - theta_ref)
                if i < lim_layers:
                    if j < lim_layers: push_layer_1_by_layer_1.append([t, d_theta_j, d_theta_i])
                    else: push_layer_1_by_layer_2.append([t, d_theta_j, d_theta_i])
                else:
                    if j < lim_layers: push_layer_2_by_layer_1.append([t, d_theta_j, d_theta_i])
                    else: push_layer_2_by_layer_2.append([t, d_theta_j, d_theta_i])
            for j in infected_by_pull[i]:
                theta_j = V[j, 1]
                d_theta_j = theta_mod_2pi(theta_j - theta_ref)
                if i < lim_layers:
                    if j < lim_layers: pull_layer_1_by_layer_1.append([t, d_theta_j, d_theta_i])
                    else: pull_layer_1_by_layer_2.append([t, d_theta_j, d_theta_i])
                else:
                    if j < lim_layers: pull_layer_2_by_layer_1.append([t, d_theta_j, d_theta_i])
                    else: pull_layer_2_by_layer_2.append([t, d_theta_j, d_theta_i])
        
        d_theta_max_m.append(min(record_layer_1_m, record_layer_2_m))
        d_theta_max_p.append(max(record_layer_1_p, record_layer_2_p))
        d_theta_max_layer_1_m.append(record_layer_1_m)
        d_theta_max_layer_1_p.append(record_layer_1_p)
        d_theta_max_layer_2_m.append(record_layer_2_m)
        d_theta_max_layer_2_p.append(record_layer_2_p)
    
    push_layer_1_by_layer_1 = np.asarray(push_layer_1_by_layer_1)
    push_layer_1_by_layer_2 = np.asarray(push_layer_1_by_layer_2)
    push_layer_2_by_layer_1 = np.asarray(push_layer_2_by_layer_1)
    push_layer_2_by_layer_2 = np.asarray(push_layer_2_by_layer_2)
    pull_layer_1_by_layer_1 = np.asarray(pull_layer_1_by_layer_1)
    pull_layer_1_by_layer_2 = np.asarray(pull_layer_1_by_layer_2)
    pull_layer_2_by_layer_1 = np.asarray(pull_layer_2_by_layer_1)
    pull_layer_2_by_layer_2 = np.asarray(pull_layer_2_by_layer_2)
    theta_max_11 = d_theta_max(radius_layers[0], radius_layers[0], R)
    theta_max_12 = d_theta_max(radius_layers[0], radius_layers[1], R)
    theta_max_22 = d_theta_max(radius_layers[1], radius_layers[1], R)
    
    t = np.linspace(0, T, T+1)
    plt.subplot(231)
    plt.title("layer 1"), plt.axis([0, T, -np.pi, np.pi])
    plt.plot(t, d_theta_max_layer_1_m, color='k', label="d_theta_max_l1")
    plt.plot(t, d_theta_max_layer_1_p, color='k')
    plt.plot(t, d_theta_max_m, color='k', linestyle=':', label="d_theta_max")
    plt.plot(t, d_theta_max_p, color='k', linestyle=':')
    plt.legend(loc='best')
    
    plt.subplot(234)
    plt.title("layer 2"), plt.axis([0, T, -np.pi, np.pi])
    plt.plot(t, d_theta_max_layer_2_m, color='k', label="d_theta_max_l2")
    plt.plot(t, d_theta_max_layer_2_p, color='k')
    plt.plot(t, d_theta_max_m, color='k', linestyle=':', label="d_theta_max")
    plt.plot(t, d_theta_max_p, color='k', linestyle=':')
    plt.legend(loc='best')
    
    plt.subplot(232)
    plt.title("push and pull l1 -> l1"), plt.axis([0, T, -np.pi, np.pi])
    #plt.plot([0, T], [theta_max_11, theta_max_11], linestyle='--', color='gray')
    #plt.plot([0, T], [-theta_max_11, -theta_max_11], linestyle='--', color='gray')
    if len(push_layer_1_by_layer_1.shape) == 2:
        for (t, d_theta_j, d_theta_i) in push_layer_1_by_layer_1:
            if abs(d_theta_i-d_theta_j) <= theta_max_11:
                plt.plot([t, t], [d_theta_j, d_theta_i], color='k')
            else:
                plt.plot([t, t], [max(d_theta_j, d_theta_i), np.pi], color='k')
                plt.plot([t, t], [min(d_theta_j, d_theta_i), -np.pi], color='k')
        plt.plot(push_layer_1_by_layer_1[:, 0], push_layer_1_by_layer_1[:, 2], marker='+', color='r', linestyle='None', label="push")
    if len(pull_layer_1_by_layer_1.shape) == 2:
        for (t, d_theta_j, d_theta_i) in pull_layer_1_by_layer_1:
            if abs(d_theta_i-d_theta_j) <= theta_max_11:
                plt.plot([t, t], [d_theta_j, d_theta_i], color='k')
            else:
                plt.plot([t, t], [max(d_theta_j, d_theta_i), np.pi], color='k')
                plt.plot([t, t], [min(d_theta_j, d_theta_i), -np.pi], color='k')
        plt.plot(pull_layer_1_by_layer_1[:, 0], pull_layer_1_by_layer_1[:, 2], marker='+', color='g', linestyle='None', label="pull")
    plt.legend(loc='best')
    
    plt.subplot(233)
    plt.title("push and pull l2 -> l1"), plt.axis([0, T, -np.pi, np.pi])
    #plt.plot([0, T], [theta_max_12, theta_max_12], linestyle='--', color='gray')
    #plt.plot([0, T], [-theta_max_12, -theta_max_12], linestyle='--', color='gray')
    if len(push_layer_1_by_layer_2.shape) == 2:
        for (t, d_theta_j, d_theta_i) in push_layer_1_by_layer_2:
            if abs(d_theta_i-d_theta_j) <= theta_max_12:
                plt.plot([t, t], [d_theta_j, d_theta_i], color='k')
            else:
                plt.plot([t, t], [max(d_theta_j, d_theta_i), np.pi], color='k')
                plt.plot([t, t], [min(d_theta_j, d_theta_i), -np.pi], color='k')
        plt.plot(push_layer_1_by_layer_2[:, 0], push_layer_1_by_layer_2[:, 2], marker='+', color='r', linestyle='None', label="push")
    if len(pull_layer_1_by_layer_2.shape) == 2:
        for (t, d_theta_j, d_theta_i) in pull_layer_1_by_layer_2:
            if abs(d_theta_i-d_theta_j) <= theta_max_12:
                plt.plot([t, t], [d_theta_j, d_theta_i], color='k')
            else:
                plt.plot([t, t], [max(d_theta_j, d_theta_i), np.pi], color='k')
                plt.plot([t, t], [min(d_theta_j, d_theta_i), -np.pi], color='k')
        plt.plot(pull_layer_1_by_layer_2[:, 0], pull_layer_1_by_layer_2[:, 2], marker='+', color='g', linestyle='None', label="pull")
    plt.legend(loc='best')
    
    plt.subplot(235)
    plt.title("push and pull l2 -> l2"), plt.axis([0, T, -np.pi, np.pi])
    #plt.plot([0, T], [theta_max_22, theta_max_22], linestyle='--', color='gray')
    #plt.plot([0, T], [-theta_max_22, -theta_max_22], linestyle='--', color='gray')
    if len(push_layer_2_by_layer_2.shape) == 2:
        for (t, d_theta_j, d_theta_i) in push_layer_2_by_layer_2:
            if abs(d_theta_i-d_theta_j) <= theta_max_22:
                plt.plot([t, t], [d_theta_j, d_theta_i], color='k')
            else:
                plt.plot([t, t], [max(d_theta_j, d_theta_i), np.pi], color='k')
                plt.plot([t, t], [min(d_theta_j, d_theta_i), -np.pi], color='k')
        plt.plot(push_layer_2_by_layer_2[:, 0], push_layer_2_by_layer_2[:, 2], marker='+', color='r', linestyle='None', label="push")
    if len(pull_layer_2_by_layer_2.shape) == 2:
        for (t, d_theta_j, d_theta_i) in pull_layer_2_by_layer_2:
            if abs(d_theta_i-d_theta_j) <= theta_max_22:
                plt.plot([t, t], [d_theta_j, d_theta_i], color='k')
            else:
                plt.plot([t, t], [max(d_theta_j, d_theta_i), np.pi], color='k')
                plt.plot([t, t], [min(d_theta_j, d_theta_i), -np.pi], color='k')
        plt.plot(pull_layer_2_by_layer_2[:, 0], pull_layer_2_by_layer_2[:, 2], marker='+', color='g', linestyle='None', label="pull")
    plt.legend(loc='best')
    
    plt.subplot(236)
    plt.title("push and pull l1 -> l2"), plt.axis([0, T, -np.pi, np.pi])
    #plt.plot([0, T], [theta_max_12, theta_max_12], linestyle='--', color='gray')
    #plt.plot([0, T], [-theta_max_12, -theta_max_12], linestyle='--', color='gray')
    if len(push_layer_2_by_layer_1.shape) == 2:
        for (t, d_theta_j, d_theta_i) in push_layer_2_by_layer_1:
            if abs(d_theta_i-d_theta_j) <= theta_max_12:
                plt.plot([t, t], [d_theta_j, d_theta_i], color='k')
            else:
                plt.plot([t, t], [max(d_theta_j, d_theta_i), np.pi], color='k')
                plt.plot([t, t], [min(d_theta_j, d_theta_i), -np.pi], color='k')
        plt.plot(push_layer_2_by_layer_1[:, 0], push_layer_2_by_layer_1[:, 2], marker='+', color='r', linestyle='None', label="push")
    if len(pull_layer_2_by_layer_1.shape) == 2:
        for (t, d_theta_j, d_theta_i) in pull_layer_2_by_layer_1:
            if abs(d_theta_i-d_theta_j) <= theta_max_12:
                plt.plot([t, t], [d_theta_j, d_theta_i], color='k')
            else:
                plt.plot([t, t], [max(d_theta_j, d_theta_i), np.pi], color='k')
                plt.plot([t, t], [min(d_theta_j, d_theta_i), -np.pi], color='k')
        plt.plot(pull_layer_2_by_layer_1[:, 0], pull_layer_2_by_layer_1[:, 2], marker='+', color='g', linestyle='None', label="pull")
    plt.legend(loc='best')