pop_size = 200
num_GA_params = 10
size_bin_code = 7
mutRate = 0.5
K = 200


x = 0
y = 8
a = 0
v = 0

xf = 0
yf = 0
af = 0
vf = 0

tolerance = 0.1

import sys
import random
import numpy as np
import math
import scipy as sp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def checker(x1, y1):
    if (x<= -4 and y>3) or (-4<x<4 and y> -1) or (x>=4 and y>3):
        return 1
    else:
        return 0


def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutRate:
            individual[i] = int(not individual[i])
    return individual





def cross(gene1, gene2):
    child1 = []
    child2 = []
    #print("gene1: ", gene1)
    cross = random.randint(0,2*10*size_bin_code-1)
    child1 = gene1[0:cross] + gene2[cross:]
    child2 = gene2[0:cross] + gene1[cross:]
    #print(len(child2))
    if random.random()<mutRate:
        child1 = mutate(child1)
    if random.random()<mutRate:
        child2 = mutate(child2)
    return child1, child2



def populator(sorted_merged_list, sorted_individual, sorted_fitness, new_gen, mutProb,pop_size, limit):
    child1, child2 = cross(new_gen[0], new_gen[1])
    new_gen.append(child1)
    new_gen.append(child2)
    while(len(new_gen) + 2 <= pop_size):
        parent1, parent2 = random.choices(sorted_individual, weights = sorted_fitness, k=2)
        child1, child2 = cross(parent1, parent2)
        new_gen.append(child1)
        new_gen.append(child2)
    #print("new_gen_len ", len(new_gen))
    limit +=1
    if limit <2000:
        return ga_algo(new_gen, pop_size, size_bin_code, mutProb, limit)
    else:
        print("adwdad", sorted_merged_list[0])
        if  sorted_merged_list[0][1] == new_gen[0]:
            print("True")
        return new_gen[0]



def funct(individual, size_bin_code):
    i = 0
    x1 = x
    y1 = y
    v1 = v
    a1 = a
    xf1 = xf
    yf1 = yf
    vf1 = vf
    af1 = af
    #print(xf1)
    g_arr = []
    b_arr = []
    while(i<len(individual)):
        gamma = individual[i:i+ size_bin_code]
        beta = individual[i+size_bin_code:i+ size_bin_code+7]
        g = (((int("".join(str(k) for k in gamma), 2))/((2**size_bin_code) -1))*(1.048)) + (-0.524)
        b = (((int("".join(str(k) for k in beta), 2))/((2**size_bin_code) -1))*(10)) + (-5)
        x1 = x1 + v1*math.cos(a1)
        y1 = y1 + v1*math.sin(a1)
        v1 = v1 + b
        a1 = a1 + g
        i = i + size_bin_code + 7
        check = checker(x1, y1)
        if check== 0:
            return K, -1
    if check == 0:
        J = K
    else:
        J = math.sqrt((xf1-x1)**2 + (yf1-y1)**2 + (af1-a1)**2 + (vf1 - v1)**2)
    return J, 1


def ga_algo(population,pop_size, size_bin_code, mutProb, limit):
    J= []
    merged_list = []
    for individual in population:
        res, flag = funct(individual, size_bin_code)
        if flag == 1:
            J.append(res)
        merged_list.append((individual,res))
    #print(len(population))
    #print(len(J))
    #merged_list = [(population[i], J[i]) for i in range(0, pop_size)]
    sorted_merged_list = sorted(merged_list, key = lambda a: a[1])
    sorted_individual = [n for n,_ in sorted_merged_list]
    sorted_j = [m for _, m in sorted_merged_list]
    #print("sortedj:", sorted_j)
    minJ = sorted_j[0]
    new_gen = []
    min1 = sorted_individual[0]
    new_gen.append(min1)
    print("Generation ", limit, ": J = ", minJ)
    if(minJ< tolerance):
        res = funct(min1, size_bin_code)
        print("res", res)
        print(min1)
        return min1
    #sorted_j.pop(0)
    min2 = sorted_individual[1]
    new_gen.append(min2)
    sorted_fitness = []
    for k in sorted_j:
        sorted_fitness.append(1/(k+1))
    return populator(sorted_merged_list, sorted_individual[2:], sorted_fitness[2:], new_gen, mutProb, pop_size, limit)



def binary_to_gray(n):
    binary_str = bin(n)[2:].zfill(size_bin_code*2*10)
    gray_str = bin(n ^ (n >> 1))[2:].zfill(size_bin_code*2*10)
    return gray_str


def plot_sample_trj():
    fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')


def plot1(x_arr,y_arr):
    x = np.linspace(-20, 20, 1000)
    y = np.zeros_like(x)
    y[x <= -4] = 3
    y[(x > -4) & (x < 4)] = -1
    y[x >= 4] = 3

    # define the arrays for the new line
    fig, ax = plt.subplots()       # create a figure and axis object

    ax.plot(x, y, color='black')    # plot the x and y arrays on the axis
    ax.plot(x_arr, y_arr, color='green') # plot the new x and y arrays on the same axis
    ax.grid(True, linestyle='--')
    ax.set_xlim(-20, 20)           # set the x limits of the plot
    ax.set_ylim(-20, 20)             # set the y limits of the plot
    plt.show()

def main():
    population = []
    for i in range(pop_size):
        individual = []
        bin_string = ""
        for j in range(size_bin_code*2*10):
            bin_string = bin_string + str(random.randint(0,1))
        gray_string = binary_to_gray(int(bin_string,2))
        individual = [int(bit) for bit in gray_string]
        population.append(individual)
    vector = ga_algo(population,pop_size,7,0.5, 0)
    plot_sample_trj()
    print(vector)
    res= funct(vector, size_bin_code)
    print("Final: ", res)
    x_arr = []
    y_arr = []
    v_arr = []
    a_arr = []
    i = 0
    x1 = x
    y1 = y
    a1 = a
    v1 = v
    g_arr = []
    b_arr = []
    while(i<len(vector)):
        gamma = vector[i:i+ size_bin_code]
        beta = vector[i+size_bin_code:i+ size_bin_code+ size_bin_code]
        g = (((int("".join(str(k) for k in gamma), 2))/((2**size_bin_code) -1))*(1.048)) + (-0.524)
        b = (((int("".join(str(k) for k in beta), 2))/((2**size_bin_code) -1))*(10)) + (-5)
        g_arr.append(g)
        b_arr.append(b)
        i = i + size_bin_code + size_bin_code
    tg = np.linspace(-0.524, 0.524,100)
    tb = np.linspace(-5,5,100)
    t = np.linspace(0, len(g_arr) - 1, len(g_arr))
    #print(t)
    # Perform cubic spline interpolation on g_arr and b_arr
    #g_spline = CubicSpline(np.arange(len(g_arr)), g_arr)
    g_spline = CubicSpline(t, g_arr)
    b_spline = CubicSpline(t, b_arr)
    t1 = np.linspace(0, len(g_arr)-1, 100)
    #b_spline = CubicSpline(np.linspace(0,1,len(g_arr)), b_arr)
    #print(g_spline)
    # Evaluate the splines on the new time grid to obtain the high-resolution control histories
    g_highres = g_spline(t1)
    b_highres = b_spline(t1)
    print(g_highres)
    print(b_highres)
    i = 0
    while i<len(g_highres):
        #print("wadawdwadw")
        x1 = x1 + v1*math.cos(a1)*0.1
        y1 = y1 + v1*math.sin(a1)*0.1
        v1 = v1 + b_highres[i]*0.1
        a1 = a1 + g_highres[i]*0.1
        i = i + 1
        x_arr.append(x1)
        y_arr.append(y1)
        v_arr.append(v1)
        a_arr.append(a1)
    print(x_arr)
    print()
    print("Final state values:")
    print("x_f = ", x1)
    print("y_f = ", y1)
    print("v1 = ", v1)
    print("a1 = ", a1)
    print("J:", math.sqrt((xf-x1)**2 + (yf-y1)**2 + (af-a1)**2 + (vf - v1)**2))
    plot1(x_arr,y_arr)
    """fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    #plt.axhline(y=0.5, color='r', linestyle='-')
    plt.show()"""
    """plt.plot(x_values, y_values)"""

    # plot the checker function conditions as lines
    #plt.plot([x1, x2], [y1, y2], linestyle='--', color='r')





if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()



"""
if random.random() < mutRate:
    for j in range(len(i)):
        if i[j] == 1:
            i[j] = 0
        else:
            i[j] = 1"""