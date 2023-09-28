"""
MAITREYA BHUPESH KOCHAREKAR
mk1651

"""

pop_size = 200
num_GA_params = 10
size_bin_code = 7
mutRate = 0.005
K = 200

J1 = float('inf')
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
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def checker(x1, y1):
    if (x1<= -4  and y1>3) or (-4<x1<4 and y1> -1) or (x1>=4 and y1>3):
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
    cross = random.randint(1,2*10*size_bin_code-1)
    child1 = gene1[0:cross] + gene2[cross:]
    child2 = gene2[0:cross] + gene1[cross:]
    #print(len(child2))
    #if random.random()<mutRate:
    child1 = mutate(child1)
    #if random.random()<mutRate:
    child2 = mutate(child2)
    return child1, child2



def populator(sorted_merged_list, sorted_individual, sorted_fitness, new_gen, mutProb,pop_size, limit, x_arr, y_arr,v_arr,a_arr, af1, vf1, b_arr,g_arr, controls):
    child1, child2 = cross(new_gen[0], new_gen[1])
    new_gen.append(child1)
    new_gen.append(child2)
    while(len(new_gen) + 2 <= pop_size):
        parent1, parent2 = random.choices(sorted_individual, weights = sorted_fitness, k=2)
        child1, child2 = cross(parent1, parent2)
        new_gen.append(child1)
        new_gen.append(child2)
    limit +=1
    if limit <1200:
        return ga_algo(new_gen, pop_size, size_bin_code, mutProb, limit)
    else:
        if  sorted_merged_list[0][1] == new_gen[0]:
            print("True")
        return new_gen[0], x_arr, y_arr,v_arr, a_arr, af1, vf1, b_arr, g_arr, controls



def funct(individual, size_bin_code, J1):
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
    x_arr = []
    y_arr = []
    v_arr = []
    a_arr = []
    g_arr = []
    b_arr = []
    controls = []
    while(i<len(individual)):
        gamma = individual[i:i+ size_bin_code]
        beta = individual[i+size_bin_code:i+ size_bin_code+ size_bin_code]
        g = (((int("".join(str(k) for k in gamma), 2))/((2**size_bin_code) -1))*(1.048)) + (-0.524)
        b = (((int("".join(str(k) for k in beta), 2))/((2**size_bin_code) -1))*(10)) + (-5)
        g_arr.append(g)
        b_arr.append(b)
        controls.append(g)
        controls.append(b)
        i = i + size_bin_code + size_bin_code
    t = np.linspace(0, len(g_arr) - 1, len(g_arr))
    g_spline = CubicSpline(t, g_arr)
    b_spline = CubicSpline(t, b_arr)
    t1 = np.linspace(0, len(g_arr)-1, 100)
    g_highres = g_spline(t1)
    b_highres = b_spline(t1)
    J = 0
    i = 0
    flag1 = 0
    while i<len(g_highres):
        x1 = x1 + v1*math.cos(a1)*0.1
        y1 = y1 + v1*math.sin(a1)*0.1
        v1 = v1 + b_highres[i]*0.1
        a1 = a1 + g_highres[i]*0.1
        x_arr.append(x1)
        y_arr.append(y1)
        v_arr.append(v1)
        a_arr.append(a1)
        i = i + 1     # size_bin_code + 7
        check = checker(x1, y1)
        if check== 0:
            return 200, x_arr, y_arr,v_arr,a_arr, J1, flag1, a1, v1, b_highres, g_highres, controls
    J = math.sqrt((xf1-x1)**2 + (yf1-y1)**2 + (af1-a1)**2 + (vf1 - v1)**2)
    if J<J1:
        J1 = J
        flag1 = 1
    return J, x_arr, y_arr,v_arr, a_arr, J1, flag1, a1, v1, b_highres, g_highres, controls



def ga_algo(population,pop_size, size_bin_code, mutProb, limit):
    J= []
    merged_list = []
    x_arr = []
    y_arr = []
    v_arr = []
    a_arr = []
    J1 = float('inf')
    af1 = 0
    vf1 = 0
    b_arr = []
    g_arr = []
    controls = []
    for individual in population:
        res, arr1, arr2, arr3, arr4, J1, flag1, a1, v1, beta_arr, gamma_arr, arr5 = funct(individual, size_bin_code, J1)
        J.append(res)
        merged_list.append((individual,res))
        if flag1 == 1:
            x_arr = arr1
            y_arr = arr2
            v_arr = arr3
            a_arr = arr4
            af1 = a1
            vf1 = v1
            b_arr = beta_arr
            g_arr = gamma_arr
            controls = arr5
    sorted_merged_list = sorted(merged_list, key = lambda a: a[1])
    sorted_individual = [n for n,_ in sorted_merged_list]
    sorted_j = [m for _, m in sorted_merged_list]
    minJ = sorted_j[0]
    new_gen = []
    min1 = sorted_individual[0]
    new_gen.append(min1)
    print("Generation ", limit, ": J = ", minJ)
    if(minJ< tolerance):
        res = funct(min1, size_bin_code, J1)
        return min1, x_arr, y_arr,v_arr, a_arr, af1,vf1,b_arr, g_arr, controls
    min2 = sorted_individual[1]
    new_gen.append(min2)
    sorted_fitness = []
    for k in sorted_j:
        sorted_fitness.append(1/(k+1))
    return populator(sorted_merged_list, sorted_individual[2:], sorted_fitness[2:], new_gen, mutProb, pop_size, limit, x_arr, y_arr,v_arr, a_arr, af1, vf1,b_arr, g_arr, controls)



def binary_to_gray(n):
    l = len(n)
    n = int(n,2)
    gray_str = bin(n ^ (n >> 1))[2:].zfill(l)
    return gray_str


def plot_sample_trj():
    fig, ax = plt.subplots()
    ax.grid(True, linestyle='--')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

def main():
    population = []
    for i in range(pop_size):
        individual = []
        bin_string = ""
        for j in range(size_bin_code*2*10):
            bin_string = bin_string + str(random.randint(0,1))
        gray_string = binary_to_gray(bin_string)
        individual = [int(bit) for bit in gray_string]
        population.append(individual)
    vector, x_arr, y_arr, v_arr, a_arr,af1, vf1, b_arr, g_arr, controls = ga_algo(population,pop_size,7,0.5, 0)
    print()
    print("Final state values:")
    print("x_f = ", x_arr[-1])
    print("y_f = ", y_arr[-1])
    print("alpha_f = ", af1)
    print("v_f = ", vf1)
    with open('controls.dat', 'w') as f:
        for element in controls:
            f.write(str(element) + '\n')
    #print("adawdwa", len(x_arr))
    plot1(x_arr,y_arr)
    plot2(x_arr)
    plot3(y_arr)
    plot4(a_arr)
    plot5(v_arr)
    plot6(b_arr)
    plot7(g_arr)



def plot1(x_arr,y_arr):
    x = np.linspace(-20, 20, 1000)
    y = np.zeros_like(x)
    y[x <= -4] = 3
    y[(x > -4) & (x < 4)] = -1
    y[x >= 4] = 3

    fig, ax = plt.subplots()

    ax.plot(x, y, color='black')
    ax.plot(x_arr, y_arr, color='green')
    ax.grid(True, linestyle='--')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    plt.xlabel("x (ft)")
    plt.ylabel("y (ft)")
    plt.title("State trajectory with obstacles")
    plt.show()

def plot2(x_arr):
    y = x_arr
    x = np.linspace(0,10,100)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.grid(True, linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("x (ft)")
    plt.title("x state histories")
    plt.show()

def plot3(y_arr):
    y = y_arr
    x = np.linspace(0,10,100)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.grid(True, linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("y (ft)")
    plt.title("y state histories")
    plt.show()

def plot4(a_arr):
    y = a_arr
    x = np.linspace(0,10,100)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.grid(True, linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("a (rad)")
    plt.title("alpha state histories")
    plt.show()

def plot5(v_arr):
    y = v_arr
    x = np.linspace(0,10,100)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.grid(True, linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("v (ft/s)")
    plt.title("velocity state histories")
    plt.show()


def plot6(b_arr):
    y = b_arr
    x = np.linspace(0,10,100)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.grid(True, linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("b (ft/s^2)")
    plt.title("beta state histories")
    plt.show()

def plot7(g_arr):
    y = g_arr
    x = np.linspace(0,10,100)
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.grid(True, linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("g (rad/s)")
    plt.title("gamma state histories")
    plt.show()


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()

