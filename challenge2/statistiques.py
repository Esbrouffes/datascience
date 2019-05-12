import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pickle
from collections import Counter
from networkx.algorithms import community
import community


college={}
location={}
employer={}
# The dictionaries are loaded as dictionaries from the disk (see pickle in Python doc)
with open('mediumCollege_60percent_of_empty_profile.pickle', 'rb') as handle:
    college = pickle.load(handle)
with open('mediumLocation_60percent_of_empty_profile.pickle', 'rb') as handle:
    location = pickle.load(handle)
with open('mediumEmployer_60percent_of_empty_profile.pickle', 'rb') as handle:
    employer = pickle.load(handle)


def properties(g):
    """
    Computes simple and classic graph metrics.

    Parameters
    ----------
    g : graph
       A networkx graph
    """
    # networkx short summary of information for the graph g
    print(nx.info(g))
    
    # Draw the degree distribution. Powerlow distribution for a real (complex) network
    plt.figure(num=None)
    fig = plt.figure(1)
    degree_sequence=[d for n, d in g.degree()] # degree sequence
   # print("Degree sequence %s" % degree_sequence)
    plt.hist(degree_sequence, bins='auto')  
    plt.title("powerlaw degree distribution")
    plt.ylabel("# nodes")
    plt.xlabel("degree")
   # plt.show()
    pylab.close()
    del fig
 
    precomputed_eccentricity = nx.eccentricity(g) # costly step, we save time here!
    print("Graph density %f" % nx.density(g))
    print("Diameter (maximum eccentricity): %d" % nx.diameter(g,precomputed_eccentricity))
    print("Radius (minimum eccentricity): %d" % nx.radius(g,precomputed_eccentricity)) #The radius is the minimum eccentricity.
    print("Mean eccentricity (eccentricity(v) = the maximum distance from v to all other nodes): %s" % np.mean(list(precomputed_eccentricity.values())))
    print("Center is composed of %d nodes (nodes with eccentricity equal to radius)" % len(nx.center(g, precomputed_eccentricity)))
    print("Periphery is composed of %d nodes (nodes with eccentricity equal to the diameter)" % len(nx.periphery(g,precomputed_eccentricity)))
    print("Mean clustering coefficient %f" % np.mean(list(nx.clustering(g).values())))
    total_triangles=sum(nx.triangles(g).values())/3    
    print("Total number of triangles in graph: %d" % total_triangles)



    #### Computing the homophily 
def homophily(G):
    similar_neighbors_E=0
    similar_neighbors_L=0
    similar_neighbors_C=0
    total_number_neighbors=0 
    for n in G.nodes():
        for nbr in G.neighbors(n):
            total_number_neighbors+=1
            if n in employer and nbr in employer:
                if len([val for val in employer[n] if val in employer[nbr]]) > 0:
                    similar_neighbors_E+=1
            if n in college and nbr in college:
                if len([val for val in college[n] if val in college[nbr]]) > 0:
                    similar_neighbors_C+=1
            if n in location and nbr in location:
                if len([val for val in location[n] if val in location[nbr]]) > 0:
                    similar_neighbors_L+=1
    homophily_E=similar_neighbors_E/total_number_neighbors 
    homophily_C=similar_neighbors_C/total_number_neighbors
    homophily_L=similar_neighbors_L/total_number_neighbors
    print("\n The homophily for E,L,C are respectively :  ",homophily_E,homophily_L,homophily_C)
    return homophily_E,homophily_L,homophily_C



def college_shared(i,j):
    att_shared=[]
    if i in college:
        if j in college:
            for attj in college[j]:
                for atti in college[i]:
                    if atti==attj:
                        att_shared.append(atti)
    return (att_shared)

def location_shared(i,j):
    att_shared=[]
    if i in location:
        if j in location: 
            for attj in location[j]:
                for atti in location[i]:
                    if atti==attj:
                        att_shared.append(atti)
    return (att_shared)

def employer_shared(i,j):
    att_shared=[]
    if i in employer:
        if j in employer:
            for attj in employer[j]:
                for atti in employer[i]:
                    if atti==attj:
                        att_shared.append(atti)
    return (att_shared)


def probas_conditionnelles(G):
    location_knowing_college=0
    employer_knowing_college=0
    employer_knowing_location=0
    nb_total_college=0
    nb_total_location=0
    nb_total_employer=0
    total_college_shared=0
    total_location_shared=0
    total_employer_shared=0

    for i in G.nodes:
        for j in G.nodes:
            if i!=j:
                if i in college:
                    if j in college:
                        nb_total_college+=1
                if i in location:
                    if j in location:
                        nb_total_location+=1
                if i in employer:
                    if j in employer:
                        nb_total_employer+=1


                if len(college_shared(i,j))!=0: 
                    total_college_shared+=1
                    if len(location_shared(i,j))!=0:
                        location_knowing_college+=1
                    if len(employer_shared(i,j))!=0:
                        employer_knowing_college+=1


                if len(location_shared(i,j))!=0: 
                    
                    total_location_shared+=1
                    if len(employer_shared(i,j))!=0:
                       employer_knowing_location+=1
                
                if len(employer_shared(i,j))!=0:
                    total_employer_shared+=1

    Pc=total_college_shared/nb_total_college
    Pl=total_location_shared/nb_total_location
    Pe=total_employer_shared/nb_total_employer

    Pl_w_c=location_knowing_college/total_college_shared
    Pe_w_c=employer_knowing_college/total_college_shared
    Pe_w_l=employer_knowing_location/total_location_shared
    Pl_w_e=Pe_w_l*(Pe/Pl)
    Pc_w_l=Pl_w_c*(Pl/Pc)
    Pc_w_e=Pe_w_c*(Pe/Pc)
    return(Pe_w_l,Pe_w_c,Pc_w_e,Pl_w_c,Pc_w_l,Pl_w_e)