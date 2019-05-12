# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:09:11 2017

@author: cbothore
"""


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pickle
from collections import Counter



def naive_method(graph, empty, attr):
    """   Predict the missing attribute with a simple but effective
    relational classifier. 
    
    The assumption is that two connected nodes are 
    likely to share the same attribute value. Here we chose the most frequently
    used attribute by the neighbors
    
    Parameters
    ----------
    graph : graph
       A networkx graph
    empty : list
       The nodes with empty attributes 
    attr : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.

    Returns
    -------
    predicted_values : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node (from empty), value is a list of attribute values. Here 
       only 1 value in the list.
     """
    predicted_values={}
    for n in empty:
        nbrs_attr_values=[] 
        for nbr in graph.neighbors(n):
            if nbr in attr:
                for val in attr[nbr]:
                    nbrs_attr_values.append(val)
        predicted_values[n]=[]
        if nbrs_attr_values: # non empty list
            # count the number of occurrence each value and returns a dict
            cpt=Counter(nbrs_attr_values)
            # take the most represented attribute value among neighbors
            a,nb_occurrence=max(cpt.items(), key=lambda t: t[1])
            predicted_values[n].append(a)
    return predicted_values
    
 
def evaluation_accuracy(groundtruth, pred):
    """    Compute the accuracy of your model.

     The accuracy is the proportion of true results.

    Parameters
    ----------
    groundtruth :  : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values.
    pred : dict 
       A dict of attributes, either location, employer or college attributes. 
       key is a node, value is a list of attribute values. 

    Returns
    -------
    out : float
       Accuracy.
    """
    true_positive_prediction=0   
    for p_key, p_value in pred.items():
        if p_key in groundtruth:
            # if prediction is no attribute values, e.g. [] and so is the groundtruth
            # May happen
            if not p_value and not groundtruth[p_key]:
                true_positive_prediction+=1
            # counts the number of good prediction for node p_key
            # here len(p_value)=1 but we could have tried to predict more values
            true_positive_prediction += len([c for c in p_value if c in groundtruth[p_key]])          
        # no else, should not happen: train and test datasets are consistent
    return true_positive_prediction*100/sum(len(v) for v in pred.values())
   

# load the graph
G = nx.read_gexf("mediumLinkedin.gexf")
print("Nb of users in our graph: %d" % len(G))

# load the profiles. 3 files for each type of attribute
# Some nodes in G have no attributes
# Some nodes may have 1 attribute 'location'
# Some nodes may have 1 or more 'colleges' or 'employers', so we
# use dictionaries to store the attributes
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

print("Nb of users with one or more attribute college: %d" % len(college))
print("Nb of users with one or more attribute location: %d" % len(location))
print("Nb of users with one or more attribute employer: %d" % len(employer))

# here are the empty nodes for whom your challenge is to find the profiles
empty_nodes=[]
with open('mediumRemovedNodes_60percent_of_empty_profile.pickle', 'rb') as handle:
    empty_nodes = pickle.load(handle)
print("Your mission, find attributes to %d users with empty profile" % len(empty_nodes))


# --------------------- Baseline method -------------------------------------#
# Try a naive method to predict attribute
# This will be a baseline method for you, i.e. you will compare your performance
# with this method
# Let's try with the attribute 'employer'
employer_predictions=naive_method(G, empty_nodes, employer)
location_prediction=naive_method(G,empty_nodes, location)

groundtruth_employer={}
with open('mediumEmployer.pickle', 'rb') as handle : 
    groundtruth_employer = pickle.load(handle)
with open('mediumLocation.pickle', 'rb') as handle : 
    groundtruth_location = pickle.load(handle)
result=evaluation_accuracy(groundtruth_employer,employer_predictions)

#result=evaluation_accuracy(groundtruth_location, location_prediction)
print("%f%% of the predictions are true" % result)
print("Very poor result!!! Try to do better!!!!")

# --------------------- Now your turn -------------------------------------#
# Explore, implement your strategy to fill empty profiles of empty_nodes


# and compare with the ground truth (what you should have predicted)
# user precision and recall measures

print ( "\n------------------------- Our answers -----------------------")


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

properties(G)
#### COmputing the homophily 
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
#compute the clustering coeff for each attribute will help us define where each one lives. 
#print("Clustering coefficients %f", list(nx.clustering(G).values()))
plt.hist(list(nx.clustering(G).values()))
#plt.show()
""" how to use the clustering coefficient ??? 
    how to use homophily ?? 
    if people go to the same school, they might be connected. 
    then ( contraposée ) if they are not connected it  means they were probably not in the same school -> use 1-homophily again !!!

    Can we rely on homophily as we computed it ? or should we compute another version of homophily computing it for only people we know have filled in
     the information concerning the attribute ? 
     """



#let's take the people who live at each location 


ListOfLocations={}
LocationsAndClusters={}
for j in G.nodes:
    if j in location:
        for k in location[j]:
            if k not in ListOfLocations:
                ListOfLocations[k]=[]
            ListOfLocations[k].append(j)

for l in ListOfLocations:
    connected=0
    total=0

    for k in ListOfLocations[l]:
        for m in ListOfLocations[l]:
            if m in G.neighbors(k):
                connected+=1
            total+=1
    LocationsAndClusters[l]=connected/total
    if len(ListOfLocations[l])==1:
        LocationsAndClusters[l]="Only one people ---"
#print(LocationsAndClusters) # the "clustering coefficient" for each school 


#Let's compute the probability of sharing another attribute knowing people already share an attribute. 

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
print(Pe_w_l,Pe_w_c,Pc_w_e,Pl_w_c,Pc_w_l,Pl_w_e)
        



