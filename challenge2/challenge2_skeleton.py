# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 16:09:11 2017

@author: cbothore
"""


from statistiques import * 
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pickle
from collections import Counter
from networkx.algorithms import community
import community


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
def evaluation_accuracy2(groundtruth, pred):
    true_positive_prediction=0   
    for p_key, p_value in pred.items():
        if p_key in groundtruth:
            # if prediction is no attribute values, e.g. [] and so is the groundtruth
            # May happen
            if not p_value and not groundtruth[p_key]:
                true_positive_prediction+=1
            # counts the number of good prediction for node p_key
            # here len(p_value)=1 but we could have tried to predict more values
            if p_value in groundtruth[p_key]:
                true_positive_prediction += 1       #len([c for c  in p_value if c in groundtruth[p_key]])          
        # no else, should not happen: train and test datasets are consistent
    return true_positive_prediction*100/len(pred.items())

   

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
location_prediction =naive_method(G,empty_nodes, location)

groundtruth_employer={}
with open('mediumEmployer.pickle', 'rb') as handle : 
    groundtruth_employer = pickle.load(handle)
with open('mediumLocation.pickle', 'rb') as handle : 
    groundtruth_location = pickle.load(handle)
with open('mediumCollege.pickle', 'rb') as handle : 
    groundtruth_college = pickle.load(handle)
result=evaluation_accuracy(groundtruth_employer,employer_predictions)

#result=evaluation_accuracy(groundtruth_location, location_prediction)
print("%f%% of the predictions are true" % result)


# --------------------- Now your turn -------------------------------------#
# Explore, implement your strategy to fill empty profiles of empty_nodes


# and compare with the ground truth (what you should have predicted)
# user precision and recall measures

print ( "\n------------------------- Our answers -----------------------")



properties(G)

#compute the clustering coeff for each attribute will help us define where each one lives. 
#print("Clustering coefficients %f", list(nx.clustering(G).values()))
#plt.hist(list(nx.clustering(G).values()))
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


""" functions """ 

#let's determine where are the communities 
#print(community.k_clique_communities(G,5))


#print(partition)
def extracting_communities(partition):
    communities={}
    for i in G.nodes:
        if partition[i] in communities:
            communities[partition[i]].append(i)
        else:
            communities[partition[i]]=[i]
    return(communities)




def voir_partition(partition,G):
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)
    count = 0.
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                    node_color = str(count / size))


    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
#print(location)
 ### USING LOUVAIN COMMUNITY - NAIVE MODEL 

partition =community.best_partition(G)
communities=extracting_communities(partition) 
#print(extracting_communities(partition))
"""echo_graph ??????????????????????????????????????????????"""
def louvain_naive(G):
    predicted_location={}
    predicted_employer={}
    predicted_college={}
    for i in empty_nodes:
        if i not in location: 
            
            possible_locations={}
            for j in communities[partition[i]]:
                if j in location:
                    for l in location[j]:
                        if l in possible_locations:
                            possible_locations[l]+=1
                        else: 
                            possible_locations[l]=0
            max=0
            for k in possible_locations:
               
                if possible_locations[k]>=max:
                    max=possible_locations[k]
                    predicted_location[i]=k

        if i not in college: 
            possible_colleges={}
            for j in communities[partition[i]]:
                if j in college:
                    for c in college[j]:
                        if c in possible_colleges:
                            possible_colleges[c]+=1
                        else: 
                            possible_colleges[c]=0
            max=0
            for k in possible_colleges:
               
                if possible_colleges[k]>=max:
                    max=possible_colleges[k]
                    predicted_college[i]=k

        if i not in employer: 
            possible_employers={}
            for j in communities[partition[i]]:
                if j in employer:
                    for e in employer[j]:

                        if e in possible_employers:
                            possible_employers[e]+=1
                        else: 
                            possible_employers[e]=0
            max=0
            for k in possible_employers:
               
                if possible_employers[k]>=max:
                    max=possible_employers[k]
                    predicted_employer[i]=k
    
    return predicted_college,predicted_location,predicted_employer

predicted_college,predicted_location,predicted_employer=louvain_naive(G)

#print(empty_nodes)
#print(predicted_employer)
#print(groundtruth_employer)

result=evaluation_accuracy2(groundtruth_location,predicted_location)
print(result)
#print(groundtruth)

def ego_niveau2(i):
    niveau2=[]
    for j in G.neighbors(i):
        niveau2.append(j)
        for k in G.neighbors(j):
            if k!=i:
                niveau2.append(k)
    return niveau2



def louvain_and_ego(G):
    predicted_location={}
    predicted_employer={}
    predicted_college={}
    for i in empty_nodes:
        if i not in location: 
            
            possible_locations={}
            for j in communities[partition[i]] and G.neighbors(i) :
                if i!=j:
                    if j in location:
                        for l in location[j]:
                            if l in possible_locations:
                                possible_locations[l]+=1
                            else: 
                                possible_locations[l]=0
                max=0
                for k in possible_locations:
                   
                    if possible_locations[k]>=max:
                        max=possible_locations[k]
                        predicted_location[i]=k

        if i not in college: 
            possible_colleges={}
            for j in communities[partition[i]]:
                if j in college:
                    for c in college[j]:
                        if c in possible_colleges:
                            possible_colleges[c]+=1
                        else: 
                            possible_colleges[c]=0
            max=0
            for k in possible_colleges:
               
                if possible_colleges[k]>=max:
                    max=possible_colleges[k]
                    predicted_college[i]=k

        if i not in employer: 
            possible_employers={}
            for j in communities[partition[i]]:
                if j in employer:
                    for e in employer[j]:

                        if e in possible_employers:
                            possible_employers[e]+=1
                        else: 
                            possible_employers[e]=0
            max=0
            for k in possible_employers:
               
                if possible_employers[k]>=max:
                    max=possible_employers[k]
                    predicted_employer[i]=k
    
    return predicted_college,predicted_location,predicted_employer



predicted_college,predicted_location,predicted_employer=louvain_and_ego(G)


result=evaluation_accuracy2(groundtruth_location,predicted_location)
print(result)
#print(groundtruth)


#starting to use conditionnal porbabilities 
print(employer)
print(proba_knowing_job('google'))