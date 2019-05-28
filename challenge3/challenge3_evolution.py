

"""Let G be the graph where the nodes are 15x15 px  squares, the edges are the connections, and the slopes is the weight of the graph. Each node will have 4 attributes : Zj,Zj+1, Hj, Hj+1 """


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pickle
from collections import Counter
from networkx.algorithms import community
import community
import operator
import copy 
G=nx.Graph()
Hj={}
G.add_nodes_from(range(36))
for i in G.nodes():
	if i 
		G.add_edges_from([(i-1,i),(i+1,i),(i-7,i),(i-6,i),(i-5,i),(i+5,i),(i+6,i),(i+7,i)])
	if i==6:
		G.add_edges_from([(i-1,i),(i+1,i),(i-6,i),(i-5,i),(i+5,i),(i+6,i),(i+7,i)])
	if i==5:
		G.add_edges_from([(i-1,i),(i+1,i),(i-5,i),(i+5,i),(i+6,i),(i+7,i)])
	if i in[1,4]:
		G.add_edges_from([(i-1,i),(i+1,i),(i+5,i),(i+6,i),(i+7,i)])
	if i==0:
		G.add_edges_from([(i+1,i),(i+5,i),(i+6,i),(i+7,i)])
	if i==29:
		G.add_edges_from([(i-1,i),(i+1,i),(i-7,i),(i-6,i),(i-5,i),(i+5,i),(i+6,i)])
	if i==30:
		G.add_edges_from([(i-1,i),(i+1,i),(i-7,i),(i-6,i),(i-5,i),(i+5,i)])
	if i in [31,34]: 
		G.add_edges_from([(i-1,i),(i+1,i),(i-7,i),(i-6,i),(i-5,i)])
	if i==35: 
		G.add_edges_from([(i-1,i),(i-7,i),(i-6,i),(i-5,i)])
	if i in[13,14,15,19,21,25,26,27]: Hj[i]=110
	else:Hj[i]=0


#print(Hj)



global Zj
global Zj_1
Zj={}
Zj_1={}



Zj[20]=[80,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#print(Zj)
Zj_1[20]=[80,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

""" format {'i'[(age,nb),(age,nb),(age,nb)], 'j'[]}etc etc """

""" ZOMBIES HAVE A LIFE EXEPCTANCY """
def vieillissement(zombies,i):
	"""effectue le vieillissement d'une liste de zombies"""
	if i in zombies:
		zombies[i].insert(0,0)
		deads=zombies[i].pop()
		return deads

	

"""zombies will stay if there are no humans around"""	
def spreading(cell):
	"""how will we use fill the graph ? """
	""" good idea nb of zombies in an array, filled with 0s """
	if cell in Zj:
		H_cell=0
		for i in G.neighbors(cell):
			H_cell+=Hj[i]
			
		
		### STEP ONE 
		N=sum(Zj[cell]) #nombre de zombies dans la cellule originale
		if H_cell!=0:
			for i in G.neighbors(cell):
				if i in Zj_1:
					for j in range(len(Zj[cell])):

						Zj_1[i][j]+=round(Zj[cell][j]*Hj[i]/H_cell )
						Zj_1[cell][j]-=round(Zj[cell][j]*Hj[i]/H_cell )
						
				else: 
					Zj_1[i]=[]
					for j in range(len(Zj[cell])):
						Zj_1[i].append(round(Zj[cell][j]*Hj[i]/H_cell))
						Zj_1[cell][j]-=round(Zj[cell][j]*Hj[i]/H_cell )
			
		
		

		




"""for i in G.neighbors(cell): 
			if i in Zj_1:
				for j in range(len(Zj[cell])):
					if Zj[cell][j][0]==Zj_1[i][j][0]:
						Zj_1[i][j][1]+=Zj[cell][j][1]*Hj[i]/H_cell
					else: #l'age n'est pas dans la liste des zombies
						Zj_1[i].append((Zj[cell][j][0],Zj[cell][j][1]*Hj[i]/H_cell))
			else : 
				Zj_1[i]=[]
				for j in range(len(Zj[cell])):
					Zj_1[i].append((Zj[cell][j][0],Zj[cell][j][1]*Hj[i]/H_cell)) ### NB DE ZOMBIES PAS ENTIER !!!!!!!!

	if cell in Zj:
			if H_cell!=0:
				Zj_1.pop(cell)
	return 0"""
"""
				for ages in range(len(Zj[i])):
					print(age)
					if Zj[i][ages][0]==Zj[cell][ages][0]:
						Zj_1[i][ages][1]+=N/len(Zj[i])*Hj[i]/H_cell
						print('aaa')
					else:
						Zj_1[i].append((Zj[cell][ages][1],N/len(Zj[i])*Hj[i]/H_cell))
						print(Zj_1)
				#vieillisemment(Z_j,i) now or after
				nb=N*Hj[i]/H_cell
				age=Zj_1
		"""


def zombified(cell):
	"""zombies killing humans"""
	if cell in Zj_1:
		N=sum(Zj_1[cell])
		M=Hj[cell]
		if 10*N<=M:
			Hj[cell]-=10*N
			Zj_1[cell][0]=10*N
			
			return 10*N
		elif N!=0:
			Hj[cell]=0
			Zj_1[cell][0]=M
			
			return M
	else: return 0



def killed(cell): 
	"""human killing zombies"""
	if cell in Zj_1:
		N=sum(Zj_1[cell])
		M=Hj[cell]
		if 10*M<=N:
			lenzombies=0
			for j in range(len(Zj_1[cell])):
				if Zj_1[cell][j]!=0: lenzombies+=1
			for j in range(len(Zj_1[cell])):
				if Zj_1[cell][j]!=0: Zj_1[cell][j]-=round(10*M*Zj_1[cell][j]/N) #zombies die no matter the age, zombies that don't exist don't die : only - for ages with a nb of zombies != 0 
			return 10*M
		elif M!=0:
			

			Zj_1[cell]=[0]*15
			
			return M
	else: return 0

def nb_zombies(zombies):
	Z=0
	for i in  G.nodes():
		if i in zombies: Z+=sum(zombies[i])
	return Z

def daily_fights(G,Zj,Zj_1,Hj):
	Total_H=0
	Zold=nb_zombies(Zj_1)
	for c in G.nodes():
		vieillissement(Zj,c)
		vieillissement(Zj_1,c)

	#print("-1111" ,Zj,Zj_1)
	for c in G.nodes():
		spreading(c)
		
	#Zj=copy.deepcopy(Zj_1)
	print("spread",nb_zombies(Zj_1),Zj_1)
	Znew=0
	Z_killed=0
	for c in G.nodes():
		Znew+=zombified(c)
	print("---------zombified",nb_zombies(Zj_1),Zj_1)
	for c in G.nodes():
		Z_killed+=killed(c)
		Total_H+=Hj[c]

	print("-----------killed",nb_zombies(Zj_1),Zj_1)
	



	
	
	return Total_H,nb_zombies(Zj_1),Znew,Z_killed,Zold #il faudrait calculer toutes les valeurs en fin, plutot



	"""on peut avoir des compteurs pour certifier que le résultat est bon ( autant de zombies avant que après - nouveaux + ceux tués )
"""

"""Tests unitaires """

"""Test vieillissement : ok
for i in range(13):
	for i in G.nodes():
		vieillissement(Zj,i)
print(Zj)
-> fonctionnel
"""

"""Test spreading : ok 
for i in G.nodes():
	spreading(i)
print(Zj_1)
-> fonctionnel """


"""Test zombified: ok  
for i in G.nodes():
	vieillissement(Zj_1,i)
	zombified(i)

-> fonctionnel
"""

"""Test killed : """
print(Zj)
for _ in range(1,3):
	print("-------------------day ------------------",_)
	print(daily_fights(G,Zj,Zj_1,Hj))
	Zj=copy.deepcopy(Zj_1)
	
 


"""IS CENTRALITY ENOUGH ??? WE NEED TO CONSIDER THE NUMBER OF HUMANS AND THE CENTRALITY IN ORDER TO DO THAT"""