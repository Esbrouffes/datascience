

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
import csv
import time
import math

"""graph creation -----------------"""
popu_cells = np.array(list(csv.reader(open("./DataPreparation/popu_cells.csv", "r"), delimiter=","))).astype("double")
elevation_cells = np.array(list(csv.reader(open("./DataPreparation/elevation_cells.csv", "r"), delimiter=","))).astype("double")



#print(elevation_cells)
#,columns=elevation_cells.shape
#print(lines,columns)


#Matrix or graph ???? graph -> Dijkstra etc 


def toGraph(M):
	Hj={}
	G=nx.Graph()
	lines,columns=M.shape
	for i in range(lines):
		for j in range(columns):
			k=i*columns+j
			G.add_node(k)
			Hj[k]=round(popu_cells[i][j])

	for i in range(lines):
		for j in range(columns):
			k=i*columns+j
			if i==0:#first line
				if j==0: #left top corner
					G.add_edges_from([(k+1,k),(k+columns,k),(k+columns+1,k)])
				elif j==columns-1: #right top corner
					G.add_edges_from([(k-1,k),(k+columns,k),(k+columns-1,k)])


				else: 
					G.add_edges_from([(k-1,k),(k+columns,k),(k+columns-1,k),(k+1,k),(k+columns+1,k)])


			elif i==lines-1:#last line
				if j==0: #left bottom corner
					G.add_edges_from([(k+1,k),(k-columns,k),(k-columns+1,k)])
				elif j==columns-1:#right bottom corner
					G.add_edges_from([(k-1,k),(k-columns,k),(k-columns-1,k)])

				else:	G.add_edges_from([(k-1,k),(k-columns,k),(k-columns-1,k),(k+1,k),(k-columns+1,k)]) # last line


			else:
				if j==0: #first column
						G.add_edges_from([(k-columns,k),(k+columns,k),(k-columns+1,k),(k+columns+1,k),(k+1,k)])

				elif j==columns-1:#last column
					G.add_edges_from([(k-columns,k),(k+columns,k),(k-columns-1,k),(k+columns-1,k),(k-1,k)])
				else:#middle cases
					G.add_edges_from([(k-columns,k),(k+columns,k),(k-columns+1,k),(k+columns+1,k),(k+1,k),(k-columns-1,k),(k+columns-1,k),(k-1,k)])


	for i in G.nodes(): # not optimal in term of execution speed but gives the opportunity to consider or not the elevation
		for j in G.neighbors(i):
			elevation_i=elevation_cells[i//columns][i%columns]
			elevation_j=elevation_cells[j//columns][j%columns]  
			angle=np.arctan((elevation_i-elevation_j)/15000)
			#print(np.degrees(angle))
			
			if math.degrees(abs(angle))>=10:
				#print("to high")		
				G[i][j]['weight'] = 1

			else:
				
				G[i][j]['weight']=abs(math.degrees(angle)/10)





	return G,Hj
t=time.time()
G,Hj=toGraph(elevation_cells)
a=0
for i in G.nodes():
	if i%5==0:
		a+=1
#print(a) ( tests basiques de durée etc )

#print(time.time()-t)
#print(G.edges())



"""GRAPH TEST : 
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

"""
#print(Hj)



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
		lambda_list=[]
		for i in G.neighbors(cell):
			H_cell+=Hj[i]
			lambda_list	.append(1-G[cell][i]['weight'])
			#print((G[cell][i]['weight']))
		lambda_mean=np.mean(lambda_list)
		if lambda_mean==0:
			lambda_mean=1 #so no division by zero 
		
	
		
		### STEP ONE 
		N=sum(Zj[cell]) #nombre de zombies dans la cellule originale
		if H_cell!=0:
			for i in G.neighbors(cell):
				if i in Zj_1:
					for j in range(len(Zj[cell])):
						
						Zj_1[i][j]+=round(Zj[cell][j]*Hj[i]/H_cell * (1-G[cell][i]['weight'])/lambda_mean)
						Zj_1[cell][j]-=round(Zj[cell][j]*Hj[i]/H_cell * (1-G[cell][i]['weight'])/lambda_mean )
						if Zj_1[cell][j] <0: Zj_1[cell][j] = 0 # so the round doesn't make negative numbers of zombies 
				else: 
					Zj_1[i]=[]
					for j in range(len(Zj[cell])):
						Zj_1[i].append(round(Zj[cell][j]*Hj[i]/H_cell * (1-G[cell][i]['weight'])/lambda_mean))
						Zj_1[cell][j]-=round(Zj[cell][j]*Hj[i]/H_cell * (1-G[cell][i]['weight'])/lambda_mean )
						if Zj_1[cell][j] <0: Zj_1[cell][j] = 0 
			
		 
		

		




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
		if N!=0:
			if 10*M<=N:
				lenzombies=0
				for j in range(len(Zj_1[cell])):
					if Zj_1[cell][j]!=0: lenzombies+=1
				for j in range(len(Zj_1[cell])):
					if Zj_1[cell][j]!=0: 
						Zj_1[cell][j]-=round(10*M*Zj_1[cell][j]/N)
						if Zj_1[cell][j] <0: Zj_1[cell][j] = 0  #zombies die no matter the age, zombies that don't exist don't die : only - for ages with a nb of zombies != 0 
				return 10*M
			elif M!=0:
				

				Zj_1[cell]=[0]*15
				
				return M
		if N==0 : return 0
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
	#print("spread",nb_zombies(Zj_1),Zj_1)
	Znew=0
	Z_killed=0
	for c in G.nodes():
		Znew+=zombified(c)
	#print("---------zombified",nb_zombies(Zj_1),Zj_1)
	for c in G.nodes():
		Z_killed+=killed(c)
		
		#Total_H+=Hj[c]

	#print("-----------killed",nb_zombies(Zj_1),Zj_1)
	



	
	
	return Total_H,nb_zombies(Zj_1),Znew,Z_killed,Zold #il faudrait calculer toutes les valeurs en fin, plutot

def reducing(zombies):
	to_pop=[]
	for j in zombies:
		if zombies[j]==[0]*15:
			to_pop.append(j) #otherwise it changes the size of dictionnary between iterations 

	for j in to_pop:
		zombies.pop(j)

def naive(humans):
	return sorted(humans,key=humans.__getitem__)[len(humans)-20:len(humans)]


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

"""Test killed : ok """


"""



global Zj
global Zj_1
Zj={}
Zj_1={}



Zj[58400]=[30,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#print(Zj)
Zj_1[58400]=[30,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for _ in range():
	print("-------------------day ------------------",_)   THESE IS TO SEE THE EVOLUTION 
	print(daily_fights(G,Zj,Zj_1,Hj))
	Zj=copy.deepcopy(Zj_1)
	

"""





"""IS CENTRALITY ENOUGH ??? WE NEED TO CONSIDER THE NUMBER OF HUMANS AND THE CENTRALITY IN ORDER TO DO THAT

What's left to do : 


-considering elevation ! 
-geographical slope
-finding all the shortest paths
-having models for the cells we want to protect ! 

Problem ??? -> The length of the dictionnary is bigger and bigger -> it takes a lot of time to copy it -> reducing 



"""

####  MAIN 

#Initialisation
lines,columns=elevation_cells.shape
rize=columns*141+284 
brest= columns*38+33 
#print(brest)
Hj[rize]=0
global Zj
global Zj_1
Zj={}
Zj_1={}

Zj[rize]=[95000,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

Zj_1[rize]=[95000,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

days=0

#beginning
t=time.time()
while brest not in Zj:
	days+=1
	daily_fights(G,Zj,Zj_1,Hj)
	reducing(Zj_1)
	Zj=copy.deepcopy(Zj_1)
		
	print("------------------------------------------------------ day ",days, " -------------------------------------------------------------------")
	#print(Zj)
	if days==61: 
		print("two months after the beginning of the zombie apocalypse")
		print(naive(Hj))
		for i in naive(Hj):
			#print(Hj[i])
			Zj.pop(i) #the zombies are killed ! 
			Hj.pop(i) #pop ? we don't know what to do -> store them in another dictionnary



evaluation : days before reaching brest, number of humans killed before troops vs after, number of zombies 10 days after before / after , number of zombies 30 days after. 
Quantitatif vs qualitatif 









print("BREST IS ZOMBIFIED !!! This happens at day ",days)

print("time to execute = ", time.time()-t)
