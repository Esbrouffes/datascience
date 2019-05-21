

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
G=nx.Graph()
Hj={}
G.add_nodes_from(range(36))
for i in G.nodes():
	if i in [7,28]:
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
	Hj[i]=15





global Zj
global Zj_1
Zj={}
Zj_1={}



Zj[0]=[(0,8),(1,9)]
Zj_1[0]=[(0,8),(1,9)]


""" format {'i'[(age,nb),(age,nb),(age,nb)], 'j'[]}etc etc """

""" ZOMBIES HAVE A LIFE EXEPCTANCY """
def vieillissement(zombies,i):
	"""effectue le vieillissement d'une liste de zombies"""
	if i in zombies:
	
		
		for j in range(len(zombies[i])):

			if zombies[i][j][0]<14:

				n=zombies[i][j][1]
				a=zombies[i][j][0]+1

				zombies[i][j]=(a,n)
				
			else:
				p=j
				zombies[i].pop(p)
	
def spreading(cell):
	"""how will we use fill the graph ? """
	if cell in Zj:
		H_cell=0
		for i in G.neighbors(cell):
			H_cell+=Hj[i]
			

		### STEP ONE 
		N=0 #nombre de zombies dans la cellule originale
		for z in range(len(Zj_1[cell])):
			N+=Zj_1[cell][z][1]
		print('aa',Zj_1[cell])

		for i in G.neighbors(cell): 
			if i in Zj:
				for ages in range(len(Zj[i])):
					if Zj[i][ages][0]==Zj[cell][ages][0]:
						Zj_1[i][ages][1]+=N/len(Zj[i])*Hj[i]/H_cell

					else:
						Zj_1[i].append((Zj[cell][ages][1],N/len(Zj[i])*Hj[i]/H_cell))
						print(Zj_1)
				#vieillisemment(Z_j,i) now or after
				nb=N*Hj[i]/H_cell
				age=Zj_1
			


def zombified(cell):
	"""zombies killing humans"""
	N=0
	for z in range(len(Zj_1[cell])):
		N+=Zj_1[cell][z][1]
	M=Hj[cell]
	if 10*N>=M:
		Hj[cell]-=10*N
		Zj_1[cell].append((0,10*N))
		print(10*N,'aaaaa')
		return 10*N
	elif N!=0:
		Hj[cell]=0
		Zj_1[cell].append((0,M))
		print(M,'aaaaa')
		return M



def killed(cell): 
	"""human killing zombies"""
	N=0
	for z in range(len(Zj_1[cell])):
		N+=Zj_1[cell][z][1]
	M=Hj[cell]
	if 10*M>=N:
		for j in Zj_1[cell]:
			Zj_1[cell][j]-=10*M/len(Zj_1[cell]) #zombies die no matter the age.
			return 10*M
	elif M!=0:

		Zj_1[cell]=[]
		return N



def daily_fights(G,Zj,Zj_1,Hj):
	Total_H=0
	Total_Z=0
	for c in G.nodes():
		vieillissement(Zj,c)
		vieillissement(Zj_1,c)
		spreading(c)
		for z in range(len(Zj_1[c])):
			Total_Z+= Zj[c][z][1]

		Total_H+=Hj[c]
	Znew=0
	for c in G.nodes():
		Znew+=zombified(cell)
	Z_killed=0
	for c in G.nodes():
		Z_killed+=killed(c)

	Zj=Zj_1
	
	return Total_H,Total_Z,Znew,Z_killed
	"""on peut avoir des compteurs pour certifier que le résultat est bon ( autant de zombies avant que après - nouveaux + ceux tués )
"""

"""Test vieillissement : ok
for i in range(13):
	for i in G.nodes():
		vieillissement(Zj,i)
print(Zj)
"""
daily_fights(G,Zj,Zj_1,Hj)

problems avec spreading !!! 





