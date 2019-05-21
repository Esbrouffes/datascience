

"""Let G be the graph where the nodes are 15x15 px  squares, the edges are the connections, and the slopes is the weight of the graph. Each node will have 4 attributes : Zj,Zj+1, Hj, Hj+1 """


from statistiques import * 
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np
import pickle
from collections import Counter
from networkx.algorithms import community
import community
import operator
G=nx.graph()

Zj={}
Zj_1={}
Hj={}
Hj_1={}
""" format {'i'[(age,nb),(age,nb),(age,nb)], 'j'[]}etc etc """

""" ZOMBIES HAVE A LIOFE EXEPCTANCY """
def viellissement(zombies,i):
	"""effectue le vieillissement d'une liste de zombies"""
	for j in zombies[i]:
		if zombies[i][j][0]<=14:
			zombies[i][j][0]+=1
		else:
			p=j
	zombies.pop(i[p])

def spreading(cell):
	"""how will we use fill the graph ? """
	H_cell=0
	for i in G.neighbors(cell):
		H_cell+=Hj[i]

	### STEP ZERO
	

	### STEP ONE 

	for i in G.neighbors(cell):
		Zj[i]=Zj_1[i] #Maintenant ou en fin 
		#vieillisemment(Z_j,i) now or after
		Zj_1[i]=Z_j[cell]*Hj[i]/H_cell*G[cell][i]['weight']



def zombified(cell):
	"""zombies killing humans"""
	N=0
	for z in Zj_1[cell]:
		N+=Zj_1[cell][z][1]
	M=Hj[cell]
	if 10*N>=M:
		Hj[cell]-=10*N
		Zj_1+=10*N
		return 10*N
	else if N!=0:
		Hj[cell]=0
		Zj_1[cell].append((0,M))
		return M



def killed(cell): 
	"""human killing zombies"""
	N=0
	for z in Zj_1[cell]:
		N+=Zj_1[cell][z][1]
	M=Hj[cell]
	if 10*M>=N:
		for j in Zj_1[cell]:
			Zj_1[cell][j]-=10*M/len(Zj_1[cell]) #zombies die no matter the age.
			return 10*M
	else if M!=0:
		Zj_1[cell]=[]
		return N



def daily_fights(G):
	Total_H=0
	Total_Z=0
	for c in G.nodes():
		vieillissement(Zj,c)
		vieillissement(Zj_1,c)
		spreading(c)
		for j in Zj[cell]:
			Total_Z+= Zj[cell][j][1]

		Total_H+=Hj[cell]
	Znew=0
	for c in G.nodes():
		Znew+=zombified(cell)
	Z_killed=0
	for c in G.nodes():
		Z_killed+=killed(c)
	return Total_H,Total_Z,Znew,Z_killed
	"""on peut avoir des compteurs pour certifier que le résultat est bon ( autant de zombies avant que après - nouveaux + ceux tués )













