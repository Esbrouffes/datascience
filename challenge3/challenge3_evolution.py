

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
	""" a function to convert a matrix or array into a graph, taking into account the elevation and population"""
	Hj={}
	G=nx.Graph()
	lines,columns=M.shape
	for i in range(lines):
		for j in range(columns): 
			k=i*columns+j #creating the nodes of the graph with nodes numbered by creascent number of column 
			G.add_node(k)
			Hj[k]=round(popu_cells[i][j]) # inserting the population cells 

	for i in range(lines):
		for j in range(columns): # we will now add the adges between the nodes, considering their position in the original matrix, weight being the angle between two cells.
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


G,Hj=toGraph(elevation_cells) #we create the graph 




"""GRAPH TEST : This graph was created so we could test how the zombie epidemics spreads with time.
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
	"""gets zombies older"""
	if i in zombies:
		zombies[i].insert(0,0)
		deads=zombies[i].pop()
		return deads

	

"""zombies will stay if there are no humans around"""	
def spreading(cell):
	"""this functions make the spreading of zombies from the cell given in arguments to its neighbors"""

	if cell in Zj:
		H_cell=0
		lambda_list=[]
		for i in G.neighbors(cell):
			if i in Hj: H_cell+=Hj[i]
			lambda_list	.append(1-G[cell][i]['weight']) # we will use it to compute the geographical slope factor. It is the angle divided by the mean angle, this way 100% of the zombies move
			#print((G[cell][i]['weight']))
		lambda_mean=np.mean(lambda_list)
		if lambda_mean==0:
			lambda_mean=1 #so no division by zero -> we are looking at absolute values therefore if mean is 0, they are all 0 , it will be 0/1 = 0. Moreover, due to the symetry of the slope factor, it isn't possible to have a "hole" or " peak" with zombies inside ! 
		
	
		
		### STEP ONE 
		N=sum(Zj[cell]) #number of zombies in the original cell
		if H_cell!=0:
			for i in G.neighbors(cell):
				if i in Zj_1: #it means the cell is already in the zombie land
					for j in range(len(Zj[cell])):
						
						Zj_1[i][j]+=round(Zj[cell][j]*Hj[i]/H_cell * (1-G[cell][i]['weight'])/lambda_mean)
						Zj_1[cell][j]-=round(Zj[cell][j]*Hj[i]/H_cell * (1-G[cell][i]['weight'])/lambda_mean )
						if Zj_1[cell][j] <0: Zj_1[cell][j] = 0 # so the round doesn't make negative numbers of zombies 
				else: 
					Zj_1[i]=[] #we have to create the cell
					for j in range(len(Zj[cell])):
						Zj_1[i].append(round(Zj[cell][j]*Hj[i]/H_cell * (1-G[cell][i]['weight'])/lambda_mean)) #round is because there are no such things as half zombies ! 
						Zj_1[cell][j]-=round(Zj[cell][j]*Hj[i]/H_cell * (1-G[cell][i]['weight'])/lambda_mean ) #we have to take them back from the original cell (actually it's from its copy that will be kept at the end of the propagation algorithm)
						if Zj_1[cell][j] <0: Zj_1[cell][j] = 0 # the round can cause nb of zombies <0 which isn't good ! 
			
		 
		

		




"""for i in G.neighbors(cell):    A PART OF THE CODE THAT ISN'T USED ANYMORE BUT WAS WITH OUR ORIGINAL FROMAT
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
			Hj[cell]-=10*N #we take the humans back
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
						Zj_1[cell][j]-=round(10*M*Zj_1[cell][j]/N) #zombies have to die proportionnaly to their ages ! 
						if Zj_1[cell][j] <0: Zj_1[cell][j] = 0  #zombies die no matter the age, zombies that don't exist don't die : only - for ages with a nb of zombies != 0 
				return 10*M
			elif M!=0:
				

				Zj_1[cell]=[0]*15
				
				return M
		if N==0 : return 0
	else: return 0

def nb_zombies(zombies):
	"""computes the number of zombies in a zombie dictionnary ( Zj or Zj_1) """
	Z=0
	for i in G.nodes():
		if i in zombies:
			Z+=sum(zombies[i])
	return Z
def nb_humans(humans):
	"""compute the number of humans """ 
	H=0
	for i in humans:
		H+=humans[i]
	return H



def daily_fights(G,Zj,Zj_1,Hj):
	""" does the whole process that happens during one day ( the several steps ) """
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
	



	
	
	return nb_zombies(Zj_1),Znew,Z_killed,Zold #these are important values if we want to make sure the number of zombies evolves correctly in a simple graph : no zombies are created or deleted without reason ! 

def reducing(zombies):
	"""we can pop all the cells that are empty of zombies but referenced in a dictionnary ( they are like : 'j': [0,0,0, ..., 0])"""
	to_pop=[]
	for j in zombies:
		if zombies[j]==[0]*15:
			to_pop.append(j) #otherwise it changes the size of dictionnary between iterations ( not elegant but it works fine ! )

	for j in to_pop:
		zombies.pop(j)


def naive_h(humans):
	"""returns the 20 most populated human cells at a time """
	return sorted(humans,key=humans.__getitem__)[len(humans)-20:len(humans)]

def ego_level_simple_centrality(G,r):
	sums={}
	for i in G.nodes():
		total_local =0
		for j in nx.ego_graph(G,i,r):
			total_local+=Hj[j]
		sums[i]=total_local
	return sorted(sums,key=sums.__getitem__)[len(sums)-20:len(sums)]#return the best cells according to this model 


	

"""Tests 




global Zj
global Zj_1
Zj={}
Zj_1={}


Zj[58400]=[30,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # 30 zombies in the cells in the corner, and we have to fill the humans with 110 humans in each cell so the evolution is easily computable with hand
#print(Zj)
Zj_1[58400]=[30,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
"""
"""
Test vieillissement : ok
for i in range(13):
	for i in G.nodes():
		vieillissement(Zj,i)
print(Zj)
-> fonctionnal
"""

"""Test spreading : ok 
for i in G.nodes():
	spreading(i)
print(Zj_1)
-> fonctionnal """


"""Test zombified: ok  
for i in G.nodes():
	vieillissement(Zj_1,i)
	zombified(i)

-> fonctionnal
"""

"""Test killed : ok """


""" daily fights test : ok 
for _ in range(5):
	print("-------------------day ------------------",_)   THESE IS TO SEE THE EVOLUTION 
	print(daily_fights(G,Zj,Zj_1,Hj))
	Zj=copy.deepcopy(Zj_1)


"""

####  MAIN -> the real maps and real numbers of humans 

#Initialisation



lines,columns=elevation_cells.shape
rize=columns*141+284 
brest= columns*38+33 
#print(brest)
Hj[rize]=0
global Zj
global Zj_1

Hu_plots=[]
Zu_plots=[]
for r in range(3):
	G,Hj=toGraph(elevation_cells) #we create the graph 
	Zj={}
	Zj_1={}

	Zj[rize]=[95000,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # " the whole population of rize is contamined"

	Zj_1[rize]=[95000,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

	days=0
	timelapse=[days]
	Hu=[nb_humans(Hj)]
	Zu=[nb_zombies(Zj_1)]
	#beginning
	t=time.time()
	#while days<=500: or 
	while brest not in Zj: # we stop the spreading when they arrive there, but we could continue, by commenting this line and uncommenting the previous one ! 


		print(nb_zombies(Zj_1))
		print(nb_humans(Hj)) # for few statistics
		days+=1
		timelapse.append(days) # to see the evolution
		daily_fights(G,Zj,Zj_1,Hj) # the daily routine
		reducing(Zj_1) #
		Zj=copy.deepcopy(Zj_1)
		Hu.append(nb_humans(Hj))
		Zu.append(nb_zombies(Zj_1))
		
		print("------------------------------------------------------ day ",days, " ----------------------------------------------------------------")
		if days==61: 
			for i in ego_level_simple_centrality(G,r):
				print(Hj[i])
				if i in G:G.remove_node(i)

	print("BREST IS ZOMBIFIED !!! This happens at day ",days )
	print(nb_zombies(Zj_1))
	print(nb_humans(Hj))

	print("time to execute = ", time.time()-t)


	Hu_plots.append(Hu)
	Zu_plots.append(Zu)

plt.figure()
for human in Hu_plots:
	plt.plot(timelapse,human)
plt.figure()
for zombies in Zu_plots:
	plt.plot(timelapse,zombies)
plt.show()
			


"""
	evaluation : days before reaching brest, number of humans killed before troops vs after, number of zombies 10 days after before / after , number of zombies 30 days after. 
	Quantitatif vs qualitatif 


	"""





""" 


RESULT : 

2 month after, 

2123775.0
2871845.0


2123775.0
2813804.0"""