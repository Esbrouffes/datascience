import csv
import numpy
import networkx as nx

popu_cells = numpy.array(list(csv.reader(open("../DataPreparation/popu_cells.csv", "r"), delimiter=","))).astype("double")
elevation_cells = numpy.array(list(csv.reader(open("../DataPreparation/elevation_cells.csv", "r"), delimiter=","))).astype("double")


print(elevation_cells)
#,columns=elevation_cells.shape
#print(lines,columns)


#Matrix or graph ???? graph -> Dijkstra etc 


def toGraph(M):
	G=nx.Graph()
	lines,columns=M.shape
	for i in range(M.shape):
		for j in range(columns):
			k=i*columns+j
			G.add_node(k)

	for i in range(M.shape):
		for j in range(columns):
			k=i*columns+j
			if i==0:#first line
				if j==0: #left top corner
					G.add_edge_from([k+1,k],[k+columns,k],[k+columns+1,k])
				if j==columns-1: #right top corner
					G.add_edge_from([k-1,k],[k+columns,k],[k+columns-1,k])


				else: 
					G.add_edge_from([k-1,k],[k+columns,k],[k+columns-1,k],[k+1,k],[k+columns+1,k])


			if i==lines-1#last line
				if j==0: #left bottom corner
					G.add_edge_from([k+1,k],[k-columns,k],[k-columns+1,k])
				if j==columns-1:#right bottom corner
					G.add_edge_from([k-1,k],[k-columns,k],[k-columns-1,k])

				else:	G.add_edge_from([k-1,k],[k-columns,k],[k-columns-1,k],[k+1,k],[k-columns+1,k])

					# last line


			else:
				if j==0: #first column


				if j==columns-1:#last column

				else:#middle cases





