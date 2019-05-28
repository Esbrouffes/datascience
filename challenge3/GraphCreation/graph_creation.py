import csv
import numpy
import networkx as nx

popu_cells = numpy.array(list(csv.reader(open("/Users/ricardorodriguez/Documents/Ciclo2019_1/DataScience/Challenge3/DataPreparation/popu_cells.csv", "r"), delimiter=","))).astype("double")
elevation_cells = numpy.array(list(csv.reader(open("/Users/ricardorodriguez/Documents/Ciclo2019_1/DataScience/Challenge3/DataPreparation/elevation_cells.csv", "r"), delimiter=","))).astype("double")

print((elevation_cells.shape))
