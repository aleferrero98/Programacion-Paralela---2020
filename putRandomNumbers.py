""" Agrega numeros decimales aleatoriamente como una columna mas de un archivo .csv """
from random import randint, uniform, random

ARCHIVO_LEER = 'inputs/movisB.csv'
ARCHIVO_ESCRIBIR = 'movisB_2feature.csv'
LIM_INFERIOR = 0
LIM_SUPERIOR = 4

file_rd = open(ARCHIVO_LEER, 'r')
file_wr = open(ARCHIVO_ESCRIBIR, 'w')

for line in file_rd:
    elementos = line.split(sep='\n')[0].split(sep=',')
    if((elementos[0] != '') and (elementos[1] != '')):
        #print(float(elementos[0]), float(elementos[1]))
        file_wr.write(str(elementos[0]) + ',' + str(elementos[1]) + ',' + str(uniform(LIM_INFERIOR, LIM_SUPERIOR)) + '\n')

file_rd.close()
file_wr.close()