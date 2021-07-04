""" Generador de numeros decimales aleatoriamente. """
from random import randint, uniform, random

CANT_DATOS = 10000000
NOMBRE_ARCHIVO = 'randomData.csv'
LIM_INFERIOR = 0
LIM_SUPERIOR = 20

file = open(NOMBRE_ARCHIVO, 'w')
for i in range(0, CANT_DATOS):
    file.write(str(i) + ',' + str(uniform(LIM_INFERIOR, LIM_SUPERIOR)) + '\n')

file.close()
