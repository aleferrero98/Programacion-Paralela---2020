""" Generador de numeros decimales aleatoriamente. """
from random import randint, uniform, random

CANT_DATOS = 4000000
NOMBRE_ARCHIVO = 'randomData_4M_3feature.csv'
LIM_INFERIOR = 0
LIM_SUPERIOR = 20

file = open(NOMBRE_ARCHIVO, 'w')
for i in range(0, CANT_DATOS):
    file.write(str(i) + ',' + str(uniform(0, 30)) + ',' + str(uniform(10, 15)) + ',' + str(uniform(0, 4))+ '\n')

file.close()
