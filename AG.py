import numpy as np
import cmath as cm
import math as mt
import random
from ypstruct import structure

#fitting?
def G6(x1,x2):
    res = mt.pow((x1-10), 3) + mt.pow((x2-20),3)
    return res


def restricaoG1(x1,x2):
    res = -mt.pow((x1-5),2) - mt.pow((x2-5),2) + 100
    ##print("res 1: ", res)
    if(res <= 0):
        return 1
    else:
        return 0

def restricaoG2(x1,x2):
    res = -mt.pow((x1-6),2) - mt.pow((x2-5),2) + 82.81
    ##print("res 2: ", res)
    if(res <= 0):
        return 1
    else:
        return 0

limitesG6 = np.asarray([[13, 100], [0, 100]])

##Parametros##
##numero de variavéis
N = 2
#maximo de iterações
MaxIt = 100
#tamanho da população
npop = 10
#população a primeira é gerada aleatoriamente, valores x1,x2
pop = []
popf = []
#soluções para cada x1,x2 gerado
solutions = []
#melhor solução encontrada até o momento dois campos indiviuo e solução
# problema de minimização então inicializamos com os maiores valres possiveis
bestSol = structure()
bestSol.individuo = [100,100]
bestSol.sol = np.inf
Melhor = np.empty(MaxIt)


def fitness(individuo):
    res = G6(individuo[0], individuo[1])
    #print(res)
    return res

def cruzamento(p1,p2):
    print("p1 e p2 chegando: ", p1,p2, "\n")
    c1 = p1
    c2 = p2
    alpha = np.random.uniform(0,1)
   # print("alpha: ", alpha)
    #print("c1 antes: ", c1)
    c1 = alpha*p1+(1-alpha)*p2
    c2 = alpha*p2+(1-alpha)*p1
    return c1,c2

def mutacao(x):
    print("x mutacao: ", x)
    y = x
    u = np.random.uniform(0,1)
    sigma = 1
    y = sigma*(limitesG6[1]-limitesG6[0])*(2*u)
    print(" mutacao: ", y)
    return y



def generatepopulation(): 
    for i in range(npop):
        individuo = limitesG6[:, 0] + np.random.rand(len(limitesG6)) * (limitesG6[:, 1] - limitesG6[:, 0])
        if(restricaoG1(individuo[0], individuo[1]) and restricaoG2(individuo[0], individuo[1])):
            pop.append(individuo)
            solutions.append(G6(individuo[0],individuo[1]))
            #melhor valor da primeira população
            if(bestSol.sol > solutions[i]):
                bestSol.sol = solutions[i]
                bestSol.individuo = pop[i]
        else: continue
    print(pop)




"""
def selection(pop,t):
    candidato = [None]*len(pop)
    candidato[0] = random.choices(population=pop, k=1)
    melhor = candidato[0]
    print(candidato)
    for j in range (2,4):
        candidato[j] = random.choices(pop)
        candidato[j] = np.array(candidato[j])
        print(candidato[j])
        if(G6(melhor[0], melhor[1]) > candidato[j]):
            melhor = candidato[j]
    return melhor
"""

def init():
    generatepopulation()
    print("população: ", pop,"\n")
    #for iteraçoes
    for it in range(5):
        print("Geração {}: Melhor solução = {}".format(it, bestSol))

        #for dentro da população
        for ix in range(len(pop)):
            x=1  
            #seleção p1 e p2

            #cruzamento
            p1 = pop[0]
            p2 = pop[1]
            f1, f2 = cruzamento(p1,p2)
            print("filhos: ", f1,f2,"\n")
            #mutação
            f1 = mutacao(f1)
            f2 = mutacao(f2)
            print("após mutacao", f1,f2,"\n")
            #verificar se obdece aos limites e restrições

            #Avaliar se f1 e f2 são melhores soluções que bestSol 
            #( a primeira encontrada com a população gerada aleatoriamente)

    #print(bestSol.sol)
    #print(pop[0])
    #print(pop[1])
    #print(bestSol)
   


print('hello tp2')
print(pop)
init()