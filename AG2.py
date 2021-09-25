import numpy as np
import cmath as cm
import math as mt
import random
from numpy.core.fromnumeric import mean
from ypstruct import structure
import matplotlib.pyplot as plt
def G6(x1,x2):
    res = mt.pow((x1-10), 3) + mt.pow((x2-20),3)
    return res

#problema
problema = structure()
problema.funcao = G6
problema.nVariaveis = 2
problema.limitInf = [13,0]
problema.limitSup = [100,100]

#parametros
params = structure()
params.maxIt = 10
params.npop = 1000 #aumentar depois
params.sigma = 0.1

#individuo
individuo = structure()
individuo.x1 = None
individuo.x2 = None
individuo.sol = None

#filhos
f1 = individuo.deepcopy()
f2 = individuo.deepcopy()

pop = individuo.repeat(params.npop)
#melhor intermediario
melhorIntermediario = np.empty(params.maxIt)

def restricaoG1(x1,x2):
    res = -mt.pow((x1-5),2) - mt.pow((x2-5),2) + 100
    #print("res g1: ", res)
    if(res <= 0):
        return 1
    else:
        return 0

def restricaoG2(x1,x2):
    res = -mt.pow((x1-6),2) - mt.pow((x2-5),2) + 82.81
    #print("res g2: ", res)
    if(res <= 0):
        return 1
    else:
        return 0

#inicializar população
#melhor solução
global melhorSolucao
melhorSolucao = individuo.deepcopy()
melhorSolucao.sol = np.inf

def gerarpop(melhorSolucao):
    #print(melhorSolucao)
    for i in range(params.npop):
        px = np.random.uniform(problema.limitInf[0],problema.limitSup[0])
        py = np.random.uniform(problema.limitInf[1],problema.limitSup[1])
        pop[i].x1 = px
        pop[i].x2 = py
        pop[i].sol = G6(pop[i].x1,pop[i].x2)
        if pop[i].sol < melhorSolucao.sol:
            melhorSolucao = pop[i].deepcopy()
    return melhorSolucao


def validaLimites(x):
    if((x.x1 >= problema.limitInf[0] and x.x1 <= problema.limitSup[0]) and (x.x2 >= problema.limitInf[1] and x.x2 <= problema.limitSup[1])):
        return 1 
    else:
        return 0

#cruzamento
def cruzamento(p1x1, p1x2, p2x1, p2x2):
    f1.x1 = p1x1
    f1.x2 = p1x2
    f2.x1 = p2x1
    f2.x2 = p2x2
    alpha = np.random.uniform(0,1)

    f1.x1 = alpha*p1x1+(1-alpha)*p1x2
    f1.x2 = alpha*p1x2+(1-alpha)*p1x1

    f2.x1 = alpha*p2x1+(1-alpha)*p2x2
    f2.x2 = alpha*p2x2+(1-alpha)*p2x1

#mutacao
def mutacao(x):
    y = x.deepcopy()
    u = np.random.uniform(0,1)
    sigma = 0.1
    aux = y.x1
    aux2 = y.x2
    y.x1 = sigma*(problema.limitSup[0] - problema.limitInf[0])*((2*u) - 1)
    y.x1 = y.x1 + aux
    y.x2 = sigma*(problema.limitSup[1] - problema.limitInf[1])*(2*u)
    y.x2 = y.x2 + aux2
    return y

def proporcional(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]


#main
def init(melhorSolucao,pop,f1,f2):
    print(melhorSolucao)
    melhorSolucao = gerarpop(melhorSolucao)
    beta = 1
    for i in range(params.maxIt):

        sols = np.array([x.sol for x in pop])
        avg_sol = np.mean(sols)
        if avg_sol != 0:
            sols = sols/avg_sol
        probs = np.exp(-beta*sols)
        #selecionar os pais para o cruzamento só para teste
        #print("##########################################################")
        #print("geração: ", i)
        #print("melhor solucao intermediaria: ", melhorSolucao)

        # Perform Roulette Wheel Selection
        p1 = pop[proporcional(probs)]
        p2 = pop[proporcional(probs)]


        popf = []
        for j in range(params.npop):
            #cruzamento
            cruzamento(p1.x1,p1.x2,p2.x1,p2.x2)

            #mutação
            f1 = mutacao(f1)
            f2 = mutacao(f2)
            #print("f1 após mutação: ", f1)
            #print("f2 após mutação: ", f2)
            

            #validar limites e restrições
            if(validaLimites(f1)):
                #solução para os filhos
                if(restricaoG1(f1.x1,f1.x2) and restricaoG2(f1.x1,f1.x2)):
                    f1.sol = G6(f1.x1,f1.x2)
                    if f1.sol < melhorSolucao.sol:
                        melhorSolucao = f1.deepcopy()
                    popf.append(f1)
            if(validaLimites(f2)):
                if(restricaoG1(f2.x1,f2.x2) and restricaoG2(f2.x1,f2.x2)):
                    f2.sol = G6(f2.x1,f2.x2)
                    if f2.sol < melhorSolucao.sol:
                        melhorSolucao = f2.deepcopy()
                    popf.append(f2)
        
        pop += popf
        #print("pop antes: ", pop)
        pop = sorted(pop, key=lambda x: x.sol)
        #print("depois sorted")
        pop = pop[0:params.npop]
        #print("pop depois: ", pop)
    print("melhor solução: ", melhorSolucao)
    return melhorSolucao

listB = []
'''
listB.append(init(melhorSolucao,pop,f1,f2))
print("lista b", listB)
print("lista b", listB[0].sol)
'''
aux = melhorSolucao
menorx1 = 0
menorx2 = 0
aux2 = structure
for i in range(30):
    aux2 = (init(melhorSolucao,pop,f1,f2))
    #print("aux2: ", aux2)
    listB.append(aux2.sol)
    if(aux2.sol < aux.sol):
        menorx1 = aux2.x1
        menorx2 = aux2.x2
        aux.sol = aux2.sol
     #   print("menor x1", menorx1)
#print(listB)
print("média: ",np.mean(listB))
print("minimo: ", np.min(listB))
print("maximo: ", np.max(listB))
print("desvio padrão: ", np.std(listB))
print("O menor valor encontrado foi ",aux.sol )
print("com x1 = ",menorx1)
print("e x2 = ", menorx2 )
plt.clf()
plt.boxplot(listB)
plt.title("Gráfico da configuração B")

plt.savefig("Nomedo arquivo", format='png')
