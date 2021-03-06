import numpy as np
import cmath as cm
import math as mt
import random
from ypstruct import structure
def G6(x1):
    res = mt.pow(x1,2)
    return res

#problema
problema = structure()
problema.funcao = G6
problema.nVariaveis = 2
problema.limitInf = [0]
problema.limitSup = [10]

#parametros
params = structure()
params.maxIt = 10
params.npop = 100 #aumentar depois
params.sigma = 0.1

#individuo
individuo = structure()
individuo.x1 = None
individuo.sol = None

#filhos
f1 = individuo.deepcopy()
f2 = individuo.deepcopy()



pop = individuo.repeat(params.npop)
#melhor intermediario
melhorIntermediario = np.empty(params.maxIt)


#inicializar população
#melhor solução
global melhorSolucao
melhorSolucao = individuo.deepcopy()
melhorSolucao.sol = np.inf

def gerarpop(melhorSolucao):
    #print(melhorSolucao)
    for i in range(params.npop):
        px = np.random.uniform(problema.limitInf[0],problema.limitSup[0])
        pop[i].x1 = px
        pop[i].sol = G6(pop[i].x1)
        if pop[i].sol < melhorSolucao.sol:
            melhorSolucao = pop[i].deepcopy()
    return melhorSolucao


def validaLimites(x):
    if((x.x1 >= problema.limitInf[0] and x.x1 <= problema.limitSup[0])):
        return 1 
    else:
        return 0

#cruzamento
def cruzamento(p1x1, p2x1):
    f1.x1 = p1x1
    f2.x1 = p2x1
    alpha = np.random.uniform(0,1)

    f1.x1 = alpha*p1x1+(1-alpha)*p2x1

    f2.x1 = alpha*p2x1+(1-alpha)*p1x1

#mutacao
def mutacao(x):
    y = x.deepcopy()
    u = np.random.uniform(0,1)
    sigma = 0.1
    aux = y.x1
    y.x1 = sigma*(problema.limitSup[0] - problema.limitInf[0])*((2*u) - 1)
    y.x1 = y.x1 + aux
    return y

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]


#main
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
    print("##########################################################")
    print("geração: ", i)
    print("melhor solucao intermediaria: ", melhorSolucao)

    # Perform Roulette Wheel Selection
    p1 = pop[roulette_wheel_selection(probs)]
    p2 = pop[roulette_wheel_selection(probs)]


    popf = []
    for j in range(params.npop):
        #cruzamento
        cruzamento(p1.x1,p2.x1)

        #mutação
        f1 = mutacao(f1)
        f2 = mutacao(f2)
        #print("f1 após mutação: ", f1)
        #print("f2 após mutação: ", f2)
        

        #validar limites e restrições
        if(validaLimites(f1)):
            #solução para os filhos
            
            f1.sol = G6(f1.x1)
            if f1.sol < melhorSolucao.sol:
                melhorSolucao = f1.deepcopy()
            popf.append(f1)
        if(validaLimites(f2)):
           
            f2.sol = G6(f2.x1)
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

