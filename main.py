import random
def branchandbound(weight,profit,m,n):
    class Node:

        def __init__(self, data,status, queue):
            self.data = data
            self.status=status
            self.queue=queue
        def setQueue(self,queue):
            self.queue=queue
        def getQueue(self,queue):
            return self.queue
    
    def included(weight,profit,m,n,queue):
        c = 0
        currentweight=0
        u=0
        for x in range(len(queue)):
            currentweight= currentweight + queue[x]*weight[x]
            c = c+ profit[x]*queue[x]
        if weight[len(queue)] <= m - currentweight:
            c = c + profit [len(queue)]
            currentweight = currentweight + weight[len(queue)]
        else:
            c = c+ (m - currentweight)  * profit [x]/weight[x]
        for x in range(len(queue)+1,len(weight)):
            if weight[x] <= m - currentweight:
                c = c+ profit[x]
                currentweight = currentweight + weight[x]
            else :
              c = c+ (m - currentweight)  * profit [x]/weight[x]
        #calculate u 
        currentweight=0
        for x in range(len(queue)):
            currentweight= currentweight + queue[x]*weight[x]
            u = u+ profit[x]*queue[x]
        if weight[len(queue)] <= m - currentweight:
            u = u + profit [len(queue)]
            currentweight = currentweight + weight[len(queue)]
        for x in range(len(queue)+1,n):
            if weight[x] <= m - currentweight:
                u = u+ profit[x]
                currentweight = currentweight + weight[x]
        return [-u,-c]
    def notincluded(weight,profit,m,n,queue):
        c = 0
        currentweight=0
        u=0
        for x in range(len(queue)):
            currentweight= currentweight + queue[x]*weight[x]
            c = c+ profit[x]*queue[x]
        # if weight[len(queue)] <= m - currentweight:
        #     c = c + profit [len(queue)]
        #     currentweight = currentweight + weight[len(queue)]
        # else:
        #     c = c+ (m - currentweight)  * profit [x]/weight[x]
        for x in range(len(queue)+1,len(weight)):
            if weight[x] <= m - currentweight:
                c = c+ profit[x]
                currentweight = currentweight + weight[x]
            else :
              c = c+ (m - currentweight)  * profit [x]/weight[x]
        #calculate u 
        currentweight=0
        for x in range(len(queue)):
            currentweight= currentweight + queue[x]*weight[x]
            u = u+ profit[x]*queue[x]
        # if weight[len(queue)] <= m - currentweight:
        #     u = u + profit [len(queue)]
        #     currentweight = currentweight + weight[len(queue)]
        for x in range(len(queue)+1,n):
            if weight[x] <= m - currentweight:
                u = u+ profit[x]
                currentweight = currentweight + weight[x]
        return [-u,-c]
    upper = 9999 #infinity
    root = Node(included(weight,profit,m,n,[]),1,[])
    tree = []
    tree.append(root)
    indxplore=0
    found = False
    while found == False:
         #calculate c
            leftnode= Node(included(weight,profit,m,n,list(tree[indxplore].queue)),1,list(tree[indxplore].queue))
            tree.append(leftnode)   
            leftnode.queue.append(1)
            if leftnode.data[0] < upper:
                upper = leftnode.data[0]
                for x in range(len(tree)):
                    if (tree[x].data[1]>upper):
                        tree[x].status = int(0)

            rightnode = Node(notincluded(weight,profit,m,n,list(tree[indxplore].queue)),1,list(tree[indxplore].queue))
            tree.append(rightnode)
            rightnode.queue.append(0)
            if rightnode.data[0] <upper:
                upper = rightnode.data[0]
                for x in tree:
                    if (x.data[1]>upper):
                        x.status = int(0)
            for x in tree:
                if x.status == 0:
                    tree.remove(x)
            tree.pop(indxplore) #remove current item from list
            mincost = tree[0].data[1]
            minindx = 0
            for x in range(len(tree)-1): #change index to explore other nodes
                if tree[x].data[1] < mincost:
                    mincost = tree[x].data[1]
                    minindx = x
            indxplore = minindx
            if len(tree[indxplore].queue) == n :
                return tree[indxplore].queue
def bruteforce(weight,profit,m,n):
    combinations = (2**n)-1
    maxprofit = 0
    itemslist=[0,0,0,0]
    for x in range(combinations+1):
        binarytruthtable = "{0:b}".format(x)
        if len(binarytruthtable) != n:
            while len(binarytruthtable) != n:
                binarytruthtable = "0"+ binarytruthtable
        score = list(map(int,list(binarytruthtable)))
        weightcalc = 0
        currentprofit = 0
        for y in range(n):
            currentprofit = currentprofit + score[y]*profit[y]
            weightcalc = weightcalc + score[y]*weight[y]
        if weightcalc <= m and currentprofit > maxprofit:
            itemslist = list(score)
            maxprofit = currentprofit
    return itemslist
def DynamicProgramming(weight,profit,m,n): 
    Memory = [[0 for x in range(m + 1)] for x in range(n + 1)] 
    # Build table Memory[][] in bottom up  
    for x in range(n + 1): 
        for weights in range(m + 1): 
            if x == 0 or weights == 0: 
                Memory[x][weights] = 0
            elif weight[x-1] <= weights: 
                Memory[x][weights] = max(profit[x-1] + Memory[x-1][weights-weight[x-1]],  Memory[x-1][weights]) 
            else: 
                Memory[x][weights] = Memory[x-1][weights] 
  
    return Memory[n][m] 
def DivideAndConquer(weight,profit,m,n): 
    if n == 0 or m == 0 : 
        return 0
    if (weight[n-1] > m): 
        return DivideAndConquer(weight,profit,m, n-1) 
    else: 
        return max(profit[n-1] + DivideAndConquer(weight, profit,m-weight[n-1], n-1), 
                   DivideAndConquer(weight, profit,m, n-1)) 
def Genetic(weight,profit,m,n):
    class Item(object):
        def __init__(self, profit, weight):
            self.profit = profit
            self.weight = weight
    items = []
    for x in range(len(weight)):
      items.append(Item(profit[x],weight[x]))
    PopulationSize = 50
    StartPopZero = False
    def fitness(target):
        Total = 0
        currentweight = 0
        index = 0
        for i in target:        
            if index >= len(items):
                break
            if (i == 1):
                Total += items[index].profit
                currentweight += items[index].weight
            index += 1
        if currentweight <= m:
            return Total
        else:
            return 0

    def Spawn(amount):
        return [RandomIndividual() for x in range (0,amount)]

    def RandomIndividual():
        if StartPopZero:
            return [random.randint(0,0) for x in range (0,len(items))]
        else:
            return [random.randint(0,1) for x in range (0,len(items))]

    def mutate(target):
        rand = random.randint(0,len(target)-1)
        if target[rand] == 1:
            target[rand] = 0
        else:
            target[rand] = 1

    def EvolveKnapsack(pop):
        ParentChance = 0.2
        MutationProbability = 0.08
        parentSize = int(ParentChance*len(pop))
        parent = pop[:parentSize]
        for p in parent:
            if MutationProbability > random.random():
                mutate(p)

        #Crossover function
        children = []
        Length = len(pop) - len(parent)
        while len(children) < Length :
            male = pop[random.randint(0,len(parent)-1)]
            female = pop[random.randint(0,len(parent)-1)]        
            half = int(len(male)/2)
            child = male[:half] + female[half:] # from start to half from father, from half to end from mother
            if MutationProbability > random.random():
                mutate(child)
            children.append(child)

        parent.extend(children)
        return parent
    population = Spawn(PopulationSize)
    population = sorted(population, key=lambda x: fitness(x), reverse=True)
    population = EvolveKnapsack(population)
    #generation += 1
    return population[0]
def Greedy(weight,profit,m,n):
    value = []
    for x in range(len(weight)):
        value.append([profit[x]/weight[x],x])
    value.sort(key=lambda x: x[0],reverse=True)
    order= [value[x][1] for x in range(len(weight))]
    currentweight=0
    maxprofit=0
    for x in range(len(weight)):
        if currentweight + weight[order[x]] <=m:
            maxprofit = maxprofit + profit[order[x]]
            currentweight = currentweight + weight[order[x]]
    return maxprofit
def Backtracking(weight,profit, m, n): 
    memory = [[0 for weights in range(m + 1)] for x in range(n + 1)] 
    for x in range(n + 1): 
        for weights in range(m + 1): 
            if x == 0 or weights == 0: 
                memory[x][weights] = 0
            elif weight[x - 1] <= weights: 
                memory[x][weights] = max(profit[x - 1]  
                  + memory[x - 1][weights - weight[x - 1]], 
                               memory[x - 1][weights]) 
            else: 
                memory[x][weights] = memory[x - 1][weights] 
  
    answer = memory[n][m] 
    maxprofit = answer
    weights = m 
    for x in range(n, 0, -1): 
        if answer <= 0: 
            break

        if answer == memory[x - 1][weights]: 
            continue
        else: 
            weights = weights - weight[x - 1] 
            answer = answer - profit[x - 1] 

    return maxprofit

        



weight = [2,4,6,9] #weight of each item
profit = [10,10,12,18] #profit or value of each item
m = 15 #knapsack weight
n = len(weight) #number of items

print("branchandbound", branchandbound(weight,profit,m,n))
print("BruteForce", bruteforce(weight,profit,m,n))
print("DynamicProgramming" , DynamicProgramming(weight,profit,m,n))
print("DivideAndConquer" , DivideAndConquer(weight,profit,m,n))
print("Genetic" , Genetic(weight,profit,m,n))
print("Greedy", Greedy(weight,profit,m,n))
print("Backtracking", Backtracking(weight,profit,m,n))