import numpy as np
import gc
import random
import time
from sklearn.tree import DecisionTreeClassifier
from decimal import Decimal, getcontext
getcontext().prec = 3

num_tasks = 6
divisors = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 40]
#L = .4
#B = .6
#W = .7
L = .6
B = .9
W = 1.15

hyper_parameter = 2400
iterations = 100

utilityMod = 0   # 0 for linear drop, 1 for downward step, 2 for target sensitive
results = []

priods = []
utility = []
varL = []
varB = []
varW = []
deadline = []
c = []

samp = []
labl = []

rowStarts = []
choiceStarts = []
cols = []
nnz = []
rew = []

rowStarts2 = []
choiceStarts2 = []
cols2 = []
nnz2 = []
rew2 = []

optPolicy = []
optPolicy2 = []

greedyPolicy = []
EDFPolicy = []
RMPolicy = []
pseudoPolicy = []
UPAPolicy = []
randDTPolicy = []
prioritizedDTPolicy = []

rtVI = 0
rtPseudo = 0
rtGreedy = 0
rtReducedMDP = 0
rtPrioReducedMDP = 0
rtRndDT = 0
rtPrioDT = 0
rtEDF = 0
rtRM = 0
rtUPA = 0

prVI = 0
prPseudo = 0
prGreedy = 0
prReducedMDP = 0
prPrioReducedMDP = 0
prRndDT = 0
prPrioDT = 0
prEDF = 0
prRM = 0
prUPA = 0

DTnodes = []
PrioDTnodes = []

original_num_state = 0
reduced1_num_state = 0
reduced2_num_state = 0
original_num_trans = 0
reduced1_num_trans = 0
reduced2_num_trans = 0

maxActs = 3	# maximum number of actions per state for the first heuristic


def UtilityFunction(i,j,t):
    if utilityMod == 0:
        if (t+j) // priods[i] > t // priods[i] or (t+j) % priods[i] > deadline[i]:
            rw = 0
        elif (t+j) % priods[i] <= c[i]:
            rw = utility[i]
        else:
            rw = utility[i] * (1 - ((t+j) % priods[i] - c[i])/(deadline[i] - c[i]))
    elif utilityMod == 1:
        if (t+j) // priods[i] > t // priods[i] or (t+j) % priods[i] > deadline[i]:
            rw = 0
        elif (t+j) % priods[i] <= c[i]:
            rw = utility[i]
    elif utilityMod == 2:
        if (t+j) // priods[i] > t // priods[i] or (t+j) % priods[i] > deadline[i]:
            rw = 0
        elif (t+j) % priods[i] <= c[i]:
            rw = utility[i] * ((t+j) % priods[i]) / c[i]
        else:
            rw = utility[i] * (1 - ((t+j) % priods[i] - c[i])/(deadline[i] - c[i]))
        
    return rw

def computetrans(t, rq):
    length = int(pow(2,num_tasks))
    state = t * length + rq
    queue = []
    global ind1
    global ind2
    for i in range(num_tasks):
        if rq % 2 == 0:
            queue.append(0)
        else:
            queue.append(1)
        rq = rq // 2

    for i in range(num_tasks):			# each task is an action
        if queue[i] == False or (t+varL[i]) % priods[i] > deadline[i] or ((t+varL[i]) // priods[i] > t // priods[i]):
            choiceStarts.append(ind2)
            ind1 += 1
            continue
        sumpr = 0
        for j in range(varL[i],varB[i]+1):
            prob = .8/(varB[i] - varL[i]+1)
            sumpr += prob
            newq = []
            nqcode = 0
            new_t = (t + j) % hyper_parameter
            for k in range(num_tasks):
                if i == k:
                    if (t // priods[i] == new_t // priods[i]) and (new_t % priods[i] < varW[i]) or (t + j >= hyper_parameter):
                        nqcode += int(pow(2,k))
                    else:
                        pass#newq[k] = 0
                else:
                    if new_t % priods[k] >= deadline[k]:
                        pass#newq[k] = 0
                    elif (t // priods[k] < new_t // priods[k]) or (t + j >= hyper_parameter):
                        nqcode += int(pow(2,k))
                    elif queue[k] == 1:
                        nqcode += int(pow(2,k))
            is_state[new_t][nqcode] = True
            cols.append(new_t*length + nqcode)
            nnz.append(prob)
            rw = UtilityFunction(i,j,t)
            
#            if (t+j) // priods[i] > t // priods[i] or (t+j) % priods[i] > deadline[i]:
#                rw = 0
#            elif (t+j) % priods[i] <= c[i]:
#                rw = utility[i]
#            else:
#                rw = utility[i] * (1 - ((t+j) % priods[i] - c[i])/(deadline[i] - c[i]))
            rew.append(rw)
            ind2 += 1
        for j in range(varB[i]+1, varW[i]+1):
            prob = .2/(varW[i] - varB[i])
            sumpr += prob
            nqcode = 0
            new_t = (t + j) % hyper_parameter
            for k in range(num_tasks):
                if i == k:
                    if (t // priods[i] == new_t // priods[i]) and (new_t % priods[i] < varW[i]) or (t + j >= hyper_parameter):
                        nqcode += int(pow(2,k))
                    else:
                        pass#newq[k] = 0
                else:
                    if new_t % priods[k] >= deadline[k]:
                        pass#newq[k] = 0
                    elif (t // priods[k] < new_t // priods[k]) or (t + j >= hyper_parameter):
                        nqcode += int(pow(2,k))
                    elif queue[k] == 1:
                        nqcode += int(pow(2,k))
            is_state[new_t][nqcode] = True
            cols.append(new_t*length + nqcode)
            nnz.append(prob)
            rw = 0
            if (t+j) // priods[i] > t // priods[i] or (t+j) % priods[i] > deadline[i]:
                rw = 0
            elif (t+j) % priods[i] <= c[i]:
                rw = utility[i]
            else:
                rw = utility[i] * (1 - ((t+j) % priods[i] - c[i])/(deadline[i] - c[i]))
            rew.append(rw)
            ind2 += 1
        choiceStarts.append(ind2)
        ind1 += 1

    new_t = (t + 1) % hyper_parameter
    nqcode = 0
    for k in range(num_tasks):
        if t + 1 == hyper_parameter:
            nqcode += int(pow(2,k))
        elif new_t % priods[k] >= deadline[k]:
            pass#newq[k] = 0
        elif (t // priods[k] < (t+1) // priods[k]):
            nqcode += int(pow(2,k))
        elif queue[k] == 1:
            nqcode += int(pow(2,k))
    is_state[new_t][nqcode] = True
    cols.append(new_t*length + nqcode)
    nnz.append(1)
    rew.append(0)

    ind2 += 1
    choiceStarts.append(ind2)
    ind1 += 1
    rowStarts.append(ind1)

def mapstate(st):		# maps a given state from the origimal MDP to a corresponding in the reduced MDP
    length = int(pow(2, num_tasks))
    t = st // length
    rem = st % length
    tsklist = []
    ones = 0
    for i in range(num_tasks):
        if rem % 2 == 1:
            tsklist.append(1)
            ones += 1
        else:
            tsklist.append(0)
        rem = rem // 2
    if ones <= maxActs:
        return st
    rem = 0
    selecteds = 0
    i = 0
    while selecteds < maxActs:
        if tsklist[uts[i][0]] == 1:
            rem += int(pow(2,uts[i][0]))
            selecteds += 1
        i += 1
    return t * length + rem

def computetrans2(t, rq):
    length = int(pow(2,num_tasks))
    state = t * length + rq
    queue = []
    global ind1
    global ind2
    for i in range(num_tasks):
        if rq % 2 == 0:
            queue.append(0)
        else:
            queue.append(1)
        rq = rq // 2

    for i in range(num_tasks):			# each task is an action
        if queue[i] == False or (t+varL[i]) % priods[i] > deadline[i] or ((t+varL[i]) // priods[i] > t // priods[i]):
            choiceStarts2.append(ind2)
            ind1 += 1
            continue
        sumpr = 0
        for j in range(varL[i],varB[i]+1):
            prob = .8/(varB[i] - varL[i]+1)
            sumpr += prob
            newq = []
            nqcode = 0
            new_t = (t + j) % hyper_parameter
            for k in range(num_tasks):
                if i == k:
                    if (t // priods[i] == new_t // priods[i]) and (new_t % priods[i] < varW[i]) or (t + j >= hyper_parameter):
                        nqcode += int(pow(2,k))
                    else:
                        pass#newq[k] = 0
                else:
                    if new_t % priods[k] >= deadline[k]:
                        pass#newq[k] = 0
                    elif (t // priods[k] < new_t // priods[k]) or (t + j >= hyper_parameter):
                        nqcode += int(pow(2,k))
                    elif queue[k] == 1:
                        nqcode += int(pow(2,k))
            mpst = mapstate(new_t*length + nqcode)		# corresponding state in reduced MDP
            nqcode2 = mpst % length				# at most 3 tasks can be in ready queue
            is_state2[new_t][nqcode2] = True			#
            cols2.append(mpst)
            nnz2.append(prob)
            rw = UtilityFunction(i,j,t)
#            if (t+j) // priods[i] > t // priods[i] or (t+j) % priods[i] > deadline[i]:
#                rw = 0
#            elif (t+j) % priods[i] <= c[i]:
#                rw = utility[i]
#            else:
#                rw = utility[i] * (1 - ((t+j) % priods[i] - c[i])/(deadline[i] - c[i]))
            rew2.append(rw)
            ind2 += 1
        for j in range(varB[i]+1, varW[i]+1):
            prob = .2/(varW[i] - varB[i])
            sumpr += prob
            nqcode = 0
            new_t = (t + j) % hyper_parameter
            for k in range(num_tasks):
                if i == k:
                    if (t // priods[i] == new_t // priods[i]) and (new_t % priods[i] < varW[i]) or (t + j >= hyper_parameter):
                        nqcode += int(pow(2,k))
                    else:
                        pass#newq[k] = 0
                else:
                    if new_t % priods[k] >= deadline[k]:
                        pass#newq[k] = 0
                    elif (t // priods[k] < new_t // priods[k]) or (t + j >= hyper_parameter):
                        nqcode += int(pow(2,k))
                    elif queue[k] == 1:
                        nqcode += int(pow(2,k))
            mpst = mapstate(new_t*length + nqcode)		# corresponding state in reduced MDP
            nqcode2 = mpst % length				# at most 3 tasks can be in ready queue
            is_state2[new_t][nqcode2] = True			#
            cols2.append(mpst)
            nnz2.append(prob)
            rw = 0
            if (t+j) // priods[i] > t // priods[i] or (t+j) % priods[i] > deadline[i]:
                rw = 0
            elif (t+j) % priods[i] <= c[i]:
                rw = utility[i]
            else:
                rw = utility[i] * (1 - ((t+j) % priods[i] - c[i])/(deadline[i] - c[i]))
            rew2.append(rw)
            ind2 += 1
        choiceStarts2.append(ind2)
        ind1 += 1

    new_t = (t + 1) % hyper_parameter
    nqcode = 0
    for k in range(num_tasks):
        if t + 1 == hyper_parameter:
            nqcode += int(pow(2,k))
        elif new_t % priods[k] >= deadline[k]:
            pass#newq[k] = 0
        elif (t // priods[k] < (t+1) // priods[k]):
            nqcode += int(pow(2,k))
        elif queue[k] == 1:
            nqcode += int(pow(2,k))

    mpst = mapstate(new_t*length + nqcode)		# corresponding state in reduced MDP
    nqcode2 = mpst % length				# at most 3 tasks can be in ready queue
    is_state2[new_t][nqcode2] = True			#

    cols2.append(mpst)
    nnz2.append(1)
    rew2.append(0)

    ind2 += 1
    choiceStarts2.append(ind2)
    ind1 += 1
    rowStarts2.append(ind1)

def isvalidstate(t, q):
    for i in range(num_tasks):
        if q % 2 == 1 and t % priods[i] > deadline[i]:
            return False
        q = q // 2
    return True

def reducedMDP():
    maxs = 0
    for i in range(hyper_parameter):
        s = 0
        for j in range(int(pow(2,num_tasks))):
            if is_state2[i][j] == True:
                s += 1
            
        for j in range(int(pow(2,num_tasks))):
            if is_state[i][j] == True:
                s += 1
                if (s < num_tasks * (num_tasks-1) * (num_tasks-2) /12 or i < hyper_parameter * .3):
                    sttmp = mapstate(i * int(pow(2,num_tasks)) + j)
                    if is_state2[i][sttmp % int(pow(2,num_tasks))] == False:
                        is_state2[i][sttmp % int(pow(2,num_tasks))] = True
        if s > maxs:
            maxs = s
    for i in range(hyper_parameter):
        for j in range(int(pow(2,num_tasks))):
            if is_state2[i][j] == True:
                sttmp = mapstate(i * int(pow(2,num_tasks)) + j)
                computetrans2(i, sttmp % int(pow(2,num_tasks)))
            else:
                rowStarts2.append(ind1)

                
def PrioreducedMDP():    
    for i in range(hyper_parameter):
        for j in range(int(pow(2,num_tasks))):
            if is_state[i][j] == True:
                sttmp = mapstate(i * int(pow(2,num_tasks)) + j)
                if is_state2[i][sttmp % int(pow(2,num_tasks))] == False:
                    is_state2[i][sttmp % int(pow(2,num_tasks))] = True
                    
    for i in range(hyper_parameter):
        for j in range(int(pow(2,num_tasks))):
            if is_state2[i][j] == True:
                sttmp = mapstate(i * int(pow(2,num_tasks)) + j)
                if sttmp != i * int(pow(2,num_tasks)) + j:
                    computetrans2(i, j)
                else:
                    computetrans2(i, sttmp % int(pow(2,num_tasks)))
            else:
                rowStarts2.append(ind1)

    init_st = int(pow(2,num_tasks)-1)
    res = []
    X = []
    y = []
    freq = np.zeros(len(rowStarts))
    for rounds in range(40):
        s = init_st
        for step in range(500):
            i = s
            if freq[i] == 5 and rowStarts[i+1] > rowStarts[i]+1:
                X.append(i)
                freq[i] += 1
                break

            freq[i] += 1
            bestVal = 0
            bestAct = rowStarts[i+1] - 1
            for j in range(rowStarts[i],rowStarts[i+1] - 1):
                if(choiceStarts[j] < choiceStarts[j+1]):
                    if utility[j - rowStarts[i]] / varW[j - rowStarts[i]] > bestVal:
                        bestVal = utility[j - rowStarts[i]] / varW[j - rowStarts[i]]
                        bestAct = j
            j = bestAct
            lngth = (choiceStarts[j+1] - choiceStarts[j])
            cl = choiceStarts[j] + int(random.random() * lngth)
            s = cols[cl]
    return X    

                
def simulate_run(st, action, gamma, dpt, runs, policy):
    accum_rew = 0
    j = rowStarts[st] + action
    if choiceStarts[j+1] == choiceStarts[j]:
        return 0
    for i in range(runs):
        act = action
        stt = st
        factor = 1
        for iters in range(dpt):
            j = rowStarts[stt] + act
            dif = choiceStarts[j+1] - choiceStarts[j]
            k = choiceStarts[j] + ((int)(random.random()*1000)) % dif
            factor = factor * gamma
            accum_rew += nnz[k] * rew[k] * factor
            stt = cols[k]
            act = policy[stt]
    return accum_rew/runs

def simulate_run_UPA(st, action_list, gamma, dpt):
    rem = st % (int)((2**num_tasks))
    jobs = []
    for i in range(num_tasks):
        jobs.append(rem%2)
        rem = (int)(rem // 2)
    action = num_tasks
    jobs.reverse()
    for k in action_list:
        if jobs[k] > 0:
            action = k
            break
    
    accum_rew = 0
    j = rowStarts[st] + action
    if choiceStarts[j+1] == choiceStarts[j]:
        return 0

    for i in range(50):
        act = action
        stt = st
        factor = 1
        for iters in range(dpt):
            j = rowStarts[stt] + act
            dif = choiceStarts[j+1] - choiceStarts[j]
            k = choiceStarts[j] + ((int)(random.random()*1000)) % dif
            factor = factor * gamma
            accum_rew += nnz[k] * rew[k] * factor
            stt = cols[k]

            rem = stt % (int)((2**num_tasks))
            jobs = []
            for j in range(num_tasks):
                jobs.append(rem%2)
                rem = (int)(rem // 2)
            act = num_tasks
            jobs.reverse()
            for k in action_list:
                if jobs[k] > 0:
                    act = k
                    break
            
    return accum_rew/50


def valiter(gamma, epsilon):
    bestAct = np.zeros(len(rowStarts), dtype = int)
    for iters in range(4000):
        diff = 0
        for i in range(len(rowStarts)-2,0,-1):
            bestAct[i] = -1
            d1 = 0
            for j in range(rowStarts[i],rowStarts[i+1]):
                d2 = 0
                for k in range(choiceStarts[j], choiceStarts[j+1]):
                    d2 += gamma * nnz[k] * state_val[cols[k]] + nnz[k] * rew[k]                        
                if d2 > d1:
                    d1 = d2
                    bestAct[i] = j - rowStarts[i]
            if(d1 - state_val[i] > diff):
                diff = d1 - state_val[i]
            state_val[i] = d1
        if diff < epsilon:
            break
    for i in range(len(rowStarts)):
        optPolicy.append(bestAct[i])


def DTMCcalc(gamma, epsilon, policy):
    for iters in range(4000):
        diff = 0
        for i in range(len(rowStarts)-2,0,-1):
            d1 = 0
            if policy[i] == -1 or rowStarts[i+1] == rowStarts[i]:
                continue
            j = rowStarts[i] + policy[i]
            for k in range(choiceStarts[j], choiceStarts[j+1]):
                d1 += gamma * nnz[k] * state_val[cols[k]] + nnz[k] * rew[k]
            if(d1 - state_val[i] > diff):
                diff = d1 - state_val[i]
            state_val[i] = d1
        if diff < epsilon:
            break

def greedy():
    for i in range(len(rowStarts)-1):
        bestAct = -1
        bestVal = 0
        if rowStarts[i] < rowStarts[i+1]:
            bestAct = num_tasks
        for j in range(rowStarts[i],rowStarts[i+1] - 1):
            if(choiceStarts[j] < choiceStarts[j+1]):
                if utility[j - rowStarts[i]] / varW[j - rowStarts[i]] > bestVal:
                    bestVal = utility[j - rowStarts[i]] / varW[j - rowStarts[i]]
                    bestAct = j - rowStarts[i]
        greedyPolicy.append(bestAct)

def EDF():
    for i in range(len(rowStarts)-1):
        length = int(pow(2, num_tasks))
        t = i // length
        rem = i % length

        bestAct = -1
        bestVal = 90000
        if rowStarts[i] < rowStarts[i+1]:
            bestAct = num_tasks
        for j in range(rowStarts[i],rowStarts[i+1] - 1):
            if(choiceStarts[j] < choiceStarts[j+1]):
                ttt = t % priods[j - rowStarts[i]]
                if bestAct == num_tasks and ttt + varL[j - rowStarts[i]] <= deadline[j - rowStarts[i]]:
                    bestAct = j - rowStarts[i]
                elif ttt + varW[j - rowStarts[i]] <= deadline[j - rowStarts[i]] and deadline[j - rowStarts[i]] - (ttt + varW[j - rowStarts[i]]) < bestVal:
                    bestVal = deadline[j - rowStarts[i]] - (ttt + varW[j - rowStarts[i]])
                    bestAct = j - rowStarts[i]
        EDFPolicy.append(bestAct)

def RM():
    for i in range(len(rowStarts)-1):
        length = int(pow(2, num_tasks))
        t = i // length
        rem = i % length

        bestAct = -1
        bestVal = 90000
        if rowStarts[i] < rowStarts[i+1]:
            bestAct = num_tasks
        for j in range(rowStarts[i],rowStarts[i+1] - 1):
            if(choiceStarts[j] < choiceStarts[j+1]):
                ttt = priods[j - rowStarts[i]]
                if bestAct == num_tasks and ttt < bestVal:
                    bestAct = j - rowStarts[i]
                    bestVal = ttt
        RMPolicy.append(bestAct)

def pseudo():
    for i in range(len(rowStarts)-1):
        bestAct = -1
        bestVal = 0
        if rowStarts[i] < rowStarts[i+1]:
            bestAct = num_tasks
        for j in range(rowStarts[i],rowStarts[i+1] - 1):
            if(choiceStarts[j] < choiceStarts[j+1]):
                if utility[j - rowStarts[i]] / deadline[j - rowStarts[i]] > bestVal:
                    bestVal = utility[j - rowStarts[i]] / deadline[j - rowStarts[i]]
                    bestAct = j - rowStarts[i]
        pseudoPolicy.append(bestAct)

def UPA():    
    for i in range(len(rowStarts)-1):
        job_list = []

        for j in range(rowStarts[i],rowStarts[i+1] - 1):
            if(choiceStarts[j] < choiceStarts[j+1]):
                job_list.append((j - rowStarts[i], utility[j - rowStarts[i]] / deadline[j - rowStarts[i]]))
            else:
                job_list.append((j - rowStarts[i], 0))
        for j in range(len(job_list)):
            for k in range(len(job_list)-1):
                if(job_list[k][1] < job_list[k+1][1]):
                    tmp = job_list[k]
                    job_list[k] = job_list[k+1]
                    job_list[k+1] = tmp

        if len(job_list) > 50 and job_list[0][1] > 0:
            action_list = []
            for j in range(len(job_list)):
                action_list.append(job_list[j][0])

            simulate_run_UPA(i, action_list, .99, 10)
                    
        if (i % 3) >= 1 and len(job_list) > 0 and job_list[0][1] > 0:
            bestAct = job_list[0][0]
        else:
            bestAct = optPolicy[i]
                        
        UPAPolicy.append(bestAct)        
        
def randdt(K):
    states = []
    features = []
    X = []
    y = []
    pw = int(pow(2, num_tasks))
    for i in range(len(rowStarts)-1):
        rw = []
        rw.append(i // pw)
        k = i % pw
        for j in range(num_tasks):
            rw.append(k%2)
            k = k // 2
        features.append(rw)
    i = 0
    while i < K:
        st = int(random.random()*(len(rowStarts)-2))
        if rowStarts[st+1] > rowStarts[st]+1:
            X.append(features[st])
            val = 0
            best_val = 0
            best_act = 0
            for j in range(rowStarts[st],rowStarts[st+1]):
                if(choiceStarts[j] < choiceStarts[j+1]):
                    val = simulate_run(st, j - rowStarts[st], .99, 120, 200, pseudoPolicy)
                    if val > best_val:
                        best_val = val
                        best_act = j - rowStarts[st]
            y.append(best_act)
            i += 1
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X,y)
    DTnodes.append(clf.tree_.node_count)
    
    y_pred = clf.predict(features)
    
    for i in range(len(rowStarts)-1):
        if rowStarts[i+1] == rowStarts[i]:
            randDTPolicy.append(0)
        else:
            k = rowStarts[i] + y_pred[i]
            if choiceStarts[k+1] == choiceStarts[k]:	# Suggested action does not exist! -- is not valid
                randDTPolicy.append(0)
            else:
                randDTPolicy.append(y_pred[i])

def valiter2(gamma, epsilon):
    bestAct2 = np.zeros(len(rowStarts2), dtype = int)
    for iters in range(4000):
        diff = 0
        for i in range(len(rowStarts2)-2,0,-1):
            bestAct2[i] = -1
            d1 = 0
            for j in range(rowStarts2[i],rowStarts2[i+1]):
                d2 = 0
                for k in range(choiceStarts2[j], choiceStarts2[j+1]):
                    d2 += gamma * nnz2[k] * state_val2[cols2[k]] + nnz2[k] * rew2[k]
                if d2 > d1:
                    d1 = d2
                    bestAct2[i] = j - rowStarts2[i]
            if(d1 - state_val2[i] > diff):
                diff = d1 - state_val2[i]
            state_val2[i] = d1
        if diff < epsilon:
            break
    for i in range(len(rowStarts2)):
        optPolicy2.append(bestAct2[i])
    
def state_prioritizing_dt(K):
    k0 = K
    states = []
    features = []
    X = []
    y = []
    pw = int(pow(2, num_tasks))
    for i in range(len(rowStarts)-1):
        rw = []
        rw.append(i // pw)
        k = i % pw
        for j in range(num_tasks):
            rw.append(k%2)
            k = k // 2
        features.append(rw)

    init_st = int(pow(2,num_tasks)-1)
    res = []
    freq = np.zeros(len(rowStarts))
    for rounds in range(200):
        if K <= k0/2:
            break
        s = init_st
        for step in range(500):
            i = s
            if freq[i] == 5 and rowStarts[i+1] > rowStarts[i]+1:
                X.append(features[i])

                val = 0
                best_val = 0
                best_act = 0
                for j in range(rowStarts[s],rowStarts[s+1]):
                    if(choiceStarts[j] < choiceStarts[j+1]):
                        val = simulate_run(s, j - rowStarts[s], .99, 120, 200, pseudoPolicy)
                        if val > best_val:
                            best_val = val
                            best_act = j - rowStarts[s]
                y.append(best_act)

                K -= 1
            if K <= k0/2:
                break
            freq[i] += 1
            bestVal = 0
            bestAct = rowStarts[i+1] - 1
            for j in range(rowStarts[i],rowStarts[i+1] - 1):
                if(choiceStarts[j] < choiceStarts[j+1]):
                    if utility[j - rowStarts[i]] / varW[j - rowStarts[i]] > bestVal:
                        bestVal = utility[j - rowStarts[i]] / varW[j - rowStarts[i]]
                        bestAct = j
            j = bestAct
            lngth = (choiceStarts[j+1] - choiceStarts[j])
            cl = choiceStarts[j] + int(random.random() * lngth)
            s = cols[cl]
    for i in range(len(rowStarts)):
        if freq[i] > 0:
            res.append(i)
    i = 0
    samp.clear()
    labl.clear()

    while i < k0/25+K:
        st = int(random.random()*(len(rowStarts)-2))
        if rowStarts[st+1] > rowStarts[st]+1:
            X.append(features[st])
            samp.append(st)
            val = 0
            best_val = 0
            best_act = 0
            for j in range(rowStarts[st],rowStarts[st+1]):
                if(choiceStarts[j] < choiceStarts[j+1]):
                    val = simulate_run(st, j - rowStarts[st], .99, 120, 200, pseudoPolicy)
                    if val > best_val:
                        best_val = val
                        best_act = j - rowStarts[st]
            y.append(best_act)#optPolicy[st])
            labl.append(best_act)

            #y.append(optPolicy[st])
            i += 1
    clf = DecisionTreeClassifier(random_state=0)        
    clf.fit(X,y)
    PrioDTnodes.append(clf.tree_.node_count)
    
    y_pred = clf.predict(features)
    for i in range(len(rowStarts)-1):
        if rowStarts[i+1] == rowStarts[i]:
            prioritizedDTPolicy.append(0)
        else:
            k = rowStarts[i] + y_pred[i]
            if choiceStarts[k+1] == choiceStarts[k]:	# Suggested action does not exist! -- is not valid
                prioritizedDTPolicy.append(0)
            else:
                prioritizedDTPolicy.append(y_pred[i])


for iters in range(iterations):
    print('Iteration #{}'.format(iters))
    priods.clear()
    utility.clear()
    varL.clear()
    varB.clear()
    varW.clear()
    deadline.clear()
    c.clear()
    
    rowStarts = []
    choiceStarts = []
    cols = []
    nnz = []
    rew = []

    rowStarts2 = []
    choiceStarts2 = []
    cols2 = []
    nnz2 = []
    rew2 = []

    optPolicy = []
    optPolicy2 = []

    greedyPolicy = []
    EDFPolicy = []
    RMPolicy = []
    pseudoPolicy = []
    UPAPolicy = []
    randDTPolicy = []
    prioritizedDTPolicy = []

    rowStarts.append(0)
    choiceStarts.append(0)
    rowStarts2.append(0)
    choiceStarts2.append(0)

    ind1 = 0
    ind2 = 0
    sumL = sumB = sumW = 0
    start = time.time()

    for i in range(num_tasks):
        ind = int(random.random()*len(divisors))
        priods.append(int(hyper_parameter/divisors[ind]))
        utility.append(2+int(31*random.random()))
        
        num = int(12*(random.random()-.5))
        if sumL > 12 and num > 0:
            num = -num
        if sumL < -12 and num < 0:
            num = -num
        varL.append(int((L+.01*num)*priods[i]/num_tasks))
        sumL += num

        num = int(12*(random.random()-.5))
        if sumB > 12 and num > 0:
            num = -num
        if sumB < -12 and num < 0:
            num = -num	
        varB.append(int((B+.01*num)*priods[i]/num_tasks))
        sumB += num

        num = int(12*(random.random()-.5))	
        if sumW > 12 and num > 0:
            num = -num
        if sumW < -12 and num < 0:
            num = -num
        varW.append(int((W+.01*num)*priods[i]/num_tasks))
        sumW += num
        dl = varW[i] + int(random.random() * (priods[i] - varW[i]))
        deadline.append(dl)
        c.append(int(random.random() * dl))

    uts = []
    print("Periods:", priods)
    print("L:", varL)
    print("W:", varW)
    print("deadlines:", deadline)
    for i in range(num_tasks):
        uts.append((i,utility[i] / deadline[i]))
    for i in range(num_tasks):
        for j in range(num_tasks - 1):
            if uts[j][1] < uts[j+1][1]:
                tmp = uts[j]
                uts[j] = uts[j+1]
                uts[j+1] = tmp

    start = time.time()

    is_state = np.zeros((hyper_parameter, int(pow(2,num_tasks))), dtype = int)
    is_state[0][int(pow(2,num_tasks)-1)] = True
    state_val = np.zeros(hyper_parameter * int(pow(2,num_tasks)), dtype = float)

    for i in range(hyper_parameter):
        for j in range(int(pow(2,num_tasks))):
            if is_state[i][j] == True:
                computetrans(i,j)
            else:
                rowStarts.append(ind1)
    stop = time.time()
        
    ind1 = ind2 = 0
    is_state2 = np.zeros((hyper_parameter, int(pow(2,num_tasks))), dtype = int)
    mpst = mapstate(int(pow(2,num_tasks)-1))
    is_state2[0][mpst] = True
    state_val2 = np.zeros(hyper_parameter * int(pow(2,num_tasks)), dtype = float)

    valiter(.99, .01)

    stop2 = time.time()
    print('******')
    print('Computed value (VI): {:.3f}, {:<35} running time: {:.3f}'.format(state_val[int(pow(2,num_tasks)-1)], '|', stop2 - start))
    rtVI += stop2 - start
    val1 = state_val[int(pow(2,num_tasks)-1)]
    start = stop2
    
    best_stval = np.zeros(len(state_val))
    for i in range(len(state_val)):
        best_stval[i] = state_val[i]
        state_val[i] = 0
    ava_g = 0
    greedy()

    DTMCcalc(.99, .01, greedyPolicy)
        
    valgr = state_val[int(pow(2,num_tasks)-1)]
    res5 = []
    stop2 = time.time()
    print('Computed value (greedy): {:.3f} {:<10} efficiency rate:{:.3f} {:<6} running time: {:.3f}'.format(state_val[int(pow(2,num_tasks)-1)],'|', state_val[int(pow(2,num_tasks)-1)]/val1, '|',  stop2 - start))
    rtGreedy += stop2 - start
    start = stop2
    res5.append(state_val[int(pow(2,num_tasks)-1)]/val1)
    prGreedy += state_val[int(pow(2,num_tasks)-1)]/val1

    
    for i in range(len(state_val)):
        state_val[i] = 0
    pseudo()
    DTMCcalc(.99, .01, pseudoPolicy)

    stop2 = time.time()
    print('Computed value (seudo): {:.3f} {:<11} efficiency rate:{:.3f} {:<6} running time: {:.3f}'.format(state_val[int(pow(2,num_tasks)-1)],'|', state_val[int(pow(2,num_tasks)-1)]/val1, '|',  stop2 - start))
    rtPseudo += stop2 - start
    start = stop2

    valpseudo = state_val[int(pow(2,num_tasks)-1)]
    res5.append(state_val[int(pow(2,num_tasks)-1)]/val1)
    prPseudo += state_val[int(pow(2,num_tasks)-1)]/val1
    
    for i in range(len(state_val)):
        state_val[i] = 0
    UPA()

    DTMCcalc(.99, .01, UPAPolicy)

    stop2 = time.time()
    print('Computed value (upa-0): {:.3f}  {:<11} efficiency rate:{:.3f} {:<6} running time: {:.3f}'.format(state_val[int(pow(2,num_tasks)-1)],'|', state_val[int(pow(2,num_tasks)-1)]/val1, '|',  stop2 - start))
    rtUPA += stop2 - start
    start = stop2
    res5.append(state_val[int(pow(2,num_tasks)-1)]/val1)
    prUPA += state_val[int(pow(2,num_tasks)-1)]/val1
    
        
    for i in range(len(state_val)):
        state_val[i] = 0
    randdt(5000)
    DTMCcalc(.99, .01, randDTPolicy)

    stop2 = time.time()
    print('Computed value (Rand DT): {:.3f}  {:<9} efficiency rate:{:.3f} {:<6} running time: {:.3f} {:<6} DT nodes: {:.3f}'.format(state_val[int(pow(2,num_tasks)-1)],'|', state_val[int(pow(2,num_tasks)-1)]/val1, '|',  stop2 - start, '|', DTnodes[iters]))
    rtRndDT += stop2 - start
    start = stop2

    res5.append(state_val[int(pow(2,num_tasks)-1)]/val1)
    prRndDT += state_val[int(pow(2,num_tasks)-1)]/val1


    for i in range(len(state_val)):
        state_val[i] = 0
    state_prioritizing_dt(5000)
    DTMCcalc(.99, .01, prioritizedDTPolicy)

    stop2 = time.time()
    print('Computed value (prioritized DT): {:.4f}  {:<1} efficiency rate:{:.3f} {:<6} running time: {:.3f} {:<6} DT nodes: {:.3f} '.format(state_val[int(pow(2,num_tasks)-1)],'|', state_val[int(pow(2,num_tasks)-1)]/val1, '|',  stop2 - start, '|', PrioDTnodes[iters]))

    rtPrioDT += stop2 - start
    start = stop2

    valgr = state_val[int(pow(2,num_tasks)-1)]
    res5.append(state_val[int(pow(2,num_tasks)-1)]/val1)
    prPrioDT += state_val[int(pow(2,num_tasks)-1)]/val1
    
    for i in range(len(state_val)):
        state_val[i] = 0
    EDF()
    DTMCcalc(.99, .01, EDFPolicy)

    stop2 = time.time()
    print('Computed value (EDF): {:.3f} {:<13} efficiency rate:{:.3f} {:<6} running time: {:.3f}'.format(state_val[int(pow(2,num_tasks)-1)],'|', state_val[int(pow(2,num_tasks)-1)]/val1, '|',  stop2 - start))
    rtEDF  += stop2 - start
    start = stop2

    valEDF = state_val[int(pow(2,num_tasks)-1)]
    res5.append(state_val[int(pow(2,num_tasks)-1)]/val1)
    prEDF += state_val[int(pow(2,num_tasks)-1)]/val1
    
    for i in range(len(state_val)):
        state_val[i] = 0
    RM()
    DTMCcalc(.99, .01, RMPolicy)

    stop2 = time.time()
    print('Computed value (RM): {:.3f}  {:<14} efficiency rate:{:.3f} {:<6} running time: {:.3f}'.format(state_val[int(pow(2,num_tasks)-1)],'|', state_val[int(pow(2,num_tasks)-1)]/val1, '|',  stop2 - start))
    rtRM += stop2 - start
    start = stop2

    valRM = state_val[int(pow(2,num_tasks)-1)]
    res5.append(state_val[int(pow(2,num_tasks)-1)]/val1)
    prRM += state_val[int(pow(2,num_tasks)-1)]/val1
    
    reducedMDP()
    
    for i in range(len(is_state)):
        tt = 0
        ss = 0
        for j in range(len(is_state[0])):
            if is_state[i][j] == True:
                ss += 1
    
    for i in range(len(state_val2)):
        state_val2[i] = 0

    valiter2(.99, .01)    
    
    for i in range(len(state_val)):
        state_val[i] = 0

    for i in range(len(rowStarts)-1):
        if rowStarts[i] < rowStarts[i+1]:
            optPolicy2[i] = optPolicy2[mapstate(i)]
    ava_r = 0
    stop2 = time.time()

    DTMCcalc(.99, .01, optPolicy2)

    valredmdp = state_val[int(pow(2,num_tasks)-1)]
    print('Computed value (reduced MDP): {:.3f}  {:<5} efficiency rate:{:.3f} {:<6} running time: {:.3f}'.format(state_val[int(pow(2,num_tasks)-1)],'|', state_val[int(pow(2,num_tasks)-1)]/val1, '|',  stop2 - start))

    rtReducedMDP += stop2 - start
    start = stop2
    res5.append(state_val[int(pow(2,num_tasks)-1)]/val1)
    prReducedMDP += state_val[int(pow(2,num_tasks)-1)]/val1
        
    trns1 = len(nnz)
    trns2 = len(nnz2)
    original_num_trans += trns1 
    reduced1_num_trans += trns2 
    
    for i in range(len(rowStarts)-1):
        if rowStarts[i] < rowStarts[i+1]:
            original_num_state += 1
        if rowStarts2[i] < rowStarts2[i+1]:
            reduced1_num_state += 1
        
        
    rowStarts2.clear()
    choiceStarts2.clear()
    cols2.clear()
    nnz2.clear()
    rew2.clear()
    optPolicy2.clear()
    
    ind1 = 0
    ind2 = 0

    rowStarts2.append(0)
    choiceStarts2.append(0)
    
    XX = PrioreducedMDP()
    trns3 = len(nnz2)            
    reduced2_num_trans += trns3
    for i in range(len(rowStarts)-1):
        if rowStarts2[i] < rowStarts2[i+1]:
            reduced2_num_state += 1

        
    for i in range(len(state_val2)):
        state_val2[i] = 0

    valiter2(.99, .01)    
    
    for i in range(len(state_val)):
        state_val[i] = 0
    length = int(pow(2, num_tasks))
    for i in range(len(rowStarts)-1):
        t = i // length
        rem = i % length
        if rowStarts[i] < rowStarts[i+1]:
            if is_state2[t][rem] == False and is_state[t][rem] == True:
                optPolicy2[i] = optPolicy2[mapstate(i)]

    for st in XX:
        if rowStarts[st+1] > rowStarts[st]+1:
            val = 0
            best_val = 0
            best_act = 0
            for j in range(rowStarts[st],rowStarts[st+1]):
                if(choiceStarts[j] < choiceStarts[j+1]):
                    val = simulate_run(st, j - rowStarts[st], .99, 120, 200, optPolicy2)
                    if val > best_val:
                        best_val = val
                        best_act = j - rowStarts[st]
            optPolicy2[st] = best_act

                    
    ava_r = 0
    stop2 = time.time()

    DTMCcalc(.99, .01, optPolicy2)

    valredmdp = state_val[int(pow(2,num_tasks)-1)]
    print('Computed value (prioritized MDP): {:.3f}  {:<0} efficiency rate:{:.3f} {:<6} running time: {:.3f}'.format(state_val[int(pow(2,num_tasks)-1)],'|', state_val[int(pow(2,num_tasks)-1)]/val1, '|',  stop2 - start))

    rtPrioReducedMDP += stop2 - start
    prPrioReducedMDP += state_val[int(pow(2,num_tasks)-1)]/val1    
    res5.append(state_val[int(pow(2,num_tasks)-1)]/val1)
    results.append(res5)
    print()


print('\nAverage Running times for the used approaches:')
print('VI:', rtVI/iterations)
print('Pseudo-0:', rtPseudo/iterations)
print('Greedy:', rtGreedy/iterations)
print('UPA:', rtUPA/iterations)
print('Random DT:', rtRndDT/iterations)
print('Prioritized DT:', rtPrioDT/iterations)
print('EDF:', rtEDF/iterations)
print('RM:', rtRM/iterations)
print('ReducedMDP:', rtReducedMDP/iterations)
print('PrioReducedMDP:', rtPrioReducedMDP/iterations)

print('\nAverage precision for the used approaches:')
print('Pseudo-0:', prPseudo/iterations)
print('Greedy:', prGreedy/iterations)
print('UPA:', prUPA/iterations)
print('Random DT:', prRndDT/iterations)
print('Prioritized DT:', prPrioDT/iterations)
print('EDF:', prEDF/iterations)
print('RM:', prRM/iterations)
print('ReducedMDP:', prReducedMDP/iterations)
print('PrioReducedMDP:', prPrioReducedMDP/iterations)

print('\nAverage number of DT nodes:')
print('Rand DT: ', sum(DTnodes)/iterations)
print('Prioritized Rand DT: ', sum(PrioDTnodes)/iterations)

print('\nAverage number of states:')
print('Original MDP: ', original_num_state/iterations)
print('Reduced MDP1: ', reduced1_num_state/iterations)
print('Reduced MDP2: ', reduced2_num_state/iterations)
print('\nAverage number of transitions:')
print('Original MDP: ', original_num_trans/iterations)
print('Reduced MDP1: ', reduced1_num_trans/iterations)
print('Reduced MDP2: ', reduced2_num_trans/iterations)
        
print('************************************')
print('Results for the model with' , num_tasks, ' tasks and hyperparameter = ', hyper_parameter) 
for res in results: 
    print(res)
    
filestr = 'results_' + (str)(num_tasks) + '_' + (str)(hyper_parameter) + '_highload2.log' 
myfile = open(filestr, 'w+')
for res in results:
    mystr = "["
    for i in range(len(res)-1):
        mystr += (str)(res[i]) + ', '
    i = len(res)-1
    mystr += (str)(res[i]) + ']\n'
    myfile.writelines(mystr)
myfile.close()
