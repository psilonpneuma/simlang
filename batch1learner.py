import seaborn as sns
import random as rnd
import matplotlib.pyplot as plt
from scipy.stats import beta
#hypothesis selected = p

production="sample" #softmax #map
r=1
runs=10000

#input 
def generate(starting_count_w1,n_productions):
    data=[1]*starting_count_w1 + [0]*(n_productions-starting_count_w1)
    return data

#output
def produce(p): # this implements sampling on production - try maximising here too	
	p0=1-p
	if production == "sample":
	    if rnd.random()<p:
	       return 1
	    else:
	       return 0
	#maximization
	elif production == "map":
	    if p >= 0.5:
	       data.append(0)
	    else:
	       data.append(1)
	#soft maximization
	elif production == "softmax":
    	    p1=p**r/(p**r+p0**r)
    	    if rnd.random()<p1:
	       return 1
	    else:
	       return 0

alpha=1


#----
#starting input ratio and number of productions
data=generate(9,10)
number_of_ones=float(data.count(1))


#----
# Hypothesis choice
model="sampler"
language=beta.rvs(alpha+number_of_ones, alpha+(10-number_of_ones)) # sampling
#language=(alpha+number_of_ones-1)/(alpha*2+10-2) # maximising
#language=(alpha+number_of_ones)/(alpha+10) # averaging 


#----
# Production
# every run counts the occurrencies of x
ones=[] #count of x in ogni run
for r in range(runs):
	data=[produce(language) for _ in range(10)] #one list of 01s
	#big_data.append(data)
	number_of_ones=float(data.count(1))
	ones.append(number_of_ones)
print "list of ones: ",ones[1:10]

#dictionary with x_possible_values:freqs(x), ordered by n_of_x
d = {}
for c in ones:
    count=ones.count(c)
    d[c] = count
print "dictionary: ",d.items()[1:10]

#get probabilities of proportion_of_ones as list of tuples (n,prob(n))
prob=[(n,float(freq)/len(ones)) for n, freq in d.items()]
print "probabilities: ",prob[1:10]


#plots
plt.figure()
plt.bar([x[0] for x in prob],[x[1] for x in prob],align='center',width=0.2,color='r')
plt.xticks(range(11),fontsize=18)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7],fontsize=10)
plt.xlabel("output count of variant x",fontsize=14)
plt.ylabel("P(x'|x)",fontsize=30)
plt.subplots_adjust(left=0.15)
plt.subplots_adjust(bottom=0.15)
title=model+"_bias="+str(alpha)+production
plt.savefig("/Users/chiarasemenzin/Desktop/Dissertation/graphs/batch/"+title+".png")
plt.show()




