
"""
Bayesian Iterated Learning - learning variable object labels

This code implements the Reali & Griffiths model of the evolution 
of unpredictable variation in object labels. This is a Bayesian Iterated Learning 
simulation, based around the beta-binomial model: there are two words, 0 and 1, 
a 'grammar' specifies a probability distribution over those two words, i.e. it 
specifies the probability of word 0 and the probability of word 1 - since these
must sum to 1, we can simply track the probability of word 1, pW1. The 
learner's task is to infer pW1 based on some data and a prior over possible 
values of pW1. Learning is modelled as a process of Bayesian inference: while 
pW1 is a continuous value, we use a simple grid approximation to model inference 
as a process of selecting one of a discrete set of possible values of pW1.

Usage example (you can copy and paste this to the prompt): 

pW1_by_generation,data_by_generation = iterate(0.1,10,5,10)

This will run the simulation for 10 generations, using a prior defined by alpha=0.1, 
each learner observes 10 data points before inferring pW1, and the initial language
consists of 5 examples of word 1 (and therefore 5 of word 0). It returns two 
values: a generation-by-generation record of the inferred values of pW1, and 
the data produced at each generation (specified as a number of occurences of word 1).
"""
# CHANGE!
import random as rnd
#we need a few extra functions from various libraries, for doing stuff 
#with log probabilities and probablity distributions
from scipy.stats import beta
from scipy.misc import logsumexp
from math import log, log1p, exp
from numpy import nan

# ----- functions for dealing with log probabilities -----
learning = "map"
production = "softmax"
r = 0.5

def log_subtract(x,y):
    '''
    x-y in the log domain, i.e. equivalent to log(exp(x)-exp(y))
    '''
    return x + log1p(-exp(y-x))

def normalize_probs(probs):
    '''
    Takes a list of numbers, and normalises them, so that 
    they sum to 1. 
    '''
    total=sum(probs) #calculates the summed log probabilities
    normedprobs=[]
    for p in probs:
        normedprobs.append(p/total) #normalise - subtracting in the log domain
                                        #equivalent to divising in the normal domain
    return normedprobs


def normalize_logprobs(logprobs):
    '''
    Same idea as normalize_probs, but in the log domain: takes a list of numbers 
    in the log domain, and normalises them, so that they sum to (0) (which is
    1 when converted to non-log probabilities). 
    '''
    logtotal=logsumexp(logprobs) #calculates the summed log probabilities
    normedlogs=[]
    for logp in logprobs:
        normedlogs.append(logp-logtotal) #normalise - subtracting in the log domain
                                        #equivalent to divising in the normal domain
    return normedlogs

def log_roulette_wheel(normedlogs):
    '''
    Selects a random index from normedlogs, with probability of selection being 
    proportional to probability. Note that this assumes that you feed in a normalised
    list of log probabilities.
    '''
    r=log(rnd.random()) #generate a random number in range(0,1), then convert to log eg log(0.75)
    accumulator=normedlogs[0] #eg log(0.6) #probability of w1
    for i in range(len(normedlogs)): #eg 2 = for i in range(2) -> i == 0, i == 1
        if r<accumulator: 
            return i
        accumulator=logsumexp([accumulator,normedlogs[i+1]])

def wta(items):
    '''
    This is good old winner-take-all - we are going to use it for MAP learners, 
    to pick the language with highest posterior probability.
    '''
    maxweight = max(items)
    candidates = []
    for i in range(len(items)):
        if items[i] == maxweight:
            candidates.append(i)
    return rnd.choice(candidates)


# ----- setting up the grid -----
#This sets up the grid is the granularity of the grid that we use for
#bayesian inference, which allows us to turn inference about a continuous
#probability (what proportion of the time does my teacher use word 1?) into 
#a simpler, discrete problem (which of the following proportions is my teacher 
#using?  
grid_granularity = 10
grid_increment = 1/(grid_granularity+0.)

#sets up the grid of possible probabilities to consider
possible_pW1 = []
for i in range(grid_granularity):
    possible_pW1.append(grid_increment/2 + (grid_increment*i))

#sets up the grid of log probabilities
possible_logpW1 = []
for pW1 in possible_pW1:
    possible_logpW1.append(log(pW1))


# ----- functions for Bayesian inference -----

def calculate_prior(alpha):
    '''
    Calculates the prior probability of all values of possible_pw1.
    Only produces symmetrical priors: favours regularity when alpha < 1, 
    uniform when alpha = 1, favours variability when alpha > 1.
    NOTE that this function is not called anywhere else in this code - you 
    can use it to display a prior if you like.
    '''
    logprior = []
    for pW1 in possible_pW1:
        logprior.append(beta.pdf(pW1,alpha,alpha)) 
    return normalize_probs(logprior)


def calculate_logprior(alpha):
    '''
    Calculates the log prior probability of all values of possible_pw1.
    '''
    logprior = []
    for pW1 in possible_pW1:
        logprior.append(beta.logpdf(pW1,alpha,alpha)) 
    return normalize_logprobs(logprior) 



def likelihood(data,logpW1):
    '''Calculates the (log) probability of data d, where data is a string of 0s 
    (representing word 0) and 1s (representing word 1)'''
    logpW0 = log_subtract(log(1),logpW1) #probability of w0 is 1-prob of w1
    logprobs = [logpW0,logpW1]
    loglikelihoods = []
    for d in data:
        loglikelihood_this_item = logprobs[d] #d will be either 0 or 1, 
                                                #so can use as index
        loglikelihoods.append(loglikelihood_this_item)
    return sum(loglikelihoods) #summing log probabilities = 
                                #multiply non-log probabilities

def posterior(data,prior):
    '''
    Calculates posterior probability for all possible values of logpW1, given
    data and prior (a list of log probabilities). Considers the values of 
    logpW1 given in the list possible_logpW1.
    '''
    posterior_logprobs = []
    for i in range(len(possible_logpW1)):
        logpW1 = possible_logpW1[i] 
        logp_h = prior[i] #prior probability of this pW1
        logp_d = likelihood(data,logpW1) #likelihood of data given this pW1
        posterior_logprobs.append(logp_h + logp_d) #adding logs = 
                                                        #multiplying non-logs
    return normalize_logprobs(posterior_logprobs) 

def learn(data,prior):
    '''
    Inferred the (log) probability of word 1, given prior and data: 
    calculates posterior probability distribution,
    then selects a value using 
    log_roulette_wheel - or sampling 
    '''
    posterior_logprobs = posterior(data,prior)
    return posterior_logprobs
    
def produce(posterior_logprobs, n_productions,r=None):
    '''
    Returns data, a list of 0s and 1s (representing w0 and w1)
    '''    
    data = []
    #hypotheses = []
    for p in range(n_productions):
    
    # HYPOTHESIS SELECTION
        if learning == 'sample':
            selected_index = log_roulette_wheel(posterior_logprobs)
            logpW1=possible_logpW1[selected_index] #logpW1
        elif learning == 'map':
            selected_index = wta(posterior_logprobs)
            logpW1=possible_logpW1[selected_index]
                
    # PRODUCTION FROM HYPOTHESIS
        logpW0 = log_subtract(log(1),logpW1)
        # add here softmax?
        logprobs = [logpW0,logpW1]
        if production == "sample":
            data.append(log_roulette_wheel(logprobs))
        elif production == "map":
            if logpW1 >= log(0.5):
                data.append(0)
            else:
                data.append(1)
        elif production == "softmax":
            logpW1=(r*logpW1)-logsumexp([r*logpW1, r*logpW0])
            logpW0 = log_subtract(log(1),logpW1)
            logprobs = [logpW0,logpW1]
            data.append(log_roulette_wheel(logprobs))

    
    return logpW1, data
    #return hypotheses
        

# ----- iterated learning -----


def iterate(alpha,n_productions,starting_count_w1,generations):
    '''
    Runs an iterated learning simulation. 
    Starts with data consisting of starting_count_w1 instances of w1 and 
    (n_productions-starting_count_w1) instances of w0.
    Returns two values: the inferred probability of w1 at each generation
    from 1 onwards (not a log - I convert it to a genuine probability for you), 
    and the number of productions of w1 from generation 0 onwards. 
    '''
    prior = calculate_logprior(alpha)
    pW1_accumulator=[nan]
    data_accumulator=[starting_count_w1]
    data=[1]*starting_count_w1 + [0]*(n_productions-starting_count_w1)
    for generation in range(1,generations+1):
        distr = learn(data,prior)
        logpW1,data = produce(distr,n_productions,r)
        pW1_accumulator.append(exp(logpW1))
        data_accumulator.append(sum(data))
    return pW1_accumulator,data_accumulator
    

#PLOTS


import matplotlib.pyplot as plt
 
#calculate the (log) posterior probabilities for data [1,1,1,1,1,1,1,0,0,0]
# and a learner with prior given by alpha=0.1
# log_posterior_distribution = posterior([1]*7 + [0]*3,calculate_logprior(0.1))
 
# #convert to raw probabilities
# posterior_distribution = []
# for logp in log_posterior_distribution:
#     posterior_distribution.append(exp(logp)) 

# plt.figure()
# #this will produce a bar chart with grid_granularity evenly-spaced blue bars
# plt.bar(possible_pW1,posterior_distribution,align='center',width=1./grid_granularity,color='b')
# #add nice tick marks
# plt.xticks(possible_pW1,fontsize=18)
# plt.yticks(fontsize=18)
# plt.xlabel(r'$\theta$',fontsize=30)
# plt.ylabel(r'$P(\theta | d)$',fontsize=30)
# #messing with the margins so the axis labels aren't cropped off
# plt.subplots_adjust(left=0.15)
# plt.subplots_adjust(bottom=0.15)
##save it somewhere if you want, or show it!
##plt.savefig("/Users/kennysmith/Documents/teaching/SimulatingLanguage/sl_1415/LectureSlides/LectureImages/posterior_5.png")
#plt.show()
# ----- Do a bunch of simulation runs, the pick out a particular generation and plot the theta
# ----- values for all the learners at that generation

import numpy as np
nruns = 1000 #do *a lot* of runs
generation_to_plot = 50 #plot the distribution at generation n
 
#do a bunch of runs and collate the data
simulation_data = []
for i in range(nruns):
    print '.',
    theta_values,_data = iterate(1,10,5,50)
    simulation_data.append(theta_values)
 
#OK, so now we have some data, we can plot the distribution of theta values 
#across all our runs at a given generation
 
#dig out the correct generation from each run
this_g_data = []
for run in simulation_data:
    this_g_data.append(run[generation_to_plot])
 
#now we have a long list of theta values - we want a histogram, showing how 
#many times we get each of the possible values. np_histogram does that for us
counts,_=np.histogram(this_g_data,bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
freqs = [x/(nruns+0.) for x in counts]
 
#now we can plot the data as before
plt.figure()
plt.bar(possible_pW1,freqs,align='center',width=0.1,color='b')
plt.xticks(possible_pW1,fontsize=18)
plt.yticks([0,0.1,0.2,0.3,0.4],fontsize=18)
plt.xlabel(r'$\theta$',fontsize=30)
plt.ylabel('Proportion of learners',fontsize=30)
plt.subplots_adjust(left=0.15)
plt.subplots_adjust(bottom=0.15)
plt.title("Uniform prior, generation "+str(generation_to_plot))
plt.ylim(0,0.4)
#plt.savefig("/Users/kennysmith/Documents/teaching/SimulatingLanguage/sl_1415/LectureSlides/LectureImages/iterate_u_"+str(g)+".png")
plt.show()

