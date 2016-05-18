from collections import defaultdict
import random as rand
import numpy as np
import math as m

class bam:
    def __init__(self, n, p, random=False):
        self.energy = self.mean = 0.0
        self.stdev = 0.1
        
        self.row_count = n
        self.col_count = p
        self.weight_matrix = self.make_new_weight_matrix(random)
        
        # this will count the x's encountered that were to be called in
        # the logistic function.
        self.x_count = defaultdict
        
        # the "bias node" i.e. extra input to be placed into input and
        # output vectors
        self.master_bias_in = 1
        self.master_bias_out = 1
        self.continuous = True
        
    # stochasticity will only matter if we are using a non-continuous BAM.
    # if continuous, the result that will be returned will have values
    # between 0 to 1; i.e. we're gonna be using the sigmoid function
    # instead of the tanh function
    def feedForward(self, input, stochastic=True):
        if self.continuous:
            input.append(self.master_bias_in)
            
        result = ( np.mat(input) * np.mat(self.weight_matrix) ).tolist()[0]

        if self.continuous:
            result = map(lambda x: self.sigmoid(x), result)
        else:
            if stochastic:
                result = map(lambda x: 1 if self.logistic(x) > self.random_gaussian() else -1, result)
            else:
                result = map(lambda x: 1 if x>0 else -1, result)
        
        if self.continuous:
            input = input[:-1]
            
        return result
        
    def feedBackward(self, output, stochastic=True):
        if self.continuous:
            output.append(self.master_bias_out)
            
        result = ( np.mat(output) * np.mat(zip(*self.weight_matrix)) ).tolist()[0]
        
        if self.continuous: 
            result = map(lambda x: self.sigmoid(x), result)
        else:
            if stochastic:
                result = map(lambda x: 1 if self.logistic(x) > self.random_gaussian() else -1, result)
            else:
                result = map(lambda x: 1 if x>0 else -1, result)
                
        if self.continuous:
            output = output[:-1]
            
        return result
    
    # note: the source code of sir happy originally said
    #   weight_matrix[r][c]input.at(r) * input.at(c)
    #   it was clarified na dapat output yung isa
    def computeEnergy(self, input, output):
        e = 0.0
        for r in range(len(input)):
            for c in range(len(output)):
                e+=self.weight_matrix[r][c] * input[r] * output[c]
        e*=-1
        
        if not self.continuous:
            return e
            
        for i in input:
            e-=(self.sigmoid(i)*self.master_bias_in)
            e+=self.the_other_thing(i)
            # e-=m.tanh(i)                        #this was for tanh
            # e+=(i*m.tanh(i) - m.log(m.cosh(i))) #this was for tanh
            
        for o in output:
            e-=(self.sigmoid(o)*self.master_bias_out)
            e+=self.the_other_thing(o)
            # e-=m.tanh(o)                        #this was for tanh
            # e+=(i*m.tanh(o) - m.log(m.cosh(o))) #this was for tanh
            
        return e
    
    def train(self, input, output):
        for i in range(len(input)):
            sample_in = input[i]
            sample_out = output[i]
            for r in range(self.row_count):
                for c in range(self.col_count):
                    self.weight_matrix[r][c] += sample_in[r] * sample_out[c]
        # newweight_matrix = np.mat(self.weight_matrix)
        # for pair in zip(input, output):
            # m2 = np.mat(zip(pair[0])) * np.mat(pair[1])
            # newweight_matrix += m2
        # self.weight_matrix = newweight_matrix.tolist()
        
    def logistic(self, x):
        # self.x_count[x] += 1
        # return 1-x*x
        try:
            return float("inf") if x==0 else 1.0/(1.0-(m.exp(-x)))
        except OverflowError:
            return float("inf")
        

    def random_gaussian(self):
        return np.random.normal(self.mean, self.stdev)
    
    def make_new_weight_matrix(self, random=False):
        if not random:
            return [ [0.0]*self.col_count for i in range(self.row_count) ]
        else:
            return [ [rand.randrange(-3, 3) for i in range(self.col_count)] for j in range(self.row_count)]
        
    def printweight_matrix(self):
        print np.mat(self.weight_matrix)
    
    # gotten with te help of wolframalpha.com    
    def sigmoid(self, x):
        return 1.0/(1+m.exp(-x))
        
    # gotten with te help of wolframalpha.com
    def the_other_thing(self, x):
        return( (x*m.exp(x))*1.0/(m.exp(x)+1) )
        
    def getEnergy(self):
        return self.energy
        
    def check_raw(self):
        for x in self.weight_matrix:
            for y in x:
                if y != 0:
                    return False
        return True