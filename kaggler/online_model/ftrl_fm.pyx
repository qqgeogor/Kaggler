''' Based on Tinrtgu's FTRL code: http://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory
'''

from csv import DictReader
cimport cython
from libc.math cimport exp, copysign, log, sqrt
import numpy as np

cimport numpy as np
np.import_array()

from datetime import datetime
import random

cdef class FTRL_FM:
    cdef double alpha      # learning rate
    cdef double beta
    cdef double alpha_fm      # learning rate
    cdef double beta_fm
    cdef double L1
    cdef double L2
    cdef double L1_fm
    cdef double L2_fm
    cdef double L1_fm_tmp
    cdef double L2_fm_tmp
    cdef unsigned int fm_dim
    cdef unsigned int D
    cdef double fm_initDev
    cdef double dropoutRate


    cdef unsigned int epoch
    # cdef unsigned int n
    cdef bint interaction
    cdef double[:] w
    cdef double[:] n
    cdef double[:] z
    cdef dict n_fm
    cdef dict z_fm
    cdef dict w_fm
    def __init__(
            self, 
            unsigned int fm_dim=4, 
            double fm_initDev=0.01, 
            double L1=0.0, 
            double L2=0.0, 
            double L1_fm=0.0, 
            double L2_fm=0.0, 
            unsigned int D=2*22, 
            double alpha=0.005, 
            double beta=1.0, 
            double alpha_fm = .1, 
            double beta_fm = 1.0, 
            double dropoutRate = 1.0
            ):
        ''' initialize the factorization machine.'''
        
        self.alpha = alpha              # learning rate parameter alpha
        self.beta = beta                # learning rate parameter beta
        self.L1 = L1                    # L1 regularizer for first order terms
        self.L2 = L2                    # L2 regularizer for first order terms
        self.alpha_fm = alpha_fm        # learning rate parameter alpha for factorization machine
        self.beta_fm = beta_fm          # learning rate parameter beta for factorization machine
        self.L1_fm = L1_fm              # L1 regularizer for factorization machine weights. Only use L1 after one epoch of training, because small initializations are needed for gradient.
        self.L2_fm = L2_fm              # L2 regularizer for factorization machine weights.
        self.fm_dim = fm_dim            # dimension of factorization.
        self.fm_initDev = fm_initDev    # standard deviation for random intitialization of factorization weights.
        self.dropoutRate = dropoutRate  # dropout rate (which is actually the inclusion rate), i.e. dropoutRate = .8 indicates a probability of .2 of dropping out a feature.
        
        self.L1_fm_tmp = L1_fm              # L1 regularizer for factorization machine weights. Only use L1 after one epoch of training, because small initializations are needed for gradient.
        self.L2_fm_tmp = L2_fm              # L2 regularizer for factorization machine weights.

        self.D = D
        
        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        
        # let index 0 be bias term to avoid collisions.
        self.n = np.zeros((self.D + 1,), dtype=np.float64)
        self.z = np.zeros((self.D + 1,), dtype=np.float64)
        self.w = np.zeros((self.D + 1,), dtype=np.float64)
        
        self.n_fm = {}
        self.z_fm = {}
        self.w_fm = {}
        
        
    def init_fm(self,unsigned int i):
        ''' initialize the factorization weight vector for variable i.
        '''
        cdef unsigned int k
        if i not in self.n_fm:
            self.n_fm[i] = [0.] * self.fm_dim
            self.w_fm[i] = [0.] * self.fm_dim
            self.z_fm[i] = [0.] * self.fm_dim
            
            for k in range(self.fm_dim): 
                self.z_fm[i][k] = random.gauss(0., self.fm_initDev)
    
    def predict_raw(self, list x):
        ''' predict_one the raw score prior to logit transformation.
        '''
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2
        alpha_fm = self.alpha_fm
        beta_fm = self.beta_fm
        L1_fm = self.L1_fm
        L2_fm = self.L2_fm
        
        # first order weights model
        n = self.n
        z = self.z
        w = self.w
        
        # FM interaction model
        n_fm = self.n_fm
        z_fm = self.z_fm
        w_fm = self.w_fm
        
        cdef double raw_y = 0.
        cdef unsigned int i
        cdef double sign
        cdef unsigned int len_x
        cdef unsigned int k

        # calculate the bias contribution
        for i in [0]:
            # no regularization for bias
            self.w[i] = (- self.z[i]) / ((self.beta + sqrt(self.n[i])) / self.alpha)
            
            raw_y += self.w[i]
        
        # calculate the first order contribution.
        for i in x:
            sign = -1. if self.z[i] < 0. else 1. # get sign of z[i]
            
            if sign * self.z[i] <= self.L1:
                self.w[i] = 0.
            else:
                self.w[i] = (sign * self.L1 - self.z[i]) / ((self.beta + sqrt(n[i])) / self.alpha + self.L2)
            
            raw_y += self.w[i]
        

        len_x = len(x)
        # calculate factorization machine contribution.
        for i in x:
            self.init_fm(i)
            for k in range(self.fm_dim):
                sign = -1. if self.z_fm[i][k] < 0. else 1.   # get the sign of z_fm[i][k]
                
                if sign * self.z_fm[i][k] <= self.L1_fm:
                    self.w_fm[i][k] = 0.
                else:
                    self.w_fm[i][k] = (sign * self.L1_fm - self.z_fm[i][k]) / ((self.beta_fm + sqrt(self.n_fm[i][k])) / self.alpha_fm + self.L2_fm)
        
        for i in range(len_x):
            for j in range(i + 1, len_x):
                for k in range(self.fm_dim):
                    raw_y += w_fm[x[i]][k] * w_fm[x[j]][k]
        
        return raw_y
    
    def predict_one(self, list x):
        ''' predict_one the logit
        '''
        return 1. / (1. + exp(- max(min(self.predict_raw(x), 35.), -35.)))
    
    def dropout(self, list x):
        ''' dropout variables in list x
        '''
        cdef unsigned int i
        cdef double var
        for i, var in enumerate(x):
            if random.random() > self.dropoutRate:
                del x[i]
    
    def dropoutThenPredict(self, list x):
        ''' first dropout some variables and then predict_one the logit using the dropped out data.
        '''
        self.dropout(x)
        return self.predict_one(x)
    
    def predictWithDroppedOutModel(self, list x):
        ''' predict_one using all data, using a model trained with dropout.
        '''
        return 1. / (1. + exp(- max(min(self.predict_raw(x) * self.dropoutRate, 35.), -35.)))
    
    def update(self, list x, double p, double y):
        ''' Update the parameters using FTRL (Follow the Regularized Leader)
        '''
        # alpha = self.alpha
        # alpha_fm = self.alpha_fm
        
        # # model
        # n = self.n
        # z = self.z
        # w = self.w
        
        # # FM model
        # n_fm = self.n_fm
        # z_fm = self.z_fm
        # w_fm = self.w_fm
        
        cdef double g
        # cost gradient with respect to raw prediction.
        g = p - y

        cdef int len_x
        cdef int i
        cdef int j
        cdef int k

        fm_sum = {}      # sums for calculating gradients for FM.
        len_x = len(x)
        
        for i in x + [0]:
            # update the first order weights.
            sigma = (sqrt(self.n[i] + g * g) - sqrt(self.n[i])) / self.alpha
            self.z[i] += g - sigma * self.w[i]
            self.n[i] += g * g
            
            # initialize the sum of the FM interaction weights.
            fm_sum[i] = [0.] * self.fm_dim
        
        # sum the gradients for FM interaction weights.
        for i in range(len_x):
            for j in range(len_x):
                if i != j:
                    for k in range(self.fm_dim):
                        fm_sum[x[i]][k] += self.w_fm[x[j]][k]
        
        for i in x:
            for k in range(self.fm_dim):
                g_fm = g * fm_sum[i][k]
                sigma = (sqrt(self.n_fm[i][k] + g_fm * g_fm) - sqrt(self.n_fm[i][k])) / self.alpha_fm
                self.z_fm[i][k] += g_fm - sigma * self.w_fm[i][k]
                self.n_fm[i][k] += g_fm * g_fm
    
    def write_w(self, filePath):
        ''' write out the first order weights w to a file.
        '''
        with open(filePath, "w") as f_out:
            for i, w in enumerate(self.w):
                f_out.write("%i,%f\n" % (i, w))
    
    def write_w_fm(self, filePath):
        ''' write out the factorization machine weights to a file.
        '''
        with open(filePath, "w") as f_out:
            for k, w_fm in self.w_fm.iteritems():
                f_out.write("%i,%s\n" % (k, ",".join([str(w) for w in w_fm])))


    def predict(self,testingFile,hashSalt='salt', n_epochs=5,reportFrequency=10000):
        start = datetime.now()
        # initialize a FM learner
        learner = self
        cdef int e
        cdef double cvLoss = 0.
        cdef double cvCount = 0.
        cdef double progressiveLoss = 0.
        cdef double progressiveCount = 0.
        cdef list x
        cdef double y
        cdef unsigned int t
        cdef double p
        cdef double loss
        cdef list y_preds = []
        for t, ID, x, y in data(testingFile, self.D, hashSalt):
            p = learner.predict_one(x)
            y_preds.append(p)
        return y_preds
        
        
    def fit(self,trainingFile,hashSalt='salt',n_epochs=5,reportFrequency=10000):
        start = datetime.now()
        # initialize a FM learner
        learner = self
        cdef int e
        cdef double cvLoss = 0.
        cdef double cvCount = 0.
        cdef double progressiveLoss = 0.
        cdef double progressiveCount = 0.
        cdef list x
        cdef double y
        cdef unsigned int t
        cdef double p
        cdef double loss
        print("Start Training:")
        for e in range(n_epochs):
            
            # if it is the first epoch, then don't use L1_fm or L2_fm
            if e == 0:
                learner.L1_fm = 0.
                learner.L2_fm = 0.
            else:
                learner.L1_fm = learner.L1_fm_tmp
                learner.L2_fm = learner.L1_fm_tmp
            

            for t, ID, x, y in data(trainingFile, self.D, hashSalt):
                p = learner.predict_one(x)
                loss = logLoss(p, y)
                learner.update(x, p, y)
                progressiveLoss += loss
                progressiveCount += 1.
                if t % reportFrequency == 0:                
                    print("Epoch %d\tcount: %d\tProgressive Loss: %f" % (e, t, progressiveLoss / progressiveCount))
                
            print("Epoch %d finished.\tvalidation loss: %f\telapsed time: %s" % (e, cvLoss / cvCount, str(datetime.now() - start)))
            
            
def logLoss(double p, double y):
    ''' 
    calculate the log loss cost
    p: prediction [0, 1]
    y: actual value {0, 1}
    '''
    p = max(min(p, 1. - 1e-15), 1e-15)
    return - log(p) if y == 1. else -log(1. - p)

def data(filePath, hashSize, hashSalt):
    ''' generator for data using hash trick
    
    INPUT:
        filePath
        hashSize
        hashSalt: String with which to salt the hash function
    '''
    cdef unsigned int t
    cdef double y
    cdef list x
    cdef str value
    cdef unsigned int index
    cdef dict row
    import os
    for t, row in enumerate(DictReader(filePath)):
        ID = row['activity_id']
        del row['activity_id']
        
        del row['outcome_isnull']

        y = 0.
        if 'outcome' in row:
            if row['outcome'] == '1':
                y = 1.
            del row['outcome']
        
        # date = int(row['hour'][4:6])
        
        # row['hour'] = row['hour'][6:]
        
        x = []
        
        for key in row:
            value = row[key]
            
            index = abs(hash(hashSalt + key + '_' + value)) % hashSize + 1      # 1 is added to hash index because I want 0 to indicate the bias term.
            x.append(index)
        
        yield t, ID, x, y