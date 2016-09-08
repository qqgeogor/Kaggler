# time pypy-2.4 -u runmodel.py | tee output_0.txt
from kaggler.online_model.ftrl_fm import FTRL_FM
import random
from math import log
import numpy as np
from datetime import datetime
#### RANDOM SEED ####
random.seed(5)  # seed random variable for reproducibility
#####################

####################
#### PARAMETERS ####
####################
reportFrequency = 1000000
path = "E:\\Redhat\\"
trainingFile = "E:\\Redhat\\train_le_date.csv"
testingFile = "E:\\Redhat\\test_le_date.csv"

fm_dim = 32
fm_initDev = .01
hashSalt = "salty"

alpha = 0.1
beta = 1.

alpha_fm = .01
beta_fm = 1.

p_D = 22
D = 2 ** p_D

L1 = 1.0
L2 = .1
L1_fm = 2.0
L2_fm = 3.0

n_epochs = 5

####
start = datetime.now()

# initialize a FM learner
learner = FTRL_FM(fm_dim, fm_initDev, L1, L2, L1_fm, L2_fm, D, alpha, beta, alpha_fm = alpha_fm, beta_fm = beta_fm)

learner.fit(trainingFile=open(trainingFile),n_epochs=5)
# save the weights
# w_outfile = path+"param.w.txt"
# w_fm_outfile = path+"param.w_fm.txt"
# learner.write_w(w_outfile)
# learner.write_w_fm(w_fm_outfile)
pd.to_pickle(learner,path+'ftrl_fm.pkl')


test = pd.read_csv(path+'test_le_date.csv')
activity_id = test['activity_id']
print('Make submission')
# X_t = [X_t[:,i] for i in range(X_t.shape[1])]
y_preds = learner.predict(testingFile=open(testingFile),n_epochs=5)
submission = pd.DataFrame()
submission['activity_id'] = activity_id
submission['outcome'] = outcome
submission.to_csv('submission_ftrl_fm_%s.csv'%dim,index=False)
