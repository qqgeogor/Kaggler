import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.preprocessing import LabelEncoder,LabelBinarizer,MinMaxScaler,OneHotEncoder,StandardScaler,Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge,Convolution1D,MaxPooling1D,Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Nadam
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,SReLU
from keras.models import Model
from keras.utils.visualize_util import plot
import distance
import xgboost as xgb

seed = 1024
np.random.seed(seed)

path = "../"


def str_jaccard(str1, str2):
    res = distance.jaccard(str1, str2)
    return res


question_numeric = ['char_4_q','char_5_q','char_6_q']

train = pd.read_csv(path+'invited_info_train.txt',dtype={"expert_id":str,'question_id':str})
expert_id = train['expert_id'].values
expert_id = LabelEncoder().fit_transform(expert_id)

test = pd.read_csv(path+'validate_nolabel.txt',dtype={"expert_id":str,'question_id':str}).fillna(-1)
test.columns = ['question_id','expert_id','label']
len_train = train.shape[0]


train = pd.concat([train,test])

expert = pd.read_csv(path+'user_info.txt',dtype={"expert_id":str})
question = pd.read_csv(path+'question_info.txt',dtype={"question_id":str}).fillna(-1)
question['char_3_q'] = question['char_3_q'].astype(str)

expert['char_1'] = expert['char_1'].apply(lambda x: x.replace('/',' '))
expert['char_2'] = expert['char_2'].apply(lambda x: x.replace('/',' '))
expert['char_3'] = expert['char_3'].apply(lambda x: x.replace('/',' '))

question['char_2_q'] = question['char_2_q'].apply(lambda x: x.replace('/',' '))
question['char_3_q'] = question['char_3_q'].apply(lambda x: x.replace('/',' '))

count_char_1 = CountVectorizer(ngram_range=(1,3))
tfidf_char_2 = TfidfVectorizer(ngram_range=(1,3))
tfidf_char_3 = TfidfVectorizer(ngram_range=(1,3))

count_char_1.fit(expert['char_1'].values)
tfidf_char_2.fit(expert['char_2'].values.tolist()+question['char_2_q'].values.tolist())
tfidf_char_3.fit(expert['char_3'].values.tolist()+question['char_3_q'].values.tolist())

lb_char_1_q = LabelBinarizer(sparse_output=True)
lb_char_1_q.fit(question['char_1_q'].values)


train = pd.merge(train,expert,on='expert_id',how='left')#.fillna(' ')
train = pd.merge(train,question,on='question_id',how='left')


le = LabelEncoder()
train['question_id'] = le.fit_transform(train['question_id'].values)
train['expert_id'] = le.fit_transform(train['expert_id'].values)

y = train['label'].values
features = [
    'question_id',
    'expert_id',
    ]

X = train[features].values
# X = OneHotEncoder().fit_transform(X).tocsr()
# X_char_1 = count_char_1.transform(train['char_1'].values)
# X_char_2 = tfidf_char_2.transform(train['char_2'].values)
# X_char_3 = tfidf_char_3.transform(train['char_3'].values)


# X_char_1_q = lb_char_1_q.fit_transform(train['char_1_q'].values)
# X_char_2_q = tfidf_char_2.transform(train['char_2_q'].values)
# X_char_3_q = tfidf_char_3.transform(train['char_3_q'].values)

# stand_char_4_5_6_q = StandardScaler()
# stand_char_4_5_6_q.fit(train[question_numeric].values)
# X_char_4_5_6_q = stand_char_4_5_6_q.transform(train[question_numeric].values)


print ('X raw',X.shape)

# sim_char_2 = []
# for expert_char_2,question_char_2 in zip(X_char_2,X_char_2_q):
#     cos_sim_2 = pairwise_distances(expert_char_2, question_char_2, metric='cosine')[0][0]
#     sim_char_2.append(cos_sim_2)
# sim_char_2 = np.array(sim_char_2)
# sim_char_2 = np.expand_dims(sim_char_2,1)

# sim_char_3 = []
# for expert_char_3,question_char_3 in zip(X_char_3,X_char_3_q):
#     cos_sim_3 = pairwise_distances(expert_char_3, question_char_3, metric='cosine')[0][0]
#     sim_char_3.append(cos_sim_3)
# sim_char_3 = np.array(sim_char_3)
# sim_char_3 = np.expand_dims(sim_char_3,1)

# X = ssp.hstack([
#     X,
#     # X_char_1,
#     # X_char_2,
#     # X_char_3,
#     # X_char_1_q,
#     # X_char_2_q,
#     # X_char_3_q,
#     # X_char_4_5_6_q,
#     # sim_char_2,
#     # sim_char_3,
#     ]).tocsr()

# dump_svmlight_file(X,y,path+'data.svm')

# data,y_all = load_svmlight_file(path+'data.svm')
y_all = y
data = X
num_q = len(np.unique(data[:,0]))
num_e = len(np.unique(data[:,1]))
del X
del y

X = data[:len_train]
y = y_all[:len_train]
X_t= data[len_train:]
del data
del y_all

def make_mf_lr(X ,y, clf, X_test, n_round=3):
    n = X.shape[0]
    '''
    Fit metafeature by @clf and get prediction for test. Assumed that @clf -- regressor
    '''
    print clf
    mf_tr = np.zeros(X.shape[0])
    mf_te = np.zeros(X_test.shape[0])
    for i in range(n_round):
        skf = StratifiedKFold(y, n_folds=2, shuffle=True, random_state=42+i*1000)
        for ind_tr, ind_te in skf:
            X_tr = X[ind_tr]
            X_te = X[ind_te]
            
            # print('X_tr shape',X_tr.shape)
            # print('X_te shape',X_te.shape)
            
            y_tr = y[ind_tr]
            y_te = y[ind_te]
            
            clf.fit(X_tr, y_tr)
            mf_tr[ind_te] += clf.predict_proba(X_te)[:,1]
            mf_te += clf.predict_proba(X_test)[:,1]*0.5
            y_pred = clf.predict_proba(X_te)[:,1]
            score = roc_auc_score(y_te, y_pred)
            print 'pred[{}] score:{}'.format(i, score)
    return (mf_tr / n_round, mf_te / n_round)


def make_mf_lsvc(X ,y, clf, X_test, n_round=3):
    n = X.shape[0]
    '''
    Fit metafeature by @clf and get prediction for test. Assumed that @clf -- regressor
    '''
    print clf
    mf_tr = np.zeros(X.shape[0])
    mf_te = np.zeros(X_test.shape[0])
    for i in range(n_round):
        skf = StratifiedKFold(y, n_folds=2, shuffle=True, random_state=42+i*1000)
        for ind_tr, ind_te in skf:
            X_tr = X[ind_tr]
            X_te = X[ind_te]
            
            # print('X_tr shape',X_tr.shape)
            # print('X_te shape',X_te.shape)
            
            y_tr = y[ind_tr]
            y_te = y[ind_te]
            
            clf.fit(X_tr, y_tr)
            mf_tr[ind_te] += clf.decision_function(X_te)
            mf_te += clf.decision_function(X_test)*0.5
            y_pred = clf.decision_function(X_te)
            score = roc_auc_score(y_te, y_pred)
            print 'pred[{}] score:{}'.format(i, score)
    return (mf_tr / n_round, mf_te / n_round)

def make_mf_nn(X ,y, X_test, n_round=3):
    n = X.shape[0]
    '''
    Fit metafeature by @clf and get prediction for test. Assumed that @clf -- regressor
    '''
    from kaggler.online_model.ftrl import FTRL
    mf_tr = np.zeros(X.shape[0])
    mf_te = np.zeros(X_test.shape[0])
    for i in range(n_round):
        skf = StratifiedKFold(y, n_folds=2, shuffle=True, random_state=42+i*1000)
        for ind_tr, ind_te in skf:
            clf = build_model(X)
            X_tr = [X[:,0][ind_tr],X[:,1][ind_tr]]
            X_te = [X[:,0][ind_te],X[:,1][ind_te]]

            # print('X_tr shape',X_tr.shape)
            # print('X_te shape',X_te.shape)
            
            y_tr = y[ind_tr]
            y_te = y[ind_te]
            
            clf.fit(X_tr, y_tr,nb_epoch=2,batch_size=128,validation_data=[X_te,y_te])
            mf_tr[ind_te] += clf.predict(X_te).ravel()
            mf_te += clf.predict([X_test[:,0],X_test[:,1]]).ravel()*0.5
            y_pred = clf.predict(X_te).ravel()
            score = roc_auc_score(y_te, y_pred)
            print 'pred[{}] score:{}'.format(i, score)
    return (mf_tr / n_round, mf_te / n_round)

def build_model(X,dim=128):

    inputs_p = Input(shape=(1,), dtype='int32')

    embed_p = Embedding(
                    num_q,
                    dim,
                    dropout=0.2,
                    input_length=1
                    )(inputs_p)

    inputs_d = Input(shape=(1,), dtype='int32')

    embed_d = Embedding(
                    num_e,
                    dim,
                    dropout=0.2,
                    input_length=1
                    )(inputs_d)


    flatten_p= Flatten()(embed_p)
    
    flatten_d= Flatten()(embed_d)
    
    flatten = merge([
                flatten_p,
                flatten_d,
                ],mode='concat')
    
    fc1 = Dense(512)(flatten)
    fc1 = SReLU()(fc1)
    dp1 = Dropout(0.7)(fc1)
    
    outputs = Dense(1,activation='sigmoid',name='outputs')(dp1)

    inputs = [
                inputs_p,
                inputs_d,
            ]



    model = Model(input=inputs, output=outputs)
    nadam = Nadam()
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
                optimizer=nadam,
                loss= 'binary_crossentropy'
              )

    return model

mf_nn_clf = make_mf_nn(X ,y, X_t, n_round=10)
pd.to_pickle(mf_nn_clf,path+'mf_nn_clf.pkl')
