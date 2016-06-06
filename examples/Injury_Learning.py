
# coding: utf-8

# In[183]:




import cPickle, os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from ddlite import *

matplotlib.rcParams['figure.figsize'] = (18,6)


# In[184]:

R = Relations('/Users/Jerry/Documents/CMPS290H/Project/data/temp/relation_saved_send_v1.pkl')


# In[185]:

r = R[0]


# In[186]:

DDL = DDLiteModel(R)
print "Extracted {} features for each of {} mentions".format(DDL.num_feats(), DDL.num_candidates())


# In[187]:

DDL.open_mindtagger(num_sample=20, width='100%', height=1200)


# In[188]:

DDL.add_mindtagger_tags()
gold = np.zeros((DDL.num_candidates()))
gold[np.array([1,5,16,25,32,33,39,40,49,50,51,52])] = np.array([
     1,1,1,1,-1,1,-1,-1,1,-1,-1,-1])
DDL.set_gold_labels(gold)
DDL.set_holdout(p=0.8)


# ## Define Label functions

# In[189]:

## load dictionary for state

get_injury_state = [line.rstrip() for line in open('../../../data/dictionary/state_dic_1.tsv')]
is_recover_state = [line.rstrip() for line in open('../../../data/dictionary/state_dic_recover.tsv')]
is_aggrevated_state = [line.rstrip() for line in open('../../../data/dictionary/state_dic_aggregate.tsv')]
is_return_state = [line.rstrip() for line in open('../../../data/dictionary/state_dic_return.tsv')]
print is_return_state
state_dm = DictionaryMatch(label='State', dictionary=get_injury_state, ignore_case=False)


# In[190]:

for r in R:
    print r.lemmas[r.e1_idxs[-1] : r.e2_idxs[0]] 


# In[191]:

## For learning 


def get_injury(r):
    for x in get_injury_state:
        if x in r.lemmas[r.e1_idxs[-1] : r.e2_idxs[0]]:
            return 1
    return 0

def is_recover(r):
    for x in is_recover_state:
        if x in r.lemmas[r.e1_idxs[-1] : r.e2_idxs[0]]:
            return -1
    return 0

def is_aggrevate(r):
    for x in is_aggrevated_state:
        if x in r.lemmas[r.e1_idxs[-1] : r.e2_idxs[0]]:
            return -1
    return 0

def is_return(r):
    for x in is_return_state:
        if x in r.lemmas[r.e1_idxs[-1] : r.e2_idxs[0]]:
            return -1
    return 0

def LF_mutant(m):
    return 1 if ('mutant' in m.post_window('lemmas')) or ('mutant' in m.pre_window('lemmas')) else 0
def LF_express(m):
    return 1 if ('express' in m.post_window('lemmas')) or ('express' in m.pre_window('lemmas')) else 0
def LF_mutation(m):
    return 1 if 'mutation' in [m.lemmas[m.dep_parents[i] - 1] for i in m.idxs] else 0
def LF_dna(m):
    return -1 if 'dna' in m.mention('lemmas') else 0
def LF_rna(m):
    return -1 if 'rna' in m.mention('lemmas') else 0
def LF_snp(m):
    return -1 if 'snp' in m.mention('lemmas') else 0


# In[192]:

LFs = [get_injury,is_recover,is_aggrevate,is_return]
DDL.apply_lfs(LFs, clear=True)


# In[193]:

DDL.print_lf_stats()


# In[194]:

DDL.plot_lf_stats()


# In[195]:

matplotlib.rcParams['figure.figsize'] = (12,4)
DDL.learn_weights(sample=False, maxIter=500, alpha=0.5, verbose=True, log=True)


# In[114]:

DDL.plot_calibration()


# In[ ]:




# In[115]:

DDL.show_log()


# In[117]:

DDL.open_mindtagger(width='100%', height=1200)


# In[118]:

predicted = DDL.get_predicted_probability()


# In[86]:

DDL.w[0]


# In[ ]:




# In[ ]:



