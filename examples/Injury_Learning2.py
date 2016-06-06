
# coding: utf-8

# In[18]:



import cPickle, os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from ddlite import *
matplotlib.rcParams['figure.figsize'] = (18,6)


# In[24]:

R = Relations('/Users/Jerry/Documents/CMPS290H/Project/data/temp/relation_saved_send_v3.pkl')
a = R[1]
a.ners


# In[23]:

a.doc_id


# In[45]:

## load dictionary for state

get_injury_state = [line.rstrip() for line in open('../../../data/dictionary/state_dic_1.tsv')]
is_recover_state = [line.rstrip() for line in open('../../../data/dictionary/state_dic_recover.tsv')]
is_aggrevated_state = [line.rstrip() for line in open('../../../data/dictionary/state_dic_aggregate.tsv')]
is_return_state = [line.rstrip() for line in open('../../../data/dictionary/state_dic_return.tsv')]
state_dm = DictionaryMatch(label='State', dictionary=get_injury_state, ignore_case=False)


# In[46]:

DDL = CandidateModel(R)
print "Extracted {} features for each of {} mentions".format(DDL.num_feats(), DDL.num_candidates())


# In[47]:

DDL.open_mindtagger(num_sample=40, width='100%', height=1200)


# In[193]:

i=0
for r in R:
    print i
    print r.lemmas[r.e1_idxs[-1] : r.e2_idxs[0]]
    
    i=i+1


# In[121]:

groud_truth_path = '/Users/Jerry/Documents/CMPS290H/Project/data/temp/ground_truth_2.pkl'

try:
    with open(groud_truth_path, 'rb') as f:
        gt = cPickle.load(f)
        DDL.update_gt(gt[1],gt[0])
except:
    with open(groud_truth_path, 'w+') as f:
        cPickle.dump(DDL.get_labeled_ground_truth(), f)

DDL.set_holdout(validation_frac=0.3)


# ## Define Label functions

# In[98]:

## For learning 
verb_tags = ['VB','VBD','VBN','VBP','VBZ','VBG']
noun_tags = ['NN','NNS','NNPS','NNP']


def general_too_faraway(r): return -1 if (r.e1_idxs[-1] - r.e2_idxs[0])>10 else 0
def general_pre_limit(r): return 1 if ('right' or 'left') in r.get_attr_seq('lemmas',range(min(r.e2_idxs)-4,min(r.e2_idxs))) else 0
def general_not_noun(r):
    for i in r.get_attr_seq('poses', r.e2_idxs):
        if i not in noun_tags: return -100
    return 1
def another_name_between(r): return -10 if ('PERSON' in r.get_attr_seq('ners',range(max(r.e1_idxs)+1,min(r.e2_idxs)))) else 0
def remove_head_coach(r): return -100 if (('coach' in r.post_window2('lemmas',1)) or ('coach' in r.post_window1('lemmas',1))) else 0
general_LFs = [general_pre_limit,general_too_faraway,general_not_noun,remove_head_coach,another_name_between]
key_word = get_injury_state+is_recover_state+is_aggrevated_state+is_return_state
def get_key_word(r): 
    for x in key_word:
        if x in r.lemmas[r.e1_idxs[-1] : r.e2_idxs[0]]:
            return 10
    return 0

def get_injury(r):
    for x in get_injury_state:
        if x in r.lemmas[r.e1_idxs[-1] : r.e2_idxs[0]]:
            return 10
    return 0

def is_recover(r):
    for x in is_recover_state:
        if x in r.lemmas[r.e1_idxs[-1] : r.e2_idxs[0]]:
            return 10
    return 0

def is_aggrevate(r):
    for x in is_aggrevated_state:
        if x in r.lemmas[r.e1_idxs[-1] : r.e2_idxs[0]]:
            return 10
    return 0

def is_return(r):
    for x in is_return_state:
        if x in r.lemmas[r.e1_idxs[-1] : r.e2_idxs[0]]:
            return 10
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


# ## Test Label functions

# In[99]:

another_name_between(R[1])


# In[100]:

#LFs = [get_injury,is_recover,is_aggrevate,is_return] + general_LFs
LFs = general_LFs+[get_key_word]
DDL.apply_lfs(LFs, clear=True)


# In[101]:

general_pre_limit(a)


# In[102]:

DDL.print_lf_stats()


# In[103]:

DDL.plot_lf_stats()


# In[111]:

matplotlib.rcParams['figure.figsize'] = (12,4)
mu_seq = np.ravel([0.1, 0.01, 0.001])
DDL.set_use_lfs(True)
get_ipython().magic(u'time DDL.learn_weights(sample=False, n_iter=500, alpha=0.5, mu=mu_seq,                        bias=True, verbose=True, log=True)')


# In[83]:

DDL.lf_summary_table()


# In[84]:

DDL.get_classification_accuracy()


# In[85]:

DDL.show_log()


# In[86]:

DDL.open_mindtagger(width='100%', height=1200)


# In[88]:

predicted = DDL.get_predicted_probability()


# In[89]:

predicted


# In[ ]:




# In[ ]:



