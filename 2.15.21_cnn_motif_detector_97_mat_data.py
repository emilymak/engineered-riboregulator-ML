# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 10:24:30 2021

@author: makow
"""

from sklearn.preprocessing import OneHotEncoder
from nlp_wvec_utils import *
import logomaker as lm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import scipy as sc


#%%
r2_rep1 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.6.21_97_mat_NGS\\R2_rep1.csv", header = None, index_col = None)
indices = []
for index, row in r2_rep1.iterrows():
    if len(list(row[0])) != 141:
        indices.append(index)
    if '*' in list(row[0]):
        indices.append(index)
    if 'X' in list(row[0]):
        indices.append(index)
r2_rep1.drop(r2_rep1.index[indices], inplace = True)
r2_rep1.set_index(0, inplace = True)
r3_rep1 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.6.21_97_mat_NGS\\R3_rep1.csv", header = None, index_col = None)
indices = []
for index, row in r3_rep1.iterrows():
    if len(list(row[0])) != 141:
        indices.append(index)
    if '*' in list(row[0]):
        indices.append(index)
    if 'X' in list(row[0]):
        indices.append(index)
r3_rep1.drop(r3_rep1.index[indices], inplace = True)
r3_rep1.set_index(0, inplace = True)
r5_rep1 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.6.21_97_mat_NGS\\R5_rep1.csv", header = None, index_col = None)
indices = []
for index, row in r5_rep1.iterrows():
    if len(list(row[0])) != 141:
        indices.append(index)
    if '*' in list(row[0]):
        indices.append(index)
    if 'X' in list(row[0]):
        indices.append(index)
r5_rep1.drop(r5_rep1.index[indices], inplace = True)
r5_rep1.set_index(0, inplace = True)
r6_rep1 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.6.21_97_mat_NGS\\R6_rep1.csv", header = None, index_col = None)
indices = []
for index, row in r6_rep1.iterrows():
    if len(list(row[0])) != 141:
        indices.append(index)
    if '*' in list(row[0]):
        indices.append(index)
    if 'X' in list(row[0]):
        indices.append(index)
r6_rep1.drop(r6_rep1.index[indices], inplace = True)
r6_rep1.set_index(0, inplace = True)
r7_rep1 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.6.21_97_mat_NGS\\R7_rep1.csv", header = None, index_col = None)
indices = []
for index, row in r7_rep1.iterrows():
    if len(list(row[0])) != 141:
        indices.append(index)
    if '*' in list(row[0]):
        indices.append(index)
    if 'X' in list(row[0]):
        indices.append(index)
r7_rep1.drop(r7_rep1.index[indices], inplace = True)
r7_rep1.set_index(0, inplace = True)


r2_rep2 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.6.21_97_mat_NGS\\R2_rep2.csv", header = None, index_col = None)
indices = []
for index, row in r2_rep2.iterrows():
    if len(list(row[0])) != 141:
        indices.append(index)
    if '*' in list(row[0]):
        indices.append(index)
    if 'X' in list(row[0]):
        indices.append(index)
r2_rep2.drop(r2_rep2.index[indices], inplace = True)
r2_rep2.set_index(0, inplace = True)
r3_rep2 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.6.21_97_mat_NGS\\R3_rep2.csv", header = None, index_col = None)
indices = []
for index, row in r3_rep2.iterrows():
    if len(list(row[0])) != 141:
        indices.append(index)
    if '*' in list(row[0]):
        indices.append(index)
    if 'X' in list(row[0]):
        indices.append(index)
r3_rep2.drop(r3_rep2.index[indices], inplace = True)
r3_rep2.set_index(0, inplace = True)
r5_rep2 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.6.21_97_mat_NGS\\R5_rep2.csv", header = None, index_col = None)
indices = []
for index, row in r5_rep2.iterrows():
    if len(list(row[0])) != 141:
        indices.append(index)
    if '*' in list(row[0]):
        indices.append(index)
    if 'X' in list(row[0]):
        indices.append(index)
r5_rep2.drop(r5_rep2.index[indices], inplace = True)
r5_rep2.set_index(0, inplace = True)
r6_rep2 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.6.21_97_mat_NGS\\R6_rep2.csv", header = None, index_col = None)
indices = []
for index, row in r6_rep2.iterrows():
    if len(list(row[0])) != 141:
        indices.append(index)
    if '*' in list(row[0]):
        indices.append(index)
    if 'X' in list(row[0]):
        indices.append(index)
r6_rep2.drop(r6_rep2.index[indices], inplace = True)
r6_rep2.set_index(0, inplace = True)
r7_rep2 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.6.21_97_mat_NGS\\R7_rep2.csv", header = None, index_col = None)
indices = []
for index, row in r7_rep2.iterrows():
    if len(list(row[0])) != 141:
        indices.append(index)
    if '*' in list(row[0]):
        indices.append(index)
    if 'X' in list(row[0]):
        indices.append(index)
r7_rep2.drop(r7_rep2.index[indices], inplace = True)
r7_rep2.set_index(0, inplace = True)


neg = pd.concat([r2_rep1, r2_rep2], axis = 1, ignore_index = False)
neg.dropna(inplace = True)
all_pos = pd.concat([r3_rep1, r3_rep2, r5_rep1, r5_rep2, r6_rep1, r6_rep2, r7_rep1, r7_rep2], axis = 0, ignore_index = True)
neg = neg[~neg.index.isin(all_pos.index)]
neg.reset_index(drop = False, inplace = True)

pos = pd.concat([r7_rep1, r7_rep2], axis = 1, ignore_index = False)
pos.fillna(0,inplace = True)
pos.reset_index(drop = False, inplace = True)

neg = neg.iloc[0:10000,:]
neg['label'] = [0]*10000
neg.columns = ['sequence', 'frequency2_rep1', 'frequency2_rep2', 'label']
pos = pos.iloc[0:10000,:]
pos['label'] = [1]*10000
pos.columns = ['sequence', 'frequency7_rep1','frequency7_rep2', 'label']

df_classify = pd.concat([pos, neg])


wt = list('KAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHSTYPPTFGQGTKVEIKRTSPNSASHSGSAPQTSSAPGSQEVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIY')
mutations = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.6.21_97_mat_NGS\\2.10.21_clone_mutations_binding.csv", header = 0, index_col = 0)

clones = pd.Series(mutations.iloc[:,2])


#%%
ngram_size = 5
ngram_stride = 2
dim_embedding = 400
alph_letters = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
le = LabelEncoder()
integer_encoded_letters = le.fit_transform(alph_letters)
integer_encoded_letters = integer_encoded_letters.reshape(len(integer_encoded_letters), 1)

df_classify_seq = df_classify['sequence']
sequences= []
for i in df_classify_seq:
    chars = le.transform(list(i))
    sequences.append(chars)
clones_enc= []
for i in clones:
    chars = le.transform(list(i))
    clones_enc.append(chars)

sequences = pd.DataFrame(sequences)
clones_enc = pd.DataFrame(clones_enc)



one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_sequences = []
for index, row in sequences.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = one.transform(let)
    ohe_sequences.append(ohe_let)
ohe_clones = []
for index, row in clones_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = one.transform(let)
    ohe_clones.append(ohe_let)

ohe_sequences = np.stack(ohe_sequences)
ohe_clones = np.stack(ohe_clones)


y = np.array(df_classify['label'].astype(np.float32))
y = np.transpose(np.array([y]))
print('target shape: ', y.shape)

df_classify, embeddings_index =  train_sg_embedding(df_classify = df_classify, 
                                                    ngram_size = ngram_size,
                                                    ngram_stride = ngram_stride, 
                                                    dim_embedding = dim_embedding)

x_train, x_test, y_train, y_test = train_test_split(ohe_sequences, y, test_size = 0.2)
print(ohe_sequences.shape)

#%%
from nlp_wvec_utils import *
dim_embedding = 400
filters = 10
batch_size = 256
dropout_rate = 0.2
l2_reg_coeff = 0.001
learning_rate = 0.0005
num_features = len(chars)
num_epochs = 20
num_channels = 20

conv_layer_parameters = [(5,50), (3,5)]
hidden_layer_choices = {5: (150, 60, 15),}
hidden_layers = hidden_layer_choices[5]


model = twoheaded_conv1d(conv_layer_parameters = conv_layer_parameters, hidden_layers = hidden_layers, 
                         dropout_rate = 0.3, reg_coeff = l2_reg_coeff, 
                         learning_rate = learning_rate, num_features = num_features, num_channels = num_channels)

print(model.summary())

filter_names = list(np.arange(0,100))
accuracy_test = []
accuracy_train = []
clones_predictions = []
filter_vals = []
for i in filter_names:
    model.fit(ohe_sequences, y, epochs = num_epochs, batch_size = 128, verbose = 2, validation_data = (x_test, y_test))
    predictions_test = pd.DataFrame(model.predict(x_test))
    predictions_train = pd.DataFrame(model.predict(x_train))
    clones_predictions.append(model.predict(ohe_clones))
    predictions_test['label'] = 0
    for index, row in predictions_test.iterrows():
        if predictions_test.iloc[index,0] > 0.5:
            predictions_test.loc[index, 'label'] = 1
    predictions_train['label'] = 0
    for index, row in predictions_train.iterrows():
        if predictions_train.iloc[index,0] > 0.5:
            predictions_train.loc[index, 'label'] = 1
    accuracy_test.append(accuracy_score(predictions_test['label'], y_test))
    accuracy_train.append(accuracy_score(predictions_train['label'], y_train))
    filter_weights, bw = analyze_filters_in_model(model, 'fig')
    filter_vals.append(filter_weights)

print((np.mean(accuracy_train), np.std(accuracy_train)))
print((np.mean(accuracy_test), np.std(accuracy_test)))


#%%
filter_vals_stack = []
for i in filter_names:
    weights = np.transpose(filter_vals[i])
    for j in np.arange(0,conv_layer_parameters[0][1]):
        filter_vals_stack.append(weights[j])

clones_predictions_ave = pd.DataFrame(np.hstack(clones_predictions))

for i in filter_names:
    plt.scatter(clones_predictions_ave.iloc[:,i], mutations.iloc[:,1])
    print(sc.stats.spearmanr(clones_predictions_ave.iloc[:,i], mutations.iloc[:,1], nan_policy = 'omit'))


#%%
from collections import OrderedDict
alph_letters = sorted('ACDEFGHIKLMNPQRSTVWY')
alph = list(alph_letters)
dict_of_df = dict((f, pd.DataFrame(filter_vals_stack[f].T, columns = alph)) for f in np.arange(0, len(filter_vals_stack)))

all_pwms = dict_of_df.values()
filt_width = 5
num_nucleotides = 20
list_of_seqs = []
for mat in all_pwms:
    seq = ''
    for row in range(0, len(mat.index)):
        curr_row = mat.iloc[row,:]
        max_col = curr_row.idxmax(axis=1) # to get the minimum, use max_col = curr_row.idxmin(axis=1)
        seq = seq + str(max_col)
    list_of_seqs.append(seq)

threemer_dict = {}
for seq in list_of_seqs:
    for i in range (0, 3):
        mer = seq[i:(i+3)]
        if mer in threemer_dict:
            threemer_dict[mer] = threemer_dict.get(mer) + 1
        else:
            threemer_dict[mer] = 1

keys = []
values = []
for key in sorted(threemer_dict.keys()):
    keys.append(key)
    values.append(threemer_dict[key])
df = pd.DataFrame({'keys': keys, 'values': values})


#%%
filter_max = np.max(df['values'])
best_threemers = []
for index, row in df.iterrows():
    if row[1] == filter_max:
        best_threemers.append(row[0])

threes = []
for index, row in df_classify.iterrows():
    if best_threemers[0] in row[0]:
        threes.append(1)
    else:
        threes.append(0)
        
df_classify['threemer'] = threes


#%%
# make empty array of 64 (# of possible kmers) x # bootstraps to fill in
num_bootstraps = 1000
kmer_len = 3
kmer_lib_size = 20 ** kmer_len
number_of_kmers_to_compare = 600

kmer_occurrences = np.zeros((kmer_lib_size,num_bootstraps))

# for i in range(0, number of bootstraps)
for i in range(0, num_bootstraps):
    for j in range(0, number_of_kmers_to_compare):
        # extract 600 kmers by randomly generating first the row # to extract the toehold from 
        get_rand_seq = df_classify.iloc[random.randrange(0, len(df_classify)), df_classify.columns.get_loc("sequence")]
        # and second the position to start the kmer at
        get_rand_kmer_start = random.randrange(0, len(get_rand_seq)-3)
        new_kmer = get_rand_seq[get_rand_kmer_start:get_rand_kmer_start+3]
        # now put the kmer into the proper position in the kmer_occurrences array
        kmer_index = keys.index(new_kmer)
        curr_kmer = kmer_occurrences[kmer_index, i]
        kmer_occurrences[kmer_index, i] = curr_kmer + 1

# then get the average and std dev of each kmer
avg_kmer_occurrence = np.mean(kmer_occurrences, axis = 1)
std_dev_kmer_occurrence = np.std(kmer_occurrences, axis = 1)

#%%
predictions = pd.DataFrame(model.predict(ohe_sequences).astype('float64'))
predictions['label'] = 0
for index, row in predictions.iterrows():
        if predictions.iloc[index,0] > 0.5:
            predictions.loc[index, 'label'] = 1

df_classify['predictions'] = predictions.iloc[:,0]
df_classify['predicted label'] = predictions.iloc[:,1]
df_swarm = df_classify.sample(200)

print(accuracy_score(predictions.iloc[:,1], df_classify.iloc[:,3]))
sns.swarmplot(df_swarm.iloc[:,3], df_swarm.iloc[:,10], hue = df_swarm.iloc[:,9])

print(sc.stats.ks_2samp(predictions.iloc[0:10000,0], predictions.iloc[10000:20000,0]))

sns.kdeplot(predictions.iloc[0:10000,0])
sns.kdeplot(predictions.iloc[10000:20000,0])


