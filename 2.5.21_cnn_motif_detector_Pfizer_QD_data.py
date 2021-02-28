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


#%%
qd_pos = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.2.21_Pfizer_QD_NGS\\Boco_VH_QD_P_rep1.csv", header = None, index_col = None)
pos_indices = []
for index, row in qd_pos.iterrows():
    if len(list(row[0])) != 118:
        pos_indices.append(index)
    if '*' in list(row[0]):
        pos_indices.append(index)
qd_pos.drop(qd_pos.index[pos_indices], inplace = True)
qd_pos.set_index(0, inplace = True)
qd_pos.columns = ['Frequency']
qd_pos_index = qd_pos.index
qd_neg = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.2.21_Pfizer_QD_NGS\\Boco_VH_QD_N_rep1.csv", header = None, index_col = None)
neg_indices = []
for index, row in qd_neg.iterrows():
    if len(list(row[0])) != 118:
        neg_indices.append(index)
    if '*' in list(row[0]):
        neg_indices.append(index)
qd_neg.drop(qd_neg.index[neg_indices], inplace = True)
qd_neg.set_index(0, inplace = True)
qd_neg.columns = ['Frequency']
qd_neg_index = qd_neg.index

qd_pos = qd_pos[~qd_pos_index.isin(qd_neg_index)]

qd_neg = qd_neg[~qd_neg_index.isin(qd_pos_index)]

qd_pos = qd_pos.iloc[0:5000,:]
qd_pos.reset_index(drop = False, inplace = True)
qd_pos_label = [1]*5000
qd_pos['label'] = qd_pos_label
qd_pos.columns = ['sequence', 'frequency', 'label']

qd_neg = qd_neg.iloc[0:5000,:]
qd_neg.reset_index(drop = False, inplace = True)
qd_neg_label = [0]*5000
qd_neg['label'] = qd_neg_label
qd_neg.columns = ['sequence', 'frequency', 'label']


#%%
#df_classify = pd.concat([qd_pos, qd_neg])
df_classify = pd.concat([qd_pos, qd_neg])


#%%
# parameters to train the skip-gram embedding (takes a long time)
ngram_size = 5
ngram_stride = 2
dim_embedding = 400

alph_letters = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
le = LabelEncoder()
integer_encoded_letters = le.fit_transform(alph_letters)

df_classify_seq = df_classify['sequence']
sequences= []
for i in df_classify_seq:
    chars = le.transform(list(i))
    sequences.append(chars)

sequences = pd.DataFrame(sequences)
    
one = OneHotEncoder(sparse = False)
integer_encoded_letters = integer_encoded_letters.reshape(len(integer_encoded_letters), 1)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_sequences = []
for index, row in sequences.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = one.transform(let)
    ohe_sequences.append(ohe_let)

ohe_sequences = np.stack(ohe_sequences)

y = np.array(df_classify['label'].astype(np.float32))
y = np.transpose(np.array([y]))
print('target shape: ', y.shape)


# train the skip-gram embedding and save it as a text file 
df_classify, embeddings_index =  train_sg_embedding(df_classify = df_classify, 
                                                    ngram_size = ngram_size,
                                                    ngram_stride = ngram_stride, 
                                                    dim_embedding = dim_embedding)

# split the data into training and testing sets and get the embedding lookup
#x_train, x_test, y_train, y_test, embedding_matrix, max_length = create_ds_splits(df_classify = ohe_sequences, 
#                                                                                  embeddings_index = embeddings_index, 
#                                                                                  dim_embedding = dim_embedding)

x_train, x_test, y_train, y_test = train_test_split(ohe_sequences, y, test_size = 0.2)
filter_vals = []


#%%
"""
def create_cnn_lstm_model(embedding_matrix = embedding_matrix, max_length = max_length, neurons = 64,
                          dropout_rate1 = 0.0, dropout_rate2 = 0.0, weight_constraint = 0):

    
    num_words = embedding_matrix.shape[0]
    dim_embedding = embedding_matrix.shape[1]
    
    # instantiate model
    model = Sequential()
    embedding_layer = Embedding(input_dim = num_words, output_dim = dim_embedding, 
                                weights = [embedding_matrix],
                                input_length = max_length,
                                trainable = False)
    model.add(embedding_layer)
    model.add(Dropout(dropout_rate1))
    model.add(Conv1D(filters = filters, kernel_size = 2, padding = 'same', activation = 'relu'))
    model.add(MaxPooling1D(pool_size = 4))
    model.add(Dropout(dropout_rate2))
    model.add(LSTM(units = neurons, kernel_constraint = max_norm(weight_constraint)))
    model.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'he_uniform')) # binary classification 
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
 
    return model

"""
#%%
from nlp_wvec_utils import *
filters = 10
batch_size = 128
dropout_rate = 0.2
l2_reg_coeff = 0.0001
learning_rate=0.001
num_features = len(chars)
num_epochs = 20

conv_layer_parameters = [(5,10), (3,5)]
hidden_layer_choices = {5: (150, 60, 15),} # dependent on # filters in final convolutional layer before MLP 

hidden_layers = hidden_layer_choices[5]

model = twoheaded_conv1d(conv_layer_parameters = conv_layer_parameters, num_features = num_features, num_channels = 20, hidden_layers = hidden_layers, 
                         dropout_rate = dropout_rate, reg_coeff = l2_reg_coeff, 
                         learning_rate = learning_rate)

print(model.summary())

#filter_names = list(np.arange(0,250))
filter_names = list(np.arange(0,5))
accuracy = []
for i in filter_names:
    model.fit(x_train, y_train,epochs = num_epochs, batch_size = 128, verbose = 2, validation_data = (x_test, y_test))
    predictions = pd.DataFrame(model.predict(x_test))
    predictions['label'] = 0
    for index, row in predictions.iterrows():
        if predictions.iloc[index,0] > 0.5:
            predictions.loc[index, 'label'] = 1
    accuracy.append(accuracy_score(predictions['label'], y_test))
    filter_weights, bw = analyze_filters_in_model(model, 'fig')
    filter_vals.append(filter_weights)

print((np.mean(accuracy), np.std(accuracy)))

training_prediction = model.predict(ohe_sequences)

#%%
seqs_pI = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\2.2.21_Pfizer_QD_NGS\\2.22.21_df_classify_seqs.csv", header = None, index_col = 0)
df_classify.reset_index(drop = True, inplace = True)
seqs_pI.reset_index(drop = True, inplace = True)
df_classify = pd.concat([df_classify, seqs_pI], axis = 1)
df_swarm = df_classify.sample(2000)
sns.swarmplot(df_swarm.iloc[:,2], df_swarm.iloc[:,5])


#%%
filter_vals = np.vstack(filter_vals)


#%%
from collections import OrderedDict
alph_letters = sorted('ACDEFGHIKLMNPQRSTVWY')
alph = list(alph_letters)
dict_of_df = OrderedDict((f, pd.DataFrame(filter_vals[f], index = alph).T) for f in filter_names)

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
    for i in range (0, 7):
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





