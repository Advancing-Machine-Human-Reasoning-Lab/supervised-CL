import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd

# read the dev set
df = pd.read_json("../data/snli_1.0/snli_1.0_dev.jsonl",lines=True) # update this line with wherever you have SNLI stored

agreement = [labels.count(gold_label) for labels, gold_label in zip(list(df['annotator_labels'].values), list(df['gold_label'].values)) if gold_label != '' and labels]
# calculate the agreement

agr = pd.DataFrame(agreement, columns=['agreement'])

# apparently 0.0, 0.6, 0.8, 1.0 are the only possible options, so mapping them to classes
labels = {'key_0.0': 0, 'key_0.6': 1, 'key_0.8': 2, 'key_1.0': 3}

# create the dataframe
df = pd.DataFrame({
    'pairID': df['pairID'],
    'text_a': df['sentence1'],
    'text_b': df['sentence2'],
    'labels': agr['agreement']/5
})

# get the embeedings
vec_df = pd.read_json('../data/SNLI_roberta_vectors.jsonl',lines=True)

feature_set = {
    "from_AAAI":np.asarray([]),
    "just_rob_embed_flattened":np.asarray([]),
    "just_rob_embed_stacked":np.asarray([]),
    "rob_embed_stacked_sent_len":np.asarray([]),
    "rob_embed_stacked_cdist":np.asarray([])
    }
print("creating feature set")
X = []
print("from_AAAI")
for index in range(0,len(df)):
    # create the feature vector for each and append
    question = df.iloc[index]
    vec_premise = vec_df.loc[vec_df['qid'] == question['pairID']]["premise"]
    vec_hypothesis = vec_df.loc[vec_df['qid'] == question['pairID']]["hypothesis"]
    try:
        vec_premise = vec_premise.iloc[0]
        vec_hypothesis = vec_hypothesis.iloc[0]
    except KeyError:
        print(vec_premise)
        exit()
    # 1. from AAAI
    thisX = []
    # may be more than 1 due to oversampling
    thisX.append(cdist([vec_premise],[vec_hypothesis],"cosine")[0][0])
    thisX.append(float(len(question['text_a'])))
    thisX.append(float(len(question['text_b'])))
    thisX.append(float(len(question['text_a'].split())))
    thisX.append(float(len(question['text_b'].split())))
    X.append(thisX)
feature_set['from_AAAI'] = np.asarray(X)


torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

x = feature_set['from_AAAI']
y = np.array(df['labels'].values)
y = y.reshape((len(y), 1))

X = torch.from_numpy(x)
Y = torch.from_numpy(y)

kfold = KFold(10, True, 1)




hidden_layers = [512, 1024, 2048]
epochs = [2500, 5000, 10000, 20000]
lrs = [1e-4, 1e-5, 5e-6, 1e-6]
n_hidden_layers = [2,3,4,5]


                
for hidden_layer in hidden_layers:
    for epoch in epochs:
        for lr in lrs:
            for n_hidden_layer in n_hidden_layers:
                file = open("../logs/dnn_regression_snli.txt","a")
                result = f'Hidden layers: {hidden_layer} Epoch: {epoch} LR: {lr} n_layers {n_hidden_layers} \n'
                file.write(result)
                file.write('\n')
                # this is one way to define a network
                class Net(torch.nn.Module):
                    def __init__(self, n_feature, n_hidden, n_output):
                        super(Net, self).__init__()
                        self.hidden = torch.nn.Linear(n_feature, n_hidden)
                        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

                    def forward(self, x):
                        x = F.relu(self.hidden(x))
                        for i in range(0, n_hidden_layer-1):
                            x = F.relu(x)      # activation function for hidden layer
                        x = self.predict(x)             # linear output
                        return x


                for train, test in kfold.split(X,Y):
                    x = X[train]
                    y = Y[train]

                    x, y = Variable(x), Variable(y)

                    net = Net(n_feature=5, n_hidden=hidden_layer, n_output=1)     # define the network
                    # print(net)  # net architecture
                    #optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
                    learning_rate = lr
                    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
                    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

                    # train the network
                    for t in range(epoch):

                        prediction = net(x.float())     # input x and predict based on x

                        loss = loss_func(prediction, y.float())     # must be (1. nn output, 2. target)

                        optimizer.zero_grad()   # clear gradients for next train
                        loss.backward()         # backpropagation, compute gradients
                        optimizer.step()        # apply gradients


                    test_x, test_y = Variable(X[test]), Variable(Y[test])
                    preds = net(test_x.float())
                    preds = preds.float().detach().numpy()
                    test_y = test_y.float().detach().numpy()

                    file.write(str(spearmanr(test_y, preds)))
                    file.write('\n')
                   # file.write(str(pearsonr(test_y, preds)))
                   # file.write('\n')
                    
            file.write('\n')
            file.close()
            
