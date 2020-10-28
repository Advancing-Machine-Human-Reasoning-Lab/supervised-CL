import json
from scipy.spatial.distance import cdist
import numpy as np

print('\n\n\n\n==========NEW RUN=================')

with open("aNLI_train_set_shuffled.jsonl") as F:
	anliData = [json.loads(l) for l in F.readlines()[:1014]]

with open("train-labels.lst") as F:
	idToAnswer = {i:int(s) for (i,s) in enumerate(F.readlines())}

idToSentences = dict()
idPairsToVotes = dict()

# idToSentences['difficult'] = {'s1':"The first principle taught to students of a collegiate persuasion fundamentally concerns itself with the rights of man.",
#  	's2':"This formulaic advice often conflicts with that given to them by previous educators, raising concerns of methodological incompetence.", 
#  	'h1':"Whether this is purposeful or not, cannot be determined by the principal actors.", 
#  	'h2':"It behooves us to consider the reasons for this apparent drop in standards as compared to those of our forefathers."}
# idToSentences['easy'] = {'s1':"The man in the black hat was walking down the street",
#  	's2':"The man no longer had a hat on.", 
#  	'h1':"The man took his hat off.", 
#  	'h2':"He continued walking down the street."}


# from sentence_transformers import SentenceTransformer #https://github.com/UKPLab/sentence-transformers
# model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
# model = SentenceTransformer('bert-base-nli-mean-tokens')
for i in range(1014):
	#For each problem:
	thisId = anliData[i]['index']
	s1 = anliData[i]['obs1']
	s2 = anliData[i]['obs2']
	hyp1 = anliData[i]['hyp1']
	hyp2 = anliData[i]['hyp2']
	idToSentences[thisId] = {'s1':s1, 's2':s2, 'h1':hyp1, 'h2':hyp2}
	se = model.encode([s1,s2,hyp1,hyp2])
	with open("anli_vectors_big.jsonl", 'a') as F:
		F.write(json.dumps({'id':thisId, 's1':se[0].tolist(), 's2':se[1].tolist(), 'h1':se[2].tolist(), 'h2':se[3].tolist()}) + '\n')

#add the attention check questions
#Q21 - impossible question
#Q22 - impossible question difficulty
#Q23 - easy question
#Q24 - easy question difficulty

#go through all results from qualtrics
#tsv file should be formatted so that:
#- the first few columns until the 'QIDq...' column should be deleted
#- delete rows 2 and 3
#- make sure the final row is "Random ID", second-to-final column is QIDs(a number)x(a number).
with open("qualtricsResults_fewerQuestions.tsv") as F:
	data = [l.split('\t') for l in F.readlines()]
	completionCodes = [row.pop(-1) for row in data][1:]
	header = data.pop(0)

#grade the users
plot = []
exclude = set() #users to exclude because of very low score
for (userId,row) in enumerate(data):
	correct = [] #records problems this user got correct (1) vs wrong (0)
	for (i,h) in enumerate(header):
		if ('QIDq' in h or 'QIDr' in h) and row[i].strip()!='':
			guess = int(row[i])
			if guess==idToAnswer[int(h[4:])]:
				correct.append(1)
			else:
				correct.append(0)
	score = sum(correct)*1.0/len(correct)
	# print("User", userId, "got", sum(correct), "of", len(correct), ":", score
	plot.append(score)
	if score < 0.6:
		exclude.add(userId)
		continue
		# if completionCodes[userId].strip()!='':
		# 	with open("usersToNotPay.txt",'a') as F:
		# 		F.write(completionCodes[userId].strip() + ' Score was only ' + str(score*100) + '%. Random choices yield 50%.' + '\n')

#now go through again and record only the choices made by those who had decent scores
correctGuesses = dict()
for (i,h) in enumerate(header):
	if 'QIDq' not in h and 'QIDr' not in h:
		continue
	problemId = int(h[4:])
	for (userId,row) in enumerate(data):
		if userId in exclude or row[i].strip()=='':
			# print(userId in exclude)
			# print(row[i])
			continue
		#did this user answer correctly?
		if problemId not in correctGuesses:
			correctGuesses[problemId] = []
		if int(row[i])==idToAnswer[problemId]:
			correctGuesses[problemId].append(1)
		else:
			correctGuesses[problemId].append(0)
for k in correctGuesses:
	correctGuesses[k] = sum(correctGuesses[k])*1.0/len(correctGuesses[k])


#visualize the accuracies
# import matplotlib.pyplot as plt
# print(len([a for a in accuracies if a>0.9]), len([a for a in accuracies if a<=0.9]))
# plt.hist(accuracies, bins=10)
# plt.show()
# exit()

#load vectors from file
idToVectors = dict()
with open("anli_vectors_big.jsonl") as F:
	for l in F.readlines():
		obj = json.loads(l)
		idToVectors[obj['id']] = obj
		# idToVectors[obj['id']] = obj['h1'] + obj['h2']


#create all of the inputs for training
X = []
y = []

from numpy import std
allStds = []
for problemId in correctGuesses:
	if problemId not in idToVectors:
		continue
	thisX = []
	#vector distances
	def addDist(vec1,vec2):
		thisX.append(cdist([vec1],[vec2],"cosine")[0][0])
	v = idToVectors[problemId]
	addDist(v['s1'], v['s2'])
	addDist(v['s1'], v['h1'])
	addDist(v['s1'], v['h2'])
	addDist(v['s2'], v['h1'])
	addDist(v['s2'], v['h2'])
	addDist(v['h1'], v['h2'])
	for sentenceType in ['s1', 's2', 'h1', 'h2']:
		thisX.append(len(idToSentences[problemId][sentenceType])) #character length
		thisX.append(len(idToSentences[problemId][sentenceType].split(' '))) #word length

	X.append(thisX)
	y.append(correctGuesses[problemId])
	
	#write to file
	# toWrite = dict()
	# toWrite['Q1'] = idToSentences[id1]
	# toWrite['Q1']['s1_vec'] = idToVectors[id1]['s1']
	# toWrite['Q1']['s2_vec'] = idToVectors[id1]['s2']
	# toWrite['Q1']['h1_vec'] = idToVectors[id1]['h1']
	# toWrite['Q1']['h2_vec'] = idToVectors[id1]['h2']
	# toWrite['Q2'] = idToSentences[id2]
	# toWrite['Q2']['s1_vec'] = idToVectors[id2]['s1']
	# toWrite['Q2']['s2_vec'] = idToVectors[id2]['s2']
	# toWrite['Q2']['h1_vec'] = idToVectors[id2]['h1']
	# toWrite['Q2']['h2_vec'] = idToVectors[id2]['h2']
	# toWrite['choices'] = idPairsToVotes[(id1,id2)]
	# with open("pairTaskData_fewerQuestions.jsonl",'a') as F:
	# 	F.write(json.dumps(toWrite) + '\n')


#analyze the inter-annotator agreement
# plot = []
# for (id1,id2) in idPairsToVotes:
# 	if len(idPairsToVotes[(id1,id2)])<3:
# 		continue
# 	plot.append(sum(idPairsToVotes[(id1,id2)])*1.0/len(idPairsToVotes[(id1,id2)]))
# import matplotlib.pyplot as plt
# plt.hist(plot, bins=8)
# plt.show()


split = int(0.95*len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]
print("Train/test size:", len(X_train), len(X_test))

# print(np.array(y_test))
# print(X_test[0])

from scipy.stats import spearmanr
def getCorrelation(model):
	global X_test, y_test, X_train, y_train
	# results = model.fit(X_train, y_train).predict(X_test)
	results = model.fit(X_train, y_train).predict(X_test)
	return spearmanr(y_test, results)


#REGRESSORS
from sklearn.tree import DecisionTreeRegressor
for depth in [1, 2, 3,5,10,25]:
	regressor = DecisionTreeRegressor(max_depth = depth)
	print("Score w/depth", depth, ":", getCorrelation(regressor))

from sklearn import linear_model
linreg = linear_model.LinearRegression()
print("Linear regression:", getCorrelation(linreg))

from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# svr_lin = SVR(kernel='linear', C=1e3)
# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
print("y_rbf score:", getCorrelation(svr_rbf))
# print("y_lin score:", svr_lin.fit(X_train, y_train).score(X_test, y_test))
# print("y_poly score:", svr_poly.fit(X_train, y_train).score(X_test, y_test))



#CLASSIFIERS
# from sklearn.svm import SVC
# svc = SVC(gamma='auto').fit(X_train, y_train)
# print("SVC score:", svc.score(X_test, y_test))
# print(svc.predict(X_test))

# from sklearn.tree import DecisionTreeClassifier
# print("DT score:", DecisionTreeClassifier(random_state=0).fit(X_train, y_train).score(X_test, y_test))
# print(DecisionTreeClassifier(random_state=0).fit(X_train, y_train).predict(X_test))

import pickle
#train a new one on the entire data set
pickle.dump(SVR(kernel='rbf', C=1e3, gamma=0.1).fit(X,y), open("svr_model.pkl", 'wb'))
