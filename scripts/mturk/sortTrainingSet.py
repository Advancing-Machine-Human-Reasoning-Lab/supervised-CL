import json
from scipy.spatial.distance import cdist
import numpy as np
import random


with open("aNLI_train_set_shuffled.jsonl") as F:
	anliData = [json.loads(l) for l in F.readlines()]

with open("train-labels.lst") as F:
	vals = [int(i) for i in F.readlines()]

for i in range(len(anliData)):
	anliData[i]['correct'] = vals[i]

#load classifier
import pickle
from sklearn.svm import SVR

svr = pickle.load(open("svr_model.pkl", 'rb'))


#ONE TIME ONLY: load vectors from file, one line at a time, calculate distances, then save to file.
#this is so we don't have to load the entire anli_vectors_big into memory.
# with open("/Users/licato/Downloads/anli_vectors_big.jsonl") as F:
# 	Index = 0
# 	for k in F.readlines():
# 		Index+=1
# 		if Index%2500==0:
# 			print(Index,'of 161k?')
# 		obj = json.loads(k) #['id', 's1', 's2', 'h1', 'h2']
# 		toWrite = str(obj['id'])
# 		for (v1,v2) in [('s1','s2'), ('s1','h1'), ('s1','h2'), ('s2','h1'), ('s2','h2'), ('h1','h2')]:
# 			toWrite += '\t' + str(cdist([obj[v1]], [obj[v2]],"cosine")[0][0])
# 		with open("/Users/licato/Downloads/anli_vectors_big_distances_only.tsv", 'a') as G:
# 			G.write(toWrite + '\n')
# exit()


#load vectors from file
print("Loading vector distances...")
idToVectorDists = dict()
with open("/Users/licato/Downloads/anli_vectors_big_distances_only.tsv") as F:
	for l in F.readlines():
		vals = [v for v in l.strip().split('\t')]
		Id = vals.pop(0)
		# print(obj['id'])
		# exit()
		idToVectorDists[int(Id)] = [float(f) for f in vals]

# print(list(idToVectors.keys())[:100])
# print(len(idToVectors), len(anliData))
# exit()

#given index from anliData, turns it into a vector
def createVector(i):
	# thisId = anliData[i]['index']
	thisX = [v for v in idToVectorDists[anliData[i]['index']]]
	s1 = anliData[i]['obs1']
	s2 = anliData[i]['obs2']
	h1 = anliData[i]['hyp1']
	h2 = anliData[i]['hyp2']
	for sentenceType in ['obs1', 'obs2', 'hyp1', 'hyp2']:
		thisX.append(len(anliData[i][sentenceType])) #character length
		thisX.append(len(anliData[i][sentenceType].split(' '))) #word length
	return thisX

print("Estimating difficulties...")
toRemove = []
for i in range(len(anliData)):
	# print("Currently on index", i, "anliData[i]=", anliData[i]['index'])
	# print(anliData[i]['index'] in idToVectors)
	# input()
	if i%10000==0:
		print('\t',i,'of',len(anliData))
	try:
		vec = createVector(i)
	except:
		print("Error on index", i, "anliData[i]=", anliData[i]['index'])
		print(anliData[i]['index'] in idToVectors)
		exit()
		toRemove.insert(0,i)
	anliData[i]['predictedEase'] = svr.predict([vec])[0] #note that lower values means more difficulty
for x in toRemove:
	anliData.pop(x)

#sort anliData based on predictedDifficulty
print("Sorting...") 
from operator import attrgetter
anliData = sorted(anliData, key=lambda x: x['predictedEase'], reverse=True) #sort so that easier problems are first

print("Writing to file...")
with open("aNLI_train_set_sorted.jsonl",'w') as F: 
	F.write('\n'.join([json.dumps(o) for o in anliData]))