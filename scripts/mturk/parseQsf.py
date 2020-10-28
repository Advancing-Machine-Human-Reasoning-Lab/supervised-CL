import json

with open("aNLI_train_set_shuffled.jsonl") as F:
	anliData = [json.loads(l) for l in F.readlines()]

#########FOR SINGLE QUESTION + ASSESS DIFFICULTY TASK#######
# with open("aNLI_difficulty_study_empty.json") as F:
# 	data = json.loads(F.read())
# whichRandomizerBlock = 0
# for i in range(1011):
# 	try:
# 		#For each problem:
# 		thisId = anliData[i]['index']
# 		s1 = anliData[i]['obs1']
# 		s2 = anliData[i]['obs2']
# 		hyp1 = anliData[i]['hyp1']
# 		hyp2 = anliData[i]['hyp2']

		# #add question 1
		# Q1 = json.loads("""	{"SurveyID":"SV_6XqB4UqNQIwgm0d","Element":"SQ","PrimaryAttribute":"QID19","SecondaryAttribute":"Consider the start\u00a0and end sentences below: start: fsdkljfdsjkl end: fdkjlsfdjkl Which of these t..","TertiaryAttribute":null,"Payload":
		# 		{"QuestionText":"Consider the <b>start<\/b>&nbsp;and <b>end <\/b>sentences below:<div><br><\/div><div><b>start<\/b>: %s<\/div><div><b>end<\/b>: %s<\/div><div><br><\/div><div>Which of these two sentences fits best <i>in between <\/i><b>start<\/b>&nbsp;and <b>end<\/b>?<\/div><div><br><\/div><div><b>hypothesis 1:<\/b>&nbsp;%s<\/div><div><b>hypothesis 2: <\/b>%s<\/div>","DataExportTag":"Q13","QuestionType":"MC","Selector":"SAVR","SubSelector":"TX","Configuration":
		# 			{"QuestionDescriptionOption":"UseText"},"QuestionDescription":"Consider the start\u00a0and end sentences below: start: fsdkljfdsjkl end: fdkjlsfdjkl Which of these t...","Choices":{"1":{"Display":"Hypothesis 1"},"2":{"Display":"Hypothesis 2"}},"ChoiceOrder":["1","2"],"Validation":{"Settings":{"ForceResponse":"ON","ForceResponseType":"ON","Type":"None"}},"Language":[],"NextChoiceId":4,"NextAnswerId":1,"QuestionID":"QID19"}}"""
		# 			% (s1, s2, hyp1, hyp2))
		# Q1['PrimaryAttribute'] = 'QIDq' + str(thisId)
		# Q1['Payload']['QuestionID'] = Q1['PrimaryAttribute']
		# Q1['Payload']['DataExportTag'] = Q1['PrimaryAttribute']
		# data['SurveyElements'].append(Q1)

# 		#add question 2
# 		Q2 = json.loads("""	{"SurveyID":"SV_6XqB4UqNQIwgm0d","Element":"SQ","PrimaryAttribute":"QID20","SecondaryAttribute":"How difficult was the previous question? Try to compare this to the questions you've answered so...","TertiaryAttribute":null,"Payload":{"QuestionText":"How difficult was the previous question? Try to compare this to the questions you've answered so far. If this is your first question, try to estimate based on the descriptions of the choices below.","DataExportTag":"Q14","QuestionType":"MC","Selector":"SAVR","SubSelector":"TX","Configuration":{"QuestionDescriptionOption":"UseText"},"QuestionDescription":"How difficult was the previous question? Try to compare this to the questions you've answered so...","Choices":{"1":{"Display":"Absolutely trivial: Anyone who understands the words should be able to solve it. Possibly the easiest problem in this question set."},"2":{"Display":"Somewhat trivial: It contains some nuances, but is easier than other problems I've done so far."},"3":{"Display":"Neither trivial nor difficult: It is the same as most problems I've done so far."},"4":{"Display":"Somewhat difficult: This took quite a bit of careful thinking, and is more difficult than other problems I've done so far."},"5":{"Display":"Absolutely difficult: I don't see how anyone could figure this out. Possibly the most difficult problem in this question set."}},"ChoiceOrder":["1","2","3","4","5"],"Validation":{"Settings":{"ForceResponse":"ON","ForceResponseType":"ON","Type":"None"}},"Language":[],"NextChoiceId":6,"NextAnswerId":1,"QuestionID":"QID20"}}""")
# 		Q2['PrimaryAttribute'] = 'QIDr' + str(thisId)
# 		Q2['Payload']['QuestionID'] = Q2['PrimaryAttribute']
# 		Q2['Payload']['DataExportTag'] = Q2['PrimaryAttribute']
# 		data['SurveyElements'].append(Q2)

# 		#add block that will hold question 1 and 2
# 		B = {'Type':"Standard", 'SubType':"", 'Description':'Block for Q'+str(thisId), 'ID':"BL_"+str(thisId),
# 				'BlockElements':[{'Type':'Question', 'QuestionID':Q1['PrimaryAttribute']}, {'Type':'Question', 'QuestionID':Q2['PrimaryAttribute']}]
# 			}
# 		indexToUse = -1
# 		for i in range(len(data['SurveyElements'][0]['Payload'])):
# 			if i not in data['SurveyElements'][0]['Payload']:
# 				indexToUse = i
# 				break
# 		if indexToUse==-1:
# 			raise Exception("Couldn't find a valid number!")
# 		data['SurveyElements'][0]['Payload'][indexToUse] = B

# 		#add block to survey flow
# 		if whichRandomizerBlock==0:
# 			i = 1
# 			whichRandomizerBlock=1
# 		else:
# 			i = 6
# 			whichRandomizerBlock=0
# 		# print("adding to block", i, whichRandomizerBlock)

# 		if 'Flow' not in data['SurveyElements'][1]['Payload']['Flow'][i]:
# 			data['SurveyElements'][1]['Payload']['Flow'][i]['Flow'] = []
# 		data['SurveyElements'][1]['Payload']['Flow'][i]['Flow'].append({'Type':"Standard", 'ID':B['ID'], 'FlowID':'FL_'+B['ID']})
# 	except:
# 		print("ERROR PARSING LINE ", i)


# with open("aNLI_difficulty_study.qsf", 'w') as F:
# 	F.write(json.dumps(data))


###### FOR DOUBLE QUESTION + COMPARISON TASK ########
with open("aNLI_Pair_Task.json") as F:
	data = json.loads(F.read())
whichRandomizerBlock = 0
for i in range(512):
	try:
		#add question 1
		thisId = anliData[i]['index']
		s1 = anliData[i]['obs1']
		s2 = anliData[i]['obs2']
		hyp1 = anliData[i]['hyp1']
		hyp2 = anliData[i]['hyp2']
		Q1 = json.loads("""	{"SurveyID":"SV_6XqB4UqNQIwgm0d","Element":"SQ","PrimaryAttribute":"QID19","SecondaryAttribute":"Consider the start\u00a0and end sentences below: start: fsdkljfdsjkl end: fdkjlsfdjkl Which of these t..","TertiaryAttribute":null,"Payload":
				{"QuestionText":"Consider the <b>start<\/b>&nbsp;and <b>end <\/b>sentences below:<div><br><\/div><div><b>start<\/b>: %s<\/div><div><b>end<\/b>: %s<\/div><div><br><\/div><div>Which of these two sentences fits best <i>in between <\/i><b>start<\/b>&nbsp;and <b>end<\/b>?<\/div><div><br><\/div><div><b>hypothesis 1:<\/b>&nbsp;%s<\/div><div><b>hypothesis 2: <\/b>%s<\/div>","DataExportTag":"Q13","QuestionType":"MC","Selector":"SAVR","SubSelector":"TX","Configuration":
					{"QuestionDescriptionOption":"UseText"},"QuestionDescription":"Consider the start\u00a0and end sentences below: start: fsdkljfdsjkl end: fdkjlsfdjkl Which of these t...","Choices":{"1":{"Display":"Hypothesis 1"},"2":{"Display":"Hypothesis 2"}},"ChoiceOrder":["1","2"],"Validation":{"Settings":{"ForceResponse":"ON","ForceResponseType":"ON","Type":"None"}},"Language":[],"NextChoiceId":4,"NextAnswerId":1,"QuestionID":"QID19"}}"""
					% (s1, s2, hyp1, hyp2))
		Q1['PrimaryAttribute'] = 'QIDq' + str(thisId)
		Q1['Payload']['QuestionID'] = Q1['PrimaryAttribute']
		Q1['Payload']['DataExportTag'] = Q1['PrimaryAttribute']
		thisId_Q1 = thisId

		#add question 2
		thisId = anliData[i+512]['index']
		s1 = anliData[i+512]['obs1']
		s2 = anliData[i+512]['obs2']
		hyp1 = anliData[i+512]['hyp1']
		hyp2 = anliData[i+512]['hyp2']
		Q2 = json.loads("""	{"SurveyID":"SV_6XqB4UqNQIwgm0d","Element":"SQ","PrimaryAttribute":"QID19","SecondaryAttribute":"Consider the start\u00a0and end sentences below: start: fsdkljfdsjkl end: fdkjlsfdjkl Which of these t..","TertiaryAttribute":null,"Payload":
				{"QuestionText":"Consider the <b>start<\/b>&nbsp;and <b>end <\/b>sentences below:<div><br><\/div><div><b>start<\/b>: %s<\/div><div><b>end<\/b>: %s<\/div><div><br><\/div><div>Which of these two sentences fits best <i>in between <\/i><b>start<\/b>&nbsp;and <b>end<\/b>?<\/div><div><br><\/div><div><b>hypothesis 1:<\/b>&nbsp;%s<\/div><div><b>hypothesis 2: <\/b>%s<\/div>","DataExportTag":"Q13","QuestionType":"MC","Selector":"SAVR","SubSelector":"TX","Configuration":
					{"QuestionDescriptionOption":"UseText"},"QuestionDescription":"Consider the start\u00a0and end sentences below: start: fsdkljfdsjkl end: fdkjlsfdjkl Which of these t...","Choices":{"1":{"Display":"Hypothesis 1"},"2":{"Display":"Hypothesis 2"}},"ChoiceOrder":["1","2"],"Validation":{"Settings":{"ForceResponse":"ON","ForceResponseType":"ON","Type":"None"}},"Language":[],"NextChoiceId":4,"NextAnswerId":1,"QuestionID":"QID19"}}"""
					% (s1, s2, hyp1, hyp2))
		Q2['PrimaryAttribute'] = 'QIDr' + str(thisId)
		Q2['Payload']['QuestionID'] = Q2['PrimaryAttribute']
		Q2['Payload']['DataExportTag'] = Q2['PrimaryAttribute']

		#add question 3
		Q3 = json.loads("""	{"SurveyID":"SV_6XqB4UqNQIwgm0d","Element":"SQ","PrimaryAttribute":"QID20","SecondaryAttribute":"Which was more difficult","TertiaryAttribute":null,"Payload":{"QuestionText":"Which of the previous two questions was more difficult?","DataExportTag":"Q14","QuestionType":"MC","Selector":"SAVR","SubSelector":"TX","Configuration":{"QuestionDescriptionOption":"UseText"},"QuestionDescription":"Which was more difficult","Choices":{"1":{"Display":"The first problem"},"2":{"Display":"The second problem"}},"ChoiceOrder":["1","2"],"Validation":{"Settings":{"ForceResponse":"ON","ForceResponseType":"ON","Type":"None"}},"Language":[],"NextChoiceId":6,"NextAnswerId":1,"QuestionID":"QID20"}}""")
		Q3['PrimaryAttribute'] = 'QIDs' + str(thisId_Q1) + 'x' + str(thisId)
		Q3['Payload']['QuestionID'] = Q3['PrimaryAttribute']
		Q3['Payload']['DataExportTag'] = Q3['PrimaryAttribute']

		#add block that will hold question 1 and 2
		B = {'Type':"Standard", 'SubType':"", 'Description':'Block for Q'+str(thisId), 'ID':"BL_"+str(thisId),
				'BlockElements':[{'Type':'Question', 'QuestionID':Q1['PrimaryAttribute']}, {'Type':'Question', 'QuestionID':Q2['PrimaryAttribute']}, {'Type':'Question', 'QuestionID':Q3['PrimaryAttribute']}]
			}
		indexToUse = -1
		for i in range(len(data['SurveyElements'][0]['Payload'])):
			if i not in data['SurveyElements'][0]['Payload']:
				indexToUse = i
				break
		if indexToUse==-1:
			raise Exception("Couldn't find a valid number!")
		# data['SurveyElements'][0]['Payload'][indexToUse] = B
		data['SurveyElements'][0]['Payload'].append(B)
		# print(type(data['SurveyElements'][0]['Payload']))

		data['SurveyElements'].append(Q1)
		data['SurveyElements'].append(Q2)
		data['SurveyElements'].append(Q3)

		if 'Flow' not in data['SurveyElements'][1]['Payload']['Flow'][1]:
			data['SurveyElements'][1]['Payload']['Flow'][1]['Flow'] = []
		data['SurveyElements'][1]['Payload']['Flow'][1]['Flow'].append({'Type':"Standard", 'ID':B['ID'], 'FlowID':'FL_'+B['ID']})
	except:
		print("ERROR PARSING LINE ", i)


with open("aNLI_difficulty_study.qsf", 'w') as F:
	F.write(json.dumps(data))