from flask import Flask, render_template, request
import numpy as np
import os
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hello', methods=['POST'])
def hello():
	import numpy as np
	import os


	X_list1=[]
	Y_list1=[]
	X_list2=[]
	Y_list2=[]
	X_list3=[]
	Y_list3=[]
	X_list4=[]
	Y_list4=[]

	
	i=0
	team1_name = request.form['team1_name']
	team2_name = request.form['team2_name']
	t_score1 = request.form['t_score1']
	t_score2 = request.form['t_score2']
	t_shots1 = request.form['t_shots1']
	t_shots2 = request.form['t_shots2']
	t_shotsontarget1 = request.form['t_shotsontarget1']
	t_shotsontarget2 = request.form['t_shotsontarget2']
	t_fouls1 = request.form['t_fouls1']
	t_fouls2 = request.form['t_fouls2']
	t_corners1 = request.form['t_corners1']
	t_corners2 = request.form['t_corners2']
	t_yellow1 = request.form['t_yellow1']
	t_yellow2 = request.form['t_yellow2']
	t_red1 = request.form['t_red1']
	t_red2 = request.form['t_red2']


	#return '%s %s' %(team1_name,team2_name)
	team1=" sortedData"+team1_name+".txt"
	team2=" sortedData"+team2_name+".txt"
	#return '%s %s' %(team1,team2)
	start_path="../txt"
	teams_file_names = os.listdir(start_path)
	#return '%s %s' %(team1_name,team2_name)

	for i in range(0,len(teams_file_names),1):
		#return '%s %s' %(team1_name,team2_name)
		team1_file=open(start_path+"/"+teams_file_names[i],"r")
		team1_datas=team1_file.readlines()
		getBatch(team1_datas, 0, len(team1_datas), X_list1, Y_list1,1)
		getBatch(team1_datas, 0, len(team1_datas), X_list2, Y_list2,2)
		getBatch(team1_datas, 0, len(team1_datas), X_list3, Y_list3,3)
		getBatch(team1_datas, 0, len(team1_datas), X_list4, Y_list4,4)


	X_train_set1 = []
	Y_train_set1 = []
	X_train_set2 = []
	Y_train_set2 = []
	X_train_set3 = []
	Y_train_set3 = []
	X_train_set4 = []
	Y_train_set4 = []

	selectNonRepeatingData(X_list1, Y_list1, X_train_set1, Y_train_set1)
	selectNonRepeatingData(X_list2, Y_list2, X_train_set2, Y_train_set2)
	selectNonRepeatingData(X_list3, Y_list3, X_train_set3, Y_train_set3)
	selectNonRepeatingData(X_list4, Y_list4, X_train_set4, Y_train_set4)
	print X_list1
	print Y_list1

	print X_list2
	print Y_list2

	print X_list3
	print Y_list3

	print X_list4
	print Y_list4
	#return '%s %s' %(team1_name,team2_name)
	X_test_np1=[]
	X_test_np1.append(int(t_score1))
	X_test_np1.append(int(t_score2))
	X_test_np1.append(int(t_shots1))
	X_test_np1.append(int(t_shots2))
	X_test_np1.append(int(t_shotsontarget1))
	X_test_np1.append(int(t_shotsontarget2))
	X_test_np1.append(int(t_fouls1))
	X_test_np1.append(int(t_fouls2))
	X_test_np1.append(int(t_corners1))
	X_test_np1.append(int(t_corners2))
	X_test_np1.append(int(t_yellow1))
	X_test_np1.append(int(t_yellow2))
	X_test_np1.append(int(t_red1))
	X_test_np1.append(int(t_red2))
	X_train_np1 = np.array(X_train_set1)
	Y_train_np1 = np.array(Y_train_set1)

	X_test_np2=[]
	X_test_np2.append(int(t_score1))
	X_test_np2.append(int(t_score2))
	X_test_np2.append(int(t_shots1))
	X_test_np2.append(int(t_shots2))
	X_test_np2.append(int(t_shotsontarget1))
	X_test_np2.append(int(t_shotsontarget2))
	X_test_np2.append(int(t_fouls1))
	X_test_np2.append(int(t_fouls2))
	X_test_np2.append(int(t_corners1))
	X_test_np2.append(int(t_corners2))
	X_test_np2.append(int(t_yellow1))
	X_test_np2.append(int(t_yellow2))
	X_test_np2.append(int(t_red1))
	X_test_np2.append(int(t_red2))
	X_train_np2 = np.array(X_train_set2)
	Y_train_np2 = np.array(Y_train_set2)

	X_test_np3=[]
	X_test_np3.append(int(t_shotsontarget1))
	X_test_np3.append(int(t_shotsontarget2))
	X_test_np3.append(int(t_corners1))
	X_test_np3.append(int(t_corners2))
	X_test_np3.append(int(t_red1))
	X_test_np3.append(int(t_red2))
	X_train_np3 = np.array(X_train_set3)
	Y_train_np3 = np.array(Y_train_set3)

	X_test_np4=[]
	X_test_np4.append(int(t_score1))
	X_test_np4.append(int(t_score2))
	X_test_np4.append(int(t_shots1))
	X_test_np4.append(int(t_shots2))
	X_test_np4.append(int(t_shotsontarget1))
	X_test_np4.append(int(t_shotsontarget2))
	X_test_np4.append(int(t_fouls1))
	X_test_np4.append(int(t_fouls2))
	X_test_np4.append(int(t_corners1))
	X_test_np4.append(int(t_corners2))
	X_test_np4.append(int(t_yellow1))
	X_test_np4.append(int(t_yellow2))
	X_test_np4.append(int(t_red1))
	X_test_np4.append(int(t_red2))
	X_train_np4 = np.array(X_train_set4)
	Y_train_np4 = np.array(Y_train_set4)




	treesCount = 10
	result=[]
	result.append(randomForest(X_train_np1, Y_train_np1, X_test_np1, treesCount))
	resultrf=randomForest(X_train_np1, Y_train_np1, X_test_np1, treesCount)
	result.append(naiveBayes(X_train_np2, Y_train_np2, X_test_np2))
	resultnb=naiveBayes(X_train_np2, Y_train_np2, X_test_np2)
	result.append(svmachine(X_train_np3, Y_train_np3, X_test_np3))
	resultsvm=svmachine(X_train_np3, Y_train_np3, X_test_np3)
	result.append(decisiontree(X_train_np4,Y_train_np4,X_test_np4, 4))
	resultdt=decisiontree(X_train_np4,Y_train_np4,X_test_np4, 4)
	#return '%s %s' %(team1_name,team2_name)

	if result.count(0)>=result.count(1) and result.count(0)>=result.count(2):
		result1=team2_name+" will win"
	elif result.count(1)>=result.count(0) and result.count(1)>=result.count(2):
		result1="match will draw"
	else:
		result1=team1_name+" will win"

	if resultrf==0:
		ans1=team2_name+" will win"
	elif resultrf==1:
		ans1="match will draw"
	else:
		ans1=team1_name+" will win"

	print "Random Forest: ",ans1

	if resultnb==0:
		ans2=team2_name+" will win"
	elif resultnb==1:
		ans2="match will draw"
	else:
		ans2=team1_name+" will win"

	print "Naive Bayes:  ",ans2

	if resultsvm==0:
		ans3=team2_name+" will win"
	elif resultsvm==1:
		ans3="match will draw"
	else:
		ans3=team1_name+" will win"

	print "Support Vector Machines: ",ans3

	if resultdt==0:
		ans4=team2_name+" will win"
	elif resultdt==1:
		ans4="match will draw"
	else:
		ans4=team1_name+" will win"

	print "Decission Tree: ",ans4

	return render_template("result.html",result=result1)

def getBatch(listParams, start, finish, X_batch, Y_batch, q):
	if q==1:
		for i in range (start, finish, 1):
			game_x = []
			game_y = -1
			params = listParams[i].split(";")
			score_1 = int(params[4])
			score_2 = int(params[5])
			goals_1 = int(params[2])
			goals_2 = int(params[3])
			home=params[6]
			shots_1 = int(params[7])
			shots_2 = int(params[8])
			shotsontarget_1 = int(params[9])
			shotsontarget_2 = int(params[10])
			foul_1 = int(params[11])
			foul_2 = int(params[12])
			corner_1 = int(params[13])
			corner_2 = int(params[14])
			yellow_1 = int(params[15])
			yellow_2 = int(params[16])
			red_1 = int(float(params[17]))#
			#red_2 = int(params[18])
			params[18]=params[18][:params[18].find("\n")]
			red_2=int(float(params[18]))	
			
			game_x.append(score_1)
			game_x.append(score_2)
		
			game_x.append(shots_1)
			game_x.append(shots_2)
		
			game_x.append(shotsontarget_1)
			game_x.append(shotsontarget_2)
		
			game_x.append(foul_1)
			game_x.append(foul_2)
		
			game_x.append(corner_1)
			game_x.append(corner_2)
		
			game_x.append(yellow_1)
			game_x.append(yellow_2)
		
			game_x.append(red_1)
			game_x.append(red_2)
		

		
			if goals_1 > goals_2:
				game_y = 2
			elif goals_1 < goals_2:
				game_y = 0
			else:
				game_y = 1

			X_batch.append(game_x)
			Y_batch.append(game_y)

	if q==2:
		for i in range (start, finish, 1):
			game_x = []
			game_y = -1
			params = listParams[i].split(";")
			score_1 = int(params[4])
			score_2 = int(params[5])
			goals_1 = int(params[2])
			goals_2 = int(params[3])
			home=params[6]
			shots_1 = int(params[7])
			shots_2 = int(params[8])
			shotsontarget_1 = int(params[9])
			shotsontarget_2 = int(params[10])
			foul_1 = int(params[11])
			foul_2 = int(params[12])
			corner_1 = int(params[13])
			corner_2 = int(params[14])
			yellow_1 = int(params[15])
			yellow_2 = int(params[16])
			red_1 = int(float(params[17]))#
			#red_2 = int(params[18])
			params[18]=params[18][:params[18].find("\n")]
			red_2=int(float(params[18]))	
			
			game_x.append(score_1)
			game_x.append(score_2)
			
			#game_x.append(goals_1)
			#game_x.append(goals_2)
			
			game_x.append(shots_1)
			game_x.append(shots_2)

			#if home:
			#	game_x.append(1)
			#else:
			#	game_x.append(2)
			
			game_x.append(shotsontarget_1)
			game_x.append(shotsontarget_2)
			

			game_x.append(foul_1)
			game_x.append(foul_2)
			
			game_x.append(corner_1)
			game_x.append(corner_2)

			game_x.append(yellow_1)
			game_x.append(yellow_2)
			
			game_x.append(red_1)
			game_x.append(red_2)
			
			



			selfField = False
			if params[6] == "True\n":
				selfField = True
			else:
				selfField = False

			
			
			if goals_1 > goals_2:
				game_y = 2
			elif goals_1 < goals_2:
				game_y = 0
			else:
				game_y = 1

			X_batch.append(game_x)
			Y_batch.append(game_y)

	if q == 3:
		for i in range (start, finish, 1):
			game_x = []
			game_y = -1
			params = listParams[i].split(";")
			score_1 = int(params[4])
			score_2 = int(params[5])
			goals_1 = int(params[2])
			goals_2 = int(params[3])
			home=params[6]
			shots_1 = int(params[7])
			shots_2 = int(params[8])
			shotsontarget_1 = int(params[9])
			shotsontarget_2 = int(params[10])
			foul_1 = int(params[11])
			foul_2 = int(params[12])
			corner_1 = int(params[13])
			corner_2 = int(params[14])
			yellow_1 = int(params[15])
			yellow_2 = int(params[16])
			red_1 = int(float(params[17]))#
			#red_2 = int(params[18])
			params[18]=params[18][:params[18].find("\n")]
			red_2=int(float(params[18]))	
			#game_x.append(goals_1)
			#game_x.append(goals_2)
			
			game_x.append(shotsontarget_1)
			game_x.append(shotsontarget_2)
			game_x.append(corner_1)
			game_x.append(corner_2)
			game_x.append(red_1)
			game_x.append(red_2)
			
			if goals_1 > goals_2:
				game_y = 2
			elif goals_1 < goals_2:
				game_y = 0
			else:
				game_y = 1

			X_batch.append(game_x)
			Y_batch.append(game_y)

	if q==4:
		for i in range (start, finish, 1):
			game_x = []
			game_y = -1
			params = listParams[i].split(";")
			score_1 = int(params[4])
			score_2 = int(params[5])
			goals_1 = int(params[2])
			goals_2 = int(params[3])
			home=params[6]
			shots_1 = int(params[7])
			shots_2 = int(params[8])
			shotsontarget_1 = int(params[9])
			shotsontarget_2 = int(params[10])
			foul_1 = int(params[11])
			foul_2 = int(params[12])
			corner_1 = int(params[13])
			corner_2 = int(params[14])
			yellow_1 = int(params[15])
			yellow_2 = int(params[16])
			red_1 = int(float(params[17]))#
			#red_2 = int(params[18])
			params[18]=params[18][:params[18].find("\n")]
			red_2=int(float(params[18]))	
			
			game_x.append(score_1)
			game_x.append(score_2)
			
			#game_x.append(goals_1)
			#game_x.append(goals_2)
			

			game_x.append(shots_1)
			game_x.append(shots_2)
			
			game_x.append(shotsontarget_1)
			game_x.append(shotsontarget_2)
			
			game_x.append(foul_1)
			game_x.append(foul_2)
			
			game_x.append(corner_1)
			game_x.append(corner_2)
			
			game_x.append(yellow_1)
			game_x.append(yellow_2)
			
			game_x.append(red_1)
			game_x.append(red_2)
			
			selfField = False
			if params[6] == "True\n":
				selfField = True
			else:
				selfField = False

			
			
			if goals_1 > goals_2:
				game_y = 2
			elif goals_1 < goals_2:
				game_y = 0
			else:
				game_y = 1

			X_batch.append(game_x)
			Y_batch.append(game_y)

def selectNonRepeatingData(X_list, Y_list, X_set, Y_set):
	y_delete_indexes = [] 
	for i in range(0, len(X_list), 1):
		elem = X_list[i];
		if elem not in X_set:
			X_set.append(elem)
		else:
			y_delete_indexes.append(i)
	
	for i in range(0, len(Y_list), 1):
		if i not in y_delete_indexes:
			Y_set.append(Y_list[i])

def randomForest(train_X, train_Y, test_X, treeCount):
	X = train_X[:]
	Y = train_Y[:]
	X_test = test_X[:]
	forest = RandomForestClassifier(treeCount)
	forest.fit(X, Y)
	predictedRF = forest.predict(X_test)
	return predictedRF

def naiveBayes(train_X, train_Y, test_X):
	X = train_X[:]
	Y = train_Y[:]
	X_test = test_X[:]
	model_NB = GaussianNB()
	model_NB.fit(X, Y)
	predictedNB = model_NB.predict(X_test)
	return predictedNB

def svmachine(train_X, train_Y, test_X):
	X = train_X[:]
	Y = train_Y[:]
	X_test = test_X[:]
	clf = svm.SVC()
	clf.fit(X, Y)
	predictedSVM = clf.predict(X_test)
	return predictedSVM

def decisiontree(train_X,train_Y,test_X,treedepth):
	X=train_X[:]
	Y=train_Y[:]
	X_test=test_X[:]
	decisiontree=clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=treedepth, min_samples_leaf=5)	
	decisiontree.fit(X,Y)
	predictedtree=decisiontree.predict(X_test)
	return predictedtree




if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 3000)
   