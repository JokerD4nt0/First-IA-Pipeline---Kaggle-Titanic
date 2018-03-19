# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:52:21 2018
7 mars 2018
@author: Adrien Morla
Adrien Morla
"""
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib as cm


#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
#from sklearn.metrics import accuracy_score

#plt.close('all')

#on charge le fichier csv.
train = pd.read_csv("train.csv")
kaggle = pd.read_csv("test.csv")


#passengerID = kaggle["PassengerId"].tolist()

#Suppression de la colonne cabine qui nous est d'aucune utilité.
train = train.drop("Cabin",axis = 1)
kaggle = kaggle.drop("Cabin",axis = 1)

#train["Age"] = train["Age"].astype("float")
#train["Age"] = train["Age"].fillna(train["Age"].median()) 

#on renomme les colonnes afin qu'elles soient plus parlantes.
train = train.rename(index=str, columns={"PassengerID": "ID", 
                                 "Pclass": "Class", 
                                 "SibSp": "F/S ou M/F",
                                 "Parch": "P/E",
                                 "Fare": "Price"})

kaggle = kaggle.rename(index=str, columns={"PassengerID": "ID", 
                                 "Pclass": "Class", 
                                 "SibSp": "F/S ou M/F",
                                 "Parch": "P/E",
                                 "Fare": "Price"})


correlation = train.corr()
correlation = correlation["Survived"].abs().sort_values(ascending = False)
print(correlation)

#Au vue des résultats de Nathalie et d'Adrien, nous supprimons PassengerId, qui nous est d'aucune utilité.
#cols = ['PassengerId']
#train = train.drop(cols,axis = 1)
#kaggle = kaggle.drop(cols,axis = 1)



#train = pd.DataFrame(train["Embarked"])
#one_hot = pd.get_dummies(train["from_S"])
#train = train.join(one_hot)
#print(train)

# On recode la colonne Embarked en encodant les nouvelles colonnes S, C et Q.
mask = train["Embarked"].astype("str") == "nan"
train["Embarked"].loc[mask] = "S"
recode = {'S': 1,'C': 2,'Q': 3}
train = train.replace({'Embarked' : recode})
kaggle = kaggle.replace({'Embarked' : recode})
train_test = pd.DataFrame(train["Embarked"])
kaggle_test = pd.DataFrame(kaggle["Embarked"])
train_test = pd.get_dummies(train["Embarked"])
kaggle_test = pd.get_dummies(kaggle["Embarked"])
train = pd.concat([train,train_test],axis = 1)
train=train.rename(columns = {1:'Southampton',2:'Cherbourg',3:'Queenstown'})
kaggle = pd.concat([kaggle,kaggle_test],axis = 1)
kaggle=kaggle.rename(columns = {1:'Southampton',2:'Cherbourg',3:'Queenstown'})

# on recode la colonne Sex en encondant les nouvelles colonnes H et F
recode = {"male":0,"female":1}
train = train.replace({"Sex":recode})
kaggle = kaggle.replace({"Sex":recode})
train_test = pd.DataFrame(train["Sex"])
kaggle_test = pd.DataFrame(kaggle["Sex"])
train_test = pd.get_dummies(train["Sex"])
kaggle_test = pd.get_dummies(kaggle["Sex"])
train = pd.concat([train,train_test],axis = 1)
train=train.rename(columns = {0:'H',1:'F'})
kaggle = pd.concat([kaggle,kaggle_test],axis = 1)
kaggle=kaggle.rename(columns = {0:'H',1:'F'})

# on recode la colonne Pclass en encodant les nouvelles colonnes Pclass1, 2 et 3.
train_test = pd.DataFrame(train["Class"])
kaggle_test = pd.DataFrame(kaggle["Class"])
train_test = pd.get_dummies(train["Class"])
kaggle_test = pd.get_dummies(kaggle["Class"])
train = pd.concat([train,train_test],axis = 1)
train=train.rename(columns = {1:'Class1',2:'Class2',3:'Class3'})
kaggle = pd.concat([kaggle,kaggle_test],axis = 1)
kaggle=kaggle.rename(columns = {1:'Class1',2:'Class2',3:'Class3'})

# on recode la colonne nom en encodant les nouvelles colonnes des titres.
train["Titre"] = train["Name"].str.extract("([a-zA-Z]+\.)",expand=False)
kaggle["Titre"] = kaggle["Name"].str.extract("([a-zA-Z]+\.)",expand=False)
train_test = pd.DataFrame(train["Titre"])
kaggle_test = pd.DataFrame(kaggle["Titre"])
train_test = pd.get_dummies(train["Titre"])
kaggle_test = pd.get_dummies(kaggle["Titre"])
train = pd.concat([train,train_test],axis = 1)
train=train.rename(columns = {1:'Mr.',2:'Mrs.',3:'Miss.',4:'Master.',5:'Don.',6:'Rev.', 7:'Dr.', 8:'Mme.', 9:'Ms.', 10: 'Major.'
, 11: 'Lady.', 12: 'Sir.', 13: 'Mlle.', 14: 'Col.', 15: 'Capt.',16: 'Countess.', 17: 'Jonkheer.'})
kaggle = pd.concat([kaggle,kaggle_test],axis = 1)
kaggle=kaggle.rename(columns = {1:'Mr.',2:'Mrs.',3:'Miss.',4:'Master.',5:'Don.',6:'Rev.', 7:'Dr.', 8:'Mme.', 9:'Ms.', 10: 'Major.'
, 11: 'Lady.', 12: 'Sir.', 13: 'Mlle.', 14: 'Col.', 15: 'Capt.',16: 'Countess.', 17: 'Jonkheer.'})

#on recode la colonne F/S M/F en encodant les nouvelles colonnes dans isChild, isParents
#ou pas.

# on ajoute la colonne "age_categ" sur le df train.
def process_age(train,cut_points,label_names):
    train["Age"] = train["Age"].fillna(-0.5)
    train["Age_categories"] = pd.cut(train["Age"],cut_points,labels=label_names)
    return train

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
train = process_age(train,cut_points,label_names)

# on ajoute la colonne "age_categ" sur le df kaggle.
def process_age(kaggle,cut_points,label_names):
    kaggle["Age"] = kaggle["Age"].fillna(-0.5)
    kaggle["Age_categories"] = pd.cut(kaggle["Age"],cut_points,labels=label_names)
    return kaggle

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
kaggle = process_age(kaggle,cut_points,label_names)

# Suppression des valeurs inutiles
train = (train.drop(['Class','Embarked','Sex',"Ticket","Name","Titre"], axis = 1))
kaggle = (kaggle.drop(['Class','Embarked','Sex',"Ticket","Name","Titre"], axis = 1))

kaggle.to_csv("naan.csv")

#print(train['Age'].unique())
#
#print(train['H'].unique())
#
#print(train['F'].unique())
#
#print(train['Class1'].unique())
#
#print(train['Class2'].unique())
#print(train['Class3'].unique())
#
#print(train['Price'].unique())
#
#print(train['F/S ou M/F'].unique())
#
#print(train['P/E'].types())


#nbenfant = train["P/E"].unique()
#print(nbenfant)

#nbenfant = kaggle["P/E"].unique()
#print(nbenfant)


#kaggle["Age"] = kaggle["Age"].fillna(kaggle["Age"].median())
#kaggle["Fare"] = kaggle["Fare"].fillna(kaggle["Fare"].median())

#train.info()
#kaggle.info()

#X_cols = ['Pclass',"Sex","Age","Parch","Fare","Embarked"]

#X = train[X_cols]
#y = train["Survived"]

#X_kaggle = kaggle[X_cols]

#cmap = plt.get_cmap('gnuplot')
#scatter = pd.scatter_matrix(X, c = y, marker = 'o', cmap = cmap)
#scatter
#plt.show()

#X_train, X_test, y_train, y_test = train_test_split(X,y)

#scaler = MinMaxScaler()
#scaler.fit(X_train)

#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
#X_kaggle = scaler.transform(X_kaggle)

#c = [c for c in range(1,10)]

#for c in c:
    #LR = LogisticRegression(C=c)
    #LR.fit(X_train, y_train)
    #train_score = LR.score(X_train, y_train)
    #train_score = LR.score(X_train, y_train)
   # test_score = LR.score(X_test, y_test)
    #print("Pour c =%s. Training score = %s. Test score = %s" % (c, train_score, test_score))
    
#LR = LogisticRegression(C=1)
#LR.fit(X_train, y_train)
#LR.fit(X_train, y_train)

#Survived = LR.predict(X_kaggle)

#data = pd.DataFrame({"PassengerId":passengerID,"Survived": Survived})
#data = pd.DataFrame({"PassengerId":passengerID,"Survived": Survived})
#data = data.set_index("PassengerId")
#data.to_csv("data.csv")



#train = sbn.load_dataset('titanic_train')
#train = sbn.load_dataset('titanic_test')

# Matrice des corrélations entre les variables.*

#plt.figure()
#sns.heatmap(df_titanic.corr(),annot=True)

#Boîtes à moustaches : qui à survécu/et mort selon l'âge?

#plt.figure()
#sns.boxplot(x='survived',y='age',data=df_titanic)

#Courbe représentative du nombre de passagers du même age.

#plt.figure()
#df_titanic['age'].plot(kind='kde')

#On vérifie que notre nouveau dataframe est comme on le souhaite.

#print(df_titanic2.describe())
#print(df_titanic2.head())
#print(df_titanic2.info())
#print(df_titanic2.describe())
#print(df_titanic2.dtypes)

#On supprime les valeurs manquantes.
#train = df_titanic.dropna()

#On vérifie qu'il n'y a plus de valeurs NaN.
 
#print(df_titanic2.info())

#On isole la valeur à prédire dans le 3e dataframe et on regroupe les autres valeurs dans un 4e dataframe.
#df_titanic3 = df_titanic2['survived']
#df_titanic4 = df_titanic2.drop(['survived'],axis=1)

#On sépare le jeu de donnée en deux : un train et un test.
#X = df_titanic.drop(['survived'],axis=1)
#y = df_titanic['survived']

#X_test, X_train, y_test, y_train = train_test_split(X, y,test_size=0.2)

#Fonction de prédiction du jeux d'entrainement
#y_pred = GaussianNB().fit(X_train, y_train).predict(X_train) 

#print(y_train.sum())
#print(y_pred.sum())

#print("Number of passengers out of a total %d that survived : %d" %
     # (X_train.shape[0], (y_train != y_pred).sum()))

#print("Accuracy score : %.3f" % accuracy_score(y_train, y_pred))


#Fonction de prédiction du jeux de test

#y_pred2 = GaussianNB().fit(X_test, y_test).predict(X_test)

#print(y_test.sum())
#print(y_pred2.sum())

#print("Number of passengers out of a total %d that survived : %d" %
     # (X_test.shape[0], (y_test != y_pred2).sum()))

#print("Accuracy score : %.3f" % accuracy_score(y_test, y_pred2))