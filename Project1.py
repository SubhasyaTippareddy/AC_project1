import sys, os
import numpy as np
from math import acos
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

#To load data from all sub folders and files into an array
def load_data(path):
    X=[]
    Y=[]
    # 101 faces - sub folders, 6 emotions - Angry, Disgust, Fear, Happy, Sad, Surprise
    for face_dir in os.listdir(path):
        #faces
        face_path = path+face_dir+'/'
        if os.path.isdir(face_path):
            print(face_path)
            for label in os.listdir(face_path):
                #emotions
                emotion_path = face_path+label+'/'
                for file in os.listdir(emotion_path):
                    if file.endswith(".bnd") or file.endswith(".landmark"):
                        file_path = emotion_path+file
                        points = np.loadtxt(file_path, usecols=(1, 2, 3), encoding='utf-8') #x,y,z
                        X.append(points)
                        Y.append(label)
    print(X)
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(X.shape[0], -1)  
    print("data read")  
    return X,Y  

#Translate the face, by subtracting 
def translate_data(X):
    X_translated = []
    for face in X:
        mean = np.mean(face)
        translated_face = face - mean
        X_translated.append(translated_face)
    X_translated = np.array(X_translated)
    return X_translated

#Rotate data along some axis - x,y,z
def rotate_data(X, axis):
    pi=round(2*acos(0.0), 3)
    sin_angle = math.sin(pi) #approx 0
    cos_angle = math.cos(pi) #approx -1
    
    rotated_X=[]
    if axis == 'x':
        rotation_matrix = np.array([[1, 0, 0], [0, cos_angle, sin_angle], [0, -sin_angle, cos_angle ]])
    elif axis == 'y':
        rotation_matrix = np.array([[cos_angle, 0, -sin_angle], [0, 1, 0], [sin_angle, 0, cos_angle]])
    elif axis == 'z':
        rotation_matrix = np.array([[cos_angle, sin_angle, 0], [-sin_angle, cos_angle, 0], [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    for face in X:
        rotated_data = np.dot(face.reshape(-1, 3), rotation_matrix.T).reshape(face.shape)
        rotated_X.append(rotated_data)
    rotated_X = np.array(rotated_X) 
    return rotated_X

#Model fitting, Prediction
def classification(X,Y, classifier_type='RF'):
    if classifier_type == 'RF':
        print("Classifier - Random Forest")
        clf = RandomForestClassifier()
    elif classifier_type == 'SVM':
        print("Classifier - Support Vector Machine")
        clf = SVC()
    elif classifier_type == 'TREE':
        print("Classifier - Decision Tree")
        clf = DecisionTreeClassifier()

    Y_pred = []
    test_indices = []
    cv = StratifiedKFold(n_splits=10)
    for (train,test) in cv.split(X,Y):
        clf.fit(X[train],Y[train])
        Y_pred.append(clf.predict(X[test]))
        test_indices.append(test)
    
    return Y,Y_pred,test_indices

#To calculate metrics like precision, recall, accuracy
def model_evaluation(pred, indices, y):
    finalPredictions = []
    groundTruth = []
    for p in pred:
        finalPredictions.extend(p)
    for i in indices:
        groundTruth.extend(y[i])
    print(confusion_matrix(finalPredictions, groundTruth))
    print("Precision: ", precision_score(groundTruth, finalPredictions, average='macro'))
    print("Recall: ", recall_score(groundTruth, finalPredictions, average='macro'))
    print("Accuracy: " , accuracy_score(groundTruth, finalPredictions))

#taking command line args
args = sys.argv
#path = './BU4DFE_BND_V1.1/'
print(args)
path = args[3]
X,Y = load_data(path)
m = {'TREE': 'Decision Tree', 'RF': 'Random Forest', 'SVM':'Support Vector Machine'}
#To modify the data type
if(args[2]=='x'):
    X = rotate_data(X,'x')
elif(args[2]=='y'):
    X = rotate_data(X,'y')
elif(args[2]=='z'):
    X = rotate_data(X,'z')
elif(args[2]=='t'):
    X = translate_data(X)

#Classifier - to fit the model and record the predictions
y_original, y_pred, indices = classification(X,Y, classifier_type=args[1])
print("Following are the evaluation metrics for", m[args[1]],"classifier and", args[2], "data type")
#Performing Evaluation
model_evaluation(y_pred, indices, y_original)