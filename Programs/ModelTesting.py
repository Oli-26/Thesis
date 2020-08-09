import matplotlib.pyplot as plt
from LoadData import load_from_file, load_new, split_by_project
from sklearn.model_selection import train_test_split
import numpy as np

from LogisticRegression import train_logistic_regression
from DecisionTree import train_decision_tree
from SVM import train_svm
from KNN import train_knn
from NaiveBayes import train_naive_bayes
    


def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(df['commenttext'], df['category_id'], random_state = 10, train_size = 0.66)
    return (X_train, X_test, y_train, y_test)
    
def test_model(verbose, model, X_train, X_test, y_train, y_test):
    train_score = float(model.score(X_train, y_train))
    test_score = float(model.score(X_test, y_test))
    
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(y_test, y_pred))
    
    
    if verbose:
        general_predicted = model.predict(X_test)  
        print("Train accuracy = " + str(train_score))
        print("Test accuracy = " + str(test_score))
        #Show percentage of general prediction types 
        unique, counts = np.unique(general_predicted, return_counts=True)
        print(dict(zip(unique, counts*100/(len(general_predicted)))))

    return (train_score, test_score)
    

    
def compare_all():
    ## Init variables
    number_of_examples = 100000
    verbose = False
    
    df = load_from_file('technical_debt_dataset.csv', amount = number_of_examples)
    #df = load_new('file.csv', amount = number_of_examples, binary = True)
    
    #list = split_by_project(df)
    unique = np.unique(df['project'])
    Histories = []
    
    for i in range(0, len(unique)):
        print("Running for test project " + str(unique[i]))
        newDF = df[df['project'] != unique[i]]
        test = df[df['project'] == unique[i]]
        print("Train data = " + str(len(newDF)) + "  |  Test data = " + str(len(test)))

        #X_train, X_test, y_train, y_test = split_data(df)
        X_train = newDF['commenttext']
        X_test = test['commenttext']
        y_train = newDF['category_id']
        y_test = test['category_id']
        print("Running on " + str(len(X_train)) + " samples.")

        
        name_list = ["naive_bayes", "logistic_regression", "decision_tree", "svm", "knn"]
        listTest = []
        listTrain = []
        i = 10
        
        print("NaiveBayes STATS ----------")
        m = train_naive_bayes(X_train, y_train, i)
        s = test_model(True, m, X_train, X_test, y_train, y_test)
        listTest.append(s[1])
        listTrain.append(s[0])
        print("--------------------\n")
        
        print("LG STATS ----------")
        x = train_logistic_regression(X_train, y_train, min_df = i)
        s = test_model(True, m, X_train, X_test, y_train, y_test)
        listTest.append(s[1])
        listTrain.append(s[0])
        print("--------------------\n")
        
        print("DT STATS ----------")
        m = train_decision_tree(X_train, y_train, min_df = i)
        s = test_model(True, m, X_train, X_test, y_train, y_test)
        listTest.append(s[1])
        listTrain.append(s[0])
        print("--------------------\n")
        
        print("SVM STATS ----------")
        m = train_svm(X_train, y_train, min_df = i)
        s = test_model(True, m, X_train, X_test, y_train, y_test)
        listTest.append(s[1])
        listTrain.append(s[0])
        print("--------------------\n")
        
        print("KNN STATS ----------")
        m = train_knn(X_train, y_train, amount_neighbors = 3, min_df = i)
        s = test_model(True, m, X_train, X_test, y_train, y_test)
        listTest.append(s[1])
        listTrain.append(s[0])
        print("--------------------\n")   
        
        
        feature_array = np.array(m['tfidf'].get_feature_names())
        print("Discovered = "  + str(len(feature_array)) + " features.")
        print("Ran on " + str(df.shape[0]) + " examples.")
          
        fig = plt.figure(figsize=(8,6))
        plt.scatter(name_list, listTest)
        plt.scatter(name_list, listTrain)
        plt.show()
    
#test_logistic_regression()
#test_decision_tree()    
#test_svm()    
#test_knn()    
#test_naive_bayes()
compare_all()
