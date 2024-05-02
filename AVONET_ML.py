import numpy as np
import csv
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

def min_filter_csv(input_file, output_file, min_count):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            header = next(reader)
            writer.writerow(header)

            id_data = {}

            curr_id = next(reader)[0]
            
            for row in reader:
                id_value = row[0]
                
                if id_value not in id_data:
                    id_data[id_value] = {
                        'count': 1,
                        'rows': [row]
                    }
                else:
                    id_data[id_value]['count'] += 1
                    id_data[id_value]['rows'].append(row)
                
                if id_data[curr_id]['count'] >= min_count and id_value != curr_id:
                    for r in id_data[curr_id]['rows']:
                        writer.writerow(r)
                    curr_id = id_value

                elif id_value != curr_id:
                    curr_id = id_value


def protocol_1_filter_CSV(input_file, output_file):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            header = next(reader)
            writer.writerow(header)

            for row in reader:
                if row[24] == '1':
                    writer.writerow(row)


def clean_csv(input_file='AVONET_Raw_Data.csv', min_rows=50):

    intermediate_file = 'AVONET_Protocol_1_Data.csv'

    output_file = 'AVONET_' + str(min_rows) + '_Points_Data.csv'

    protocol_1_filter_CSV(input_file, intermediate_file)

    min_filter_csv(intermediate_file, output_file, min_rows)

    return output_file



def load_data(input_file):
    with open(input_file) as infile:
        header = infile.readline()
        labels = []
        points = []
        for row in infile.readlines():
            data = row.split(',')
            labels.append(data[1])
            points.append([x for x in data[13:22]])
        
        # getting the sum and counts of each data point that isn't 'NA'
        sum_dictionary = {}
        for label, features in zip(labels, points):
            if label in sum_dictionary:
                for feat in range(9):
                    if features[feat] != 'NA' and features[feat] != '':
                        sum_dictionary[label][feat*2] = float(features[feat]) + sum_dictionary[label][feat*2]
                        sum_dictionary[label][feat*2+1] += 1
            else:
                sum_dictionary[label] = [0] * 18
                for feat in range(9):
                    if features[feat] != 'NA' and features[feat] != '':
                        sum_dictionary[label][feat*2] = float(features[feat]) + sum_dictionary[label][feat*2]
                        sum_dictionary[label][feat*2+1] = 1
                    else:
                        sum_dictionary[label][feat*2] = 0
                        sum_dictionary[label][feat*2+1] = 0
        
        # calculating mean of each feature for each label
        mean_dictionary = {}
        for label in sum_dictionary.keys():
            mean_dictionary[label] = [0] * 9
            for feat in range(9):
                if sum_dictionary[label][feat*2+1] == 0:
                    mean_dictionary[label][feat] = 0
                else:
                    mean_dictionary[label][feat] = sum_dictionary[label][feat*2] / sum_dictionary[label][feat*2+1]

        # replacing 'NA' with corresponding mean (imputing)
        for i in range(len(labels)):
            for j in range(9):
                if points[i][j] == 'NA' or points[i][j] == '':
                    points[i][j] = round(mean_dictionary[labels[i]][j], 1)
                else:
                    points[i][j] = float(points[i][j])

        #print(mean_dictionary)

        points = np.array(points)
        labels = np.array(labels)
        return points, labels
    

def generate_test_files():
    clean_files = [''] * 10
    clean_files[0] = clean_csv('AVONET_Raw_Data.csv', 100)
    clean_files[1] = clean_csv('AVONET_Raw_Data.csv', 80)
    clean_files[2] = clean_csv('AVONET_Raw_Data.csv', 60)
    clean_files[3] = clean_csv('AVONET_Raw_Data.csv', 50)
    clean_files[4] = clean_csv('AVONET_Raw_Data.csv', 40)
    clean_files[5] = clean_csv('AVONET_Raw_Data.csv', 30)
    clean_files[6] = clean_csv('AVONET_Raw_Data.csv', 20)
    clean_files[7] = clean_csv('AVONET_Raw_Data.csv', 15)
    clean_files[8] = clean_csv('AVONET_Raw_Data.csv', 10)
    clean_files[9] = clean_csv('AVONET_Raw_Data.csv', 5)

    return clean_files


def svm(X_train, X_test, y_train, y_test, mykernel='linear'):

    svm_model = SVC(kernel=mykernel, random_state=42)

    svm_model.fit(X_train, y_train)
    
    y_pred = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    #print(report)

    return report , accuracy


def randomforest(X_train, X_test, y_train, y_test, crit='gini'):

    rf_model = RandomForestClassifier(criterion=crit, random_state=42)

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    #print(report)

    return report , accuracy


def randomforest_gridsearch(X_train, X_test, y_train, y_test, grid, crit='gini'):

    rf_model = RandomForestClassifier(criterion=crit, random_state=42)

    grid_test = GridSearchCV(estimator=rf_model, param_grid=grid, scoring='accuracy', verbose=3)

    grid_test.fit(X_train, y_train)

    return grid_test


def naivebayes(X_train, X_test, y_train, y_test):

    nb_model = GaussianNB()

    nb_model.fit(X_train, y_train)

    y_pred = nb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    #print(report)

    return report , accuracy


def multilayerperceptron(X_train, X_test, y_train, y_test):

    mlp_model = MLPClassifier(max_iter=500, random_state=42)

    mlp_model.fit(X_train, y_train)

    y_pred = mlp_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    #print(report)

    return report , accuracy



def test_models(clean_files):

    print('Testing basic models...\n')

    X_train = [[]] * 10
    X_test = [[]] * 10
    y_train = [[]] * 10
    y_test = [[]] * 10

    for i in range(10):
        my_features , my_labels = load_data(clean_files[i])

        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(my_features, my_labels, test_size=0.2, random_state=42)

    outfile = open('AVONET_ML_Report.txt', 'w')

    accuracy = [[0] * 10 for i in range(6)]

    print('Testing Linear SVM...\n')

    # linear svm
    outfile.write('Results for Linear SVM - \n')
    for i in range(10):
        report , accuracy[0][i] = svm(X_train[i], X_test[i], y_train[i], y_test[i])
        outfile.write(clean_files[i] + ' -' + '\n' + '\n')
        outfile.write(report + '\n' + '\n')

    print('Testing RBF SVM...\n')

    # linear svm
    outfile.write('Results for RBF SVM - \n')
    for i in range(10):
        report , accuracy[1][i] = svm(X_train[i], X_test[i], y_train[i], y_test[i], 'rbf')
        outfile.write(clean_files[i] + ' -' + '\n' + '\n')
        outfile.write(report + '\n' + '\n')

    print('Testing Sigmoid SVM...\n')

    # linear svm
    outfile.write('Results for Sigmoid SVM - \n')
    for i in range(10):
        report , accuracy[2][i] = svm(X_train[i], X_test[i], y_train[i], y_test[i], 'sigmoid')
        outfile.write(clean_files[i] + ' -' + '\n' + '\n')
        outfile.write(report + '\n' + '\n')

    print('Testing Random Forest...\n')

    # random forest 
    outfile.write('Results for Random Forest - \n')
    for i in range(10):
        report , accuracy[3][i] = randomforest(X_train[i], X_test[i], y_train[i], y_test[i])
        outfile.write(clean_files[i] + ' -' + '\n' + '\n')
        outfile.write(report + '\n' + '\n')
    
    print('Testing Gaussian Naive Bayes...\n')

    # naive bayes
    outfile.write('Results for Gaussian Naive Bayes - \n')
    for i in range(10):
        report , accuracy[4][i] = naivebayes(X_train[i], X_test[i], y_train[i], y_test[i])
        outfile.write(clean_files[i] + ' -' + '\n' + '\n')
        outfile.write(report + '\n' + '\n')

    print('Testing Multi Layer Perceptron...\n')

    # multi layer perceptron
    outfile.write('Results for Multi Layer Perceptron - \n')
    for i in range(10):
        report , accuracy[5][i] = multilayerperceptron(X_train[i], X_test[i], y_train[i], y_test[i])
        outfile.write(clean_files[i] + ' -' + '\n' + '\n')
        outfile.write(report + '\n' + '\n')

    # printing accuracy to report file
    outfile.write(' Model | 100  |  80  |  60  |  50  |  40  |  30  |  20  |  15  |  10  |  5\n')
    outfile.write('LinSVM | ' + str(round(accuracy[0][0], 2)) + '    ' + str(round(accuracy[0][1], 2)) + '   ' 
                  + str(round(accuracy[0][2], 2)) + '   ' + str(round(accuracy[0][3], 2)) + '   ' + str(round(accuracy[0][4], 2)) + '   ' 
                  + str(round(accuracy[0][5], 2)) + '   ' + str(round(accuracy[0][6], 2)) + '   ' + str(round(accuracy[0][7], 2)) + '   ' 
                  + str(round(accuracy[0][8], 2)) + '   ' + str(round(accuracy[0][9], 2)) + '\n')
    outfile.write('RBFSVM | ' + str(round(accuracy[1][0], 2)) + '    ' + str(round(accuracy[1][1], 2)) + '   ' 
                  + str(round(accuracy[1][2], 2)) + '   ' + str(round(accuracy[1][3], 2)) + '   ' + str(round(accuracy[1][4], 2)) + '   ' 
                  + str(round(accuracy[1][5], 2)) + '   ' + str(round(accuracy[1][6], 2)) + '   ' + str(round(accuracy[1][7], 2)) + '   ' 
                  + str(round(accuracy[1][8], 2)) + '   ' + str(round(accuracy[1][9], 2)) + '\n')
    outfile.write('SigSVM | ' + str(round(accuracy[2][0], 2)) + '    ' + str(round(accuracy[2][1], 2)) + '   ' 
                  + str(round(accuracy[2][2], 2)) + '   ' + str(round(accuracy[2][3], 2)) + '   ' + str(round(accuracy[2][4], 2)) + '   ' 
                  + str(round(accuracy[2][5], 2)) + '   ' + str(round(accuracy[2][6], 2)) + '   ' + str(round(accuracy[2][7], 2)) + '   ' 
                  + str(round(accuracy[2][8], 2)) + '   ' + str(round(accuracy[2][9], 2)) + '\n')
    outfile.write('Rand F | ' + str(round(accuracy[3][0], 2)) + '    ' + str(round(accuracy[3][1], 2)) + '    ' 
                  + str(round(accuracy[3][2], 2)) + '   ' + str(round(accuracy[3][3], 2)) + '   ' + str(round(accuracy[3][4], 2)) + '   ' 
                  + str(round(accuracy[3][5], 2)) + '   ' + str(round(accuracy[3][6], 2)) + '   ' + str(round(accuracy[3][7], 2)) + '   ' 
                  + str(round(accuracy[3][8], 2)) + '   ' + str(round(accuracy[3][9], 2)) + '\n')
    outfile.write('NBayes | ' + str(round(accuracy[4][0], 2)) + '    ' + str(round(accuracy[4][1], 2)) + '    ' 
                  + str(round(accuracy[4][2], 2)) + '   ' + str(round(accuracy[4][3], 2)) + '   ' + str(round(accuracy[4][4], 2)) + '   ' 
                  + str(round(accuracy[4][5], 2)) + '   ' + str(round(accuracy[4][6], 2)) + '   ' + str(round(accuracy[4][7], 2)) + '   ' 
                  + str(round(accuracy[4][8], 2)) + '   ' + str(round(accuracy[4][9], 2)) + '\n')
    outfile.write('MLP NN | ' + str(round(accuracy[5][0], 2)) + '    ' + str(round(accuracy[5][1], 2)) + '    ' 
                  + str(round(accuracy[5][2], 2)) + '   ' + str(round(accuracy[5][3], 2)) + '   ' + str(round(accuracy[5][4], 2)) + '   ' 
                  + str(round(accuracy[5][5], 2)) + '   ' + str(round(accuracy[5][6], 2)) + '   ' + str(round(accuracy[5][7], 2)) + '   ' 
                  + str(round(accuracy[5][8], 2)) + '   ' + str(round(accuracy[5][9], 2)) + '\n')
    
    x_vals = [100, 80, 60, 50, 40, 30, 20, 15, 10, 5]

    plt.plot(x_vals, accuracy[0], label='Linear SVM')
    plt.plot(x_vals, accuracy[1], label='RBF SVM')
    plt.plot(x_vals, accuracy[2], label='Sigmoid SVM')
    plt.plot(x_vals, accuracy[3], label='Random Forest')
    plt.plot(x_vals, accuracy[4], label='Gaussian Naive Bayes')
    plt.plot(x_vals, accuracy[5], label='Multi Layer Perceptron')

    plt.xlabel('min entries per species')
    plt.ylabel('accuracy')
    plt.title('Comparison of AVONET models')
    plt.legend()

    plt.savefig('model_test_plot.png')

    plt.clf()

    print('Basic model testing complete.\n\n')

    outfile.close()

def test_rf_variants(clean_files):

    print('Testing Random Forest Criterions...\n')

    X_train = [[]] * 10
    X_test = [[]] * 10
    y_train = [[]] * 10
    y_test = [[]] * 10

    for i in range(10):
        my_features , my_labels = load_data(clean_files[i])

        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(my_features, my_labels, test_size=0.2, random_state=42)

    outfile = open('AVONET_RF_Report.txt', 'w')

    accuracy = [[0] * 10 for i in range(5)]

    print('Testing Gini...\n')

    # random forest gini
    outfile.write('Results for Random Forest Gini - \n')
    for i in range(10):
        report , accuracy[0][i] = randomforest(X_train[i], X_test[i], y_train[i], y_test[i])
        outfile.write(clean_files[i] + ' -' + '\n' + '\n')
        outfile.write(report + '\n' + '\n')

    print('Testing Entropy...\n')

    # random forest entropy
    outfile.write('Results for Random Forest Entropy - \n')
    for i in range(10):
        report , accuracy[1][i] = randomforest(X_train[i], X_test[i], y_train[i], y_test[i], 'entropy')
        outfile.write(clean_files[i] + ' -' + '\n' + '\n')
        outfile.write(report + '\n' + '\n')

    print('Testing Log Loss...\n')

    # random forest log_loss
    outfile.write('Results for Random Forest Log Loss - \n')
    for i in range(10):
        report , accuracy[2][i] = randomforest(X_train[i], X_test[i], y_train[i], y_test[i], 'log_loss')
        outfile.write(clean_files[i] + ' -' + '\n' + '\n')
        outfile.write(report + '\n' + '\n')

    # printing accuracy to report file
    outfile.write(' Model  | 100  |  80  |  60  |  50  |  40  |  30  |  20  |  15  |  10  |  5\n')
    outfile.write('  Gini  | ' + str(round(accuracy[0][0], 2)) + '    ' + str(round(accuracy[0][1], 2)) + '   ' 
                  + str(round(accuracy[0][2], 2)) + '   ' + str(round(accuracy[0][3], 2)) + '   ' + str(round(accuracy[0][4], 2)) + '   ' 
                  + str(round(accuracy[0][5], 2)) + '   ' + str(round(accuracy[0][6], 2)) + '   ' + str(round(accuracy[0][7], 2)) + '   ' 
                  + str(round(accuracy[0][8], 2)) + '   ' + str(round(accuracy[0][9], 2)) + '\n')
    outfile.write('Entropy | ' + str(round(accuracy[1][0], 2)) + '    ' + str(round(accuracy[1][1], 2)) + '    ' 
                  + str(round(accuracy[1][2], 2)) + '   ' + str(round(accuracy[1][3], 2)) + '   ' + str(round(accuracy[1][4], 2)) + '   ' 
                  + str(round(accuracy[1][5], 2)) + '   ' + str(round(accuracy[1][6], 2)) + '   ' + str(round(accuracy[1][7], 2)) + '   ' 
                  + str(round(accuracy[1][8], 2)) + '   ' + str(round(accuracy[1][9], 2)) + '\n')
    outfile.write('LogLoss | ' + str(round(accuracy[2][0], 2)) + '    ' + str(round(accuracy[2][1], 2)) + '    ' 
                  + str(round(accuracy[2][2], 2)) + '   ' + str(round(accuracy[2][3], 2)) + '   ' + str(round(accuracy[2][4], 2)) + '   ' 
                  + str(round(accuracy[2][5], 2)) + '   ' + str(round(accuracy[2][6], 2)) + '   ' + str(round(accuracy[2][7], 2)) + '   ' 
                  + str(round(accuracy[2][8], 2)) + '   ' + str(round(accuracy[2][9], 2)) + '\n')
    
    x_vals = [100, 80, 60, 50, 40, 30, 20, 15, 10, 5]

    plt.plot(x_vals, accuracy[0], label='Gini')
    plt.plot(x_vals, accuracy[1], label='Entropy')
    plt.plot(x_vals, accuracy[2], label='Log Loss')

    plt.xlabel('min entries per species')
    plt.ylabel('accuracy')
    plt.title('Comparison of Random Forest Criterions')
    plt.legend()

    plt.savefig('rf_test_plot.png')

    plt.clf()

    print('Random Forest Criterion testing complete.\n\n')

    outfile.close()


def test_rf_gini_hyperparams(clean_files):

    print('Testing Random Forest Hyperparameters...\n')

    X_train = [[]] * 10
    X_test = [[]] * 10
    y_train = [[]] * 10
    y_test = [[]] * 10

    for i in range(10):
        my_features , my_labels = load_data(clean_files[i])

        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(my_features, my_labels, test_size=0.2, random_state=42)


    outfile = open('AVONET_RF_GridSearch_Report.txt', 'w')

    gridsearch_results = [] * 10

    print('Testing using GridSearch...\n')

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    # random forest gini
    outfile.write('Results for Random Forest Grid Search - \n')
    for i in range(10):
        report = randomforest_gridsearch(X_train[i], X_test[i], y_train[i], y_test[i], param_grid)
        outfile.write(clean_files[i] + ' -' + '\n' + '\n')
        outfile.write('Best Params -\n' + str(report.best_params_) + '\n')
        outfile.write('Best Score -\n' + str(report.best_score_) + '\n')
        outfile.write('Full CV Results -\n' + str(report.cv_results_) + '\n')
    
    print('Random Forest Hyperparameters testing complete.\n\n')

    outfile.close()

def train_and_save_best_model(clean_files):
    
    my_features , my_labels = load_data(clean_files[5])

    X_train, X_test, y_train, y_test = train_test_split(my_features, my_labels, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(criterion='gini', random_state=42, bootstrap=True, max_features='sqrt', n_estimators=50)

    rf_model.fit(X_train, y_train)

    with open('rf_final_model_30.pkl', 'wb') as dumpfile:
        pickle.dump(rf_model, dumpfile)

    return X_test , y_test

def load_and_test_best_model(filename, X_test , y_test):
    with open(filename, 'rb') as loadfile:
        model = pickle.load(loadfile)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        report = classification_report(y_test, y_pred)

        print('Accuracy: ' + str(accuracy) + '\n')

        print('Classification Report: \n\n' + report)

def visualize_trees(model_filename):
    with open(model_filename, 'rb') as loadfile:
        model = pickle.load(loadfile)

        print('Num trees:' + str(len(model.estimators_)) )

        plt.figure(figsize=(20, 10))
        for i in range(3):  # Change the number to visualize more or fewer trees
            plt.subplot(1, 3, i + 1)
            plot_tree(model.estimators_[i], filled=True)
            plt.title(f"Decision Tree {i + 1}")
        
        plt.savefig('tree_visualization.png')

        plt.clf()





clean_files = generate_test_files()

# test_models(clean_files)

# test_rf_variants(clean_files)

# test_rf_gini_hyperparams(clean_files)

test_features , test_labels = train_and_save_best_model(clean_files)

load_and_test_best_model('rf_final_model_30.pkl', test_features, test_labels)

visualize_trees('rf_final_model_30.pkl')
