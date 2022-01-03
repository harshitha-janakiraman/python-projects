#!/usr/bin/env python
# coding: utf-8

# Collaborators: Yi Zhao, Radian Gondokaryono

# #q3.0: Load and Plot Data
# 

# In[ ]:


# Only run this block if you are using google collab!
from google.colab import drive
drive.mount('/content/drive/')#
get_ipython().run_line_magic('cd', 'drive/MyDrive/U of T/CSC2515/CSC2515 Assignments/A3/Yi')


get_ipython().system('ls')


# In[ ]:


'''
Question 3.0 Skeleton Code

Here you should load the data and plot
the means for each of the digit classes.
'''

import data
import numpy as np
import matplotlib.pyplot as plt

def plot_means(train_data, train_labels):
    means = []
    for i in range(0, 10):
        x=0
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        x = np.mean(i_digits, axis = 0)
        x = np.reshape(x, (8,8))
        means.append(x)

    # Plot all means on same axis
    all_concat = np.concatenate(means, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()


# In[ ]:


# Load data
train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip(r'./a3digits.zip', r'./data')

# Plot Means
plot_means(train_data, train_labels)

# Save best classifier models to this dict
models = {}


# # q3.1: K Nearest Neighbors
# 

# In[ ]:


'''
Question 3.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

        # One-hot encoding
        # Only works on int labels
        int_train_labels = train_labels.astype(int)
        self.encoded_labels = np.zeros((int_train_labels.size, int_train_labels.max()+1))
        self.encoded_labels[np.arange(int_train_labels.size), int_train_labels] = 1

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = test_point.reshape(1, -1)

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        return self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())

    def query_knn(self, test_point, k):
        '''
        Returns the digit label in int Given a 1d array test point, and k number
        of neighbors to search

        Input: test_point is a 1d numpy array
        Output: int digit label
        '''
  
        dist = self.l2_distance(test_point)
        sorted_encoded_labels = self.encoded_labels[dist[:,0].argsort()]
        sum_label = sorted_encoded_labels[0:k].sum(axis = 0)
        # When there is a tie between sum label. 3.1.2, make a random choice
        max_digits = np.argwhere(sum_label == np.amax(sum_label)).flatten() #The index are the digits!
        return np.random.choice(max_digits)

    def predict(self, k, eval_data):
        predictions = []
        for i in range(len(eval_data)):
            predictions.append(self.query_knn(eval_data[i],k))
        return np.array(predictions) 

def cross_validation(train_data, train_labels, k_range=range(1,16), k_fold_splits=10):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    kf = KFold(n_splits=k_fold_splits)
    mean_train_acc = []
    mean_test_acc = []
    count = 0
    
    for i, k in enumerate(k_range):
        train_accuracy = []
        test_accuracy = [] 
        for train_index, test_index in kf.split(train_data):
            train_data1, test_data1 = train_data[train_index], train_data[test_index]
            train_labels1, test_labels1 = train_labels[train_index], train_labels[test_index]
            
            knn = KNearestNeighbor(train_data1, train_labels1)
            train_accuracy.append(classification_accuracy(knn, k, train_data1, train_labels1))
            test_accuracy.append(classification_accuracy(knn, k, test_data1, test_labels1))


        mean_train_acc.append(np.mean(np.array([train_accuracy])))
        mean_test_acc.append(np.mean(np.array([test_accuracy])))
        print("k: {}".format(k))
        print("Mean training accuracy after cross validation:\n", mean_train_acc[i])  
        print("Mean validation accuracy after cross validation:\n", mean_test_acc[i])
        pass

    best_k_idx = np.argmax(mean_test_acc)
    best_k = k_range[best_k_idx]
    best_k_mean_train_acc = mean_train_acc[best_k_idx]
    best_k_mean_test_acc = mean_test_acc[best_k_idx]

    return best_k, best_k_mean_train_acc, best_k_mean_test_acc

def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    eval_predict_ = []
    eval_accuracy = 0

    for i in range(len(eval_data)):
        eval_predict_.append(knn.query_knn(eval_data[i],k))
        
    eval_accuracy = ((eval_predict_ == eval_labels).sum() / len(eval_labels))
    return eval_accuracy 


# In[ ]:


# 3.1.1 Build and run 
knn = KNearestNeighbor(train_data, train_labels)
models['knn'] = knn

# 3.1.1.a k = 1
print("k = 1")
print("training accuracy: ", classification_accuracy(knn, 1, train_data, train_labels))
print("test accuracy: ", classification_accuracy(knn, 1, test_data, test_labels))

# 3.1.1.b k = 15
print('k = 15')
print("training accuracy: ", classification_accuracy(knn, 15, train_data, train_labels))
print("test accuracy: ", classification_accuracy(knn, 15, test_data, test_labels))


# In[ ]:


# 3.1.3 Run kfold
k_range = range(1,16)
k_fold_splits = 10
best_k, best_k_cross_mean_train_acc, best_k_cross_mean_test_acc = cross_validation(train_data, train_labels, k_range, k_fold_splits)

best_k_train_acc = classification_accuracy(knn, best_k, train_data, train_labels)
best_k_test_acc = classification_accuracy(knn, best_k, test_data, test_labels)

print("-------------------------")
print("best_k: ", best_k)
print("best_k_cross_mean_train_acc: ", best_k_cross_mean_train_acc)
print("best_k_cross_mean_validation_acc: ", best_k_cross_mean_test_acc)
print("best_k_train_acc ", best_k_train_acc)
print("best_k_test_acc: ", best_k_test_acc)
print("-------------------------")


# # q3.2: Multi-layer Perceptron, Support Vector Classifier, and AdaBoost
# 
# 

# In[ ]:


import time
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve
from sklearn import metrics


# In[ ]:


# Train validation test split
train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip(r'./a3digits.zip', r'./data')

train_dev_data = np.copy(train_data)
train_dev_labels = np.copy(train_labels)

train_data, dev_data, train_labels, dev_labels = train_test_split(train_data, train_labels, test_size=0.33, random_state=42)
for data in [train_data, train_labels, dev_data, dev_labels, test_data, test_labels, train_dev_data, train_dev_labels]:
  print(data.shape)

def int_label_to_one_hot(labels):
  int_labels = labels.astype(int)
  one_hot = np.zeros((int_labels.size, int_labels.max()+1))
  one_hot[np.arange(int_labels.size),int_labels] = 1
  return one_hot

# Change to onehot encoding for mlp
y_train_onehot = int_label_to_one_hot(train_labels)
y_dev_onehot = int_label_to_one_hot(dev_labels)
y_test_onehot = int_label_to_one_hot(test_labels)
y_train_dev_onehot = int_label_to_one_hot(train_dev_labels)


# In[ ]:


# 3.2.1 Multi-layer Perceptron Classifier 
start_iter=50
end_iter=360+1
iter_acc_dict={}
for i in range(start_iter,end_iter,20):
  clf_mlp = MLPClassifier(random_state=1, 
                        hidden_layer_sizes=(64,64),#hidden_layer_sizestuple, length = n_layers - 2, default=(100,)
                        # activation='relu', #activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
                        solver='adam', #solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
                        # early_stopping=True,#early_stoppingbool, default=False
                        # alpha=0.0001, # loat, default=0.0001 L2 penalty (regularization term) parameter.
                        # learning_rate='adaptive', #learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
                        max_iter=i
                        )
                        
  start = time.time()
  clf_mlp.fit(train_data, y_train_onehot) 
  end=time.time()
  predicted = np.argmax(clf_mlp.predict(dev_data), axis=1)
  target = np.argmax(y_dev_onehot, axis=1)
  acc = (predicted == target).sum()/len(dev_data)
  acc = round(acc,5)
  print(f'{i:4},Accuracy:{str(acc*100):<6}%, took {end-start}s ')
  iter_acc_dict[i]=acc

plt.figure(figsize=(5, 3))
plt.plot(list(iter_acc_dict.keys()), list(iter_acc_dict.values()))
plt.xlabel('Max Iterations')
plt.ylabel('Test Accuarcy')
plt.title('Iterations vs Accuarcy')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


# One layer hidden size vs accuracy
start_hidden=10
end_hidden=350+1
hidden_size_dict={}
for i in range(start_hidden,end_hidden,20):
  clf_mlp = MLPClassifier(random_state=1, 
                        hidden_layer_sizes=(i,),#hidden_layer_sizestuple, length = n_layers - 2, default=(100,)
                        # activation='relu', #activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
                        solver='adam', #solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
                        # early_stopping=True,#early_stoppingbool, default=False
                        # alpha=0.001, # loat, default=0.0001 L2 penalty (regularization term) parameter.
                        # learning_rate='adaptive', #learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
                        # max_iter=300
                        early_stopping=True
                        )
  start = time.time()
  clf_mlp.fit(train_data, y_train_onehot) 
  end=time.time()
  predicted = np.argmax(clf_mlp.predict(dev_data), axis=1)
  target = np.argmax(y_dev_onehot, axis=1)
  acc = (predicted == target).sum()/len(dev_data)
  acc = round(acc,5)
  print(f'{i:4}, Accuracy: {str(acc*100)[:6]:<6}%, took {str(end-start)[:4]}s')
  hidden_size_dict[i]=acc

plt.figure(figsize=(5, 3))
plt.plot(list(hidden_size_dict.keys()), list(hidden_size_dict.values()))
plt.xlabel('hidden size')
plt.ylabel('Test Accuarcy')
plt.title('Hidden Size(one layer) vs Accuracy')
plt.show()


# In[ ]:


# Two layers hidden size vs accuracy
start_hidden=10
end_hidden=500+1
hidden_size_dict={}
for i in range(start_hidden,end_hidden,20):
  clf_mlp = MLPClassifier(random_state=1, 
                        hidden_layer_sizes=(i,i),#hidden_layer_sizestuple, length = n_layers - 2, default=(100,)
                        # activation='relu', #activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
                        solver='adam', #solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
                        # early_stopping=True,#early_stoppingbool, default=False
                        # alpha=0.001, # loat, default=0.0001 L2 penalty (regularization term) parameter.
                        # learning_rate='adaptive', #learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
                        # max_iter=300
                        early_stopping=True
                        )
  start = time.time()
  clf_mlp.fit(train_data, y_train_onehot) 
  end=time.time()
  predicted = np.argmax(clf_mlp.predict(dev_data), axis=1)
  target = np.argmax(y_dev_onehot, axis=1)
  acc = (predicted == target).sum()/len(dev_data)
  print(f'{i:4}, Accuracy: {str(acc*100)[:6]:<6}%, took {str(end-start)[:4]}s')
  hidden_size_dict[i]=acc

plt.figure(figsize=(5, 3))
plt.plot(list(hidden_size_dict.keys()), list(hidden_size_dict.values()))
plt.xlabel('hidden size')
plt.ylabel('Test Accuarcy')
plt.title('Hidden Size(two layers) vs Accuarcy')
plt.show()


# In[ ]:


# Three layers hidden size vs accuracy
start_hidden=30
end_hidden=500+1
hidden_size_dict={}
for i in range(start_hidden,end_hidden,20):
  clf_mlp = MLPClassifier(random_state=1, 
                        hidden_layer_sizes=(i,i,i),#hidden_layer_sizestuple, length = n_layers - 2, default=(100,)
                        # activation='relu', #activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
                        solver='adam', #solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
                        # early_stopping=True,#early_stoppingbool, default=False
                        # alpha=0.001, # loat, default=0.0001 L2 penalty (regularization term) parameter.
                        # learning_rate='adaptive', #learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
                        # max_iter=300
                        early_stopping=True
                        )
  start = time.time()
  clf_mlp.fit(train_data, y_train_onehot)
  end=time.time()
  predicted = np.argmax(clf_mlp.predict(dev_data), axis=1)
  target = np.argmax(y_dev_onehot, axis=1)
  acc = (predicted == target).sum()/len(dev_data)
  print(f'{i:4}, Accuracy: {str(acc*100)[:6]:<6}%, took {str(end-start)[:4]}s')
  hidden_size_dict[i]=acc

plt.figure(figsize=(5, 3))

plt.plot(list(hidden_size_dict.keys()), list(hidden_size_dict.values()))
plt.xlabel('hidden size')
plt.ylabel('Test Accuarcy')
plt.title('Hidden Size(three layers) vs Accuarcy')
plt.show()


# In[ ]:


# Four layers hidden size vs accuracy
start_hidden=30
end_hidden=500+1
hidden_size_dict={}
for i in range(start_hidden,end_hidden,30):
  clf_mlp = MLPClassifier(random_state=1, 
                        hidden_layer_sizes=(i,i,i,i),#hidden_layer_sizestuple, length = n_layers - 2, default=(100,)
                        # activation='relu', #activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
                        solver='adam', #solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
                        # early_stopping=True,#early_stoppingbool, default=False
                        # alpha=0.001, # loat, default=0.0001 L2 penalty (regularization term) parameter.
                        # learning_rate='adaptive', #learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
                        # max_iter=300
                        early_stopping=True
                        )
  start = time.time()
  clf_mlp.fit(train_data, y_train_onehot)
  end=time.time()
  predicted = np.argmax(clf_mlp.predict(dev_data), axis=1)
  target = np.argmax(y_dev_onehot, axis=1)
  acc = (predicted == target).sum()/len(dev_data)
  print(f'{i:4}, Accuracy: {str(acc*100)[:6]:<6}%, took {str(end-start)[:4]}s')
  hidden_size_dict[i]=acc

plt.figure(figsize=(5, 3))
plt.plot(list(hidden_size_dict.keys()), list(hidden_size_dict.values()))
plt.xlabel('hidden size')
plt.ylabel('Test Accuarcy')
plt.title('Hidden Size(four layers) vs Accuarcy')
plt.show()


# In[ ]:


# activation vs acc
activation_dict={}
for act in ['identity','logistic','tanh','relu']:
  clf_mlp = MLPClassifier(random_state=1, 
                        hidden_layer_sizes=(390,390,390,390),#hidden_layer_sizestuple, length = n_layers - 2, default=(100,)
                        activation=act, #activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
                        solver='adam', #solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
                        # early_stopping=True,#early_stoppingbool, default=False
                        # alpha=0.001, # loat, default=0.0001 L2 penalty (regularization term) parameter.
                        # learning_rate='adaptive', #learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
                        # max_iter=300
                        early_stopping=True
                        )
  start = time.time()
  clf_mlp.fit(train_data, y_train_onehot)
  end=time.time()
  predicted = np.argmax(clf_mlp.predict(dev_data), axis=1)
  target = np.argmax(y_dev_onehot, axis=1)
  acc = (predicted == target).sum()/len(dev_data)
  print(f'{act:10}, Accuracy: {str(acc*100)[:6]:<6}%, took {str(end-start)[:4]}s')
  activation_dict[act]=acc


# In[ ]:


# alpha vs accuracy
alpha_dict={}
train_acc=[]
dev_acc=[]
for a in [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.005,0.008,0.01,0.015,0.02,0.025]:
  clf_mlp = MLPClassifier(random_state=1, 
                        hidden_layer_sizes=(390,390,390,390),#hidden_layer_sizestuple, length = n_layers - 2, default=(100,)
                        # activation=act, #activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
                        solver='adam', #solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
                        # early_stopping=True,#early_stoppingbool, default=False
                        alpha=a, # loat, default=0.0001 L2 penalty (regularization term) parameter.
                        # learning_rate='adaptive', #learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
                        # max_iter=300
                        early_stopping=True
                        )
  start = time.time()
  clf_mlp.fit(train_data, y_train_onehot) 
  end=time.time()
  predicted = np.argmax(clf_mlp.predict(dev_data), axis=1)
  target = np.argmax(y_dev_onehot, axis=1)
  acc = (predicted == target).sum()/len(dev_data)
  print(f'{a:8}, Accuracy: {str(acc*100)[:6]:<6}%, took {str(end-start)[:4]}s')
  dev_acc.append(acc)
  alpha_dict[a]=acc

  #train_acc
  predicted_train = np.argmax(clf_mlp.predict(train_data), axis=1)
  target_train = np.argmax(y_train_onehot, axis=1)
  acc_train = (predicted_train == target_train).sum()/len(train_data)
  train_acc.append(acc_train)


plt.figure(figsize=(5, 3))
plt.plot(list(alpha_dict.keys()), train_acc, 'g')
plt.plot(list(alpha_dict.keys()), dev_acc, 'b')  
plt.xlabel('Alpha')
plt.ylabel('Accuarcy')
plt.title('Alpha vs Accuarcy')
plt.legend(['Train Accuarcy', 'Dev Accuarcy'])
plt.show()


# In[ ]:


alpha = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.005,0.008,0.01,0.015,0.02,0.025]
for a, dev_a in zip(alpha,dev_acc):
  print(f'alpha:{a:10} acc{dev_a:5}')


# In[ ]:


# Best mlp
clf_mlp = MLPClassifier(random_state=1, 
                      hidden_layer_sizes=(390,390,390,390),
                      activation='relu',
                      solver='adam', 
                      alpha=0.0002, #  L2 penalty (regularization term) parameter.
                      # learning_rate='constant', #default
                      # max_iter=200, #default
                      early_stopping=True
                      )

start = time.time()
clf_mlp.fit(train_dev_data, y_train_dev_onehot)
end=time.time()
models['mlp'] = clf_mlp
predicted = np.argmax(clf_mlp.predict(test_data), axis=1)
target = np.argmax(y_test_onehot, axis=1)
acc = (predicted == target).sum()/len(test_data)
print(f'Accuracy: {str(acc*100)[:6]:<6}%, took {str(end-start)[:4]}s')


# In[ ]:


# 3.2.2 SVC grid search
C = [1, 10, 50, 100, 200, 500, 1000]
gamma = [0.01, 0.005, 0.001, 0.0005, 0.0001, 'auto', 'scale']
parameters = {'kernel':('rbf', 'poly', 'sigmoid'), 'gamma': gamma, 'C':C }

svc = svm.SVC()
clf_grid_svc = GridSearchCV(svc, parameters, scoring = 'accuracy')
clf_grid_svc.fit(train_dev_data, train_dev_labels)


# In[ ]:


# Best SVC parameters and test accuracy
print('SVC best parameters ', clf_grid_svc.best_params_)

acc = (clf_grid_svc.predict(test_data) == test_labels).sum()/len(test_data)
print(f'Acc:{acc}')
models['clf_grid_svc'] = clf_grid_svc


# In[ ]:


svc = svm.SVC(kernel='rbf',C=50, gamma='scale')
acc = (clf_grid_svc.predict(test_data) == test_labels).sum()/len(test_data)
print(f'Acc:{acc}')
models['clf_grid_svc'] = clf_grid_svc


# In[ ]:


import pickle
filehandler = open("svm_weight.obj","wb")
pickle.dump(clf_grid_svc,filehandler)
filehandler.close()


# In[ ]:


# 3.2.3 AdaBoost
from sklearn.tree import DecisionTreeClassifier

model_name = "Boost Classifier"
# Running a grid search to find the best hyperparameters for adaboost classifier with decision tree. 
abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

parameters = {'base_estimator__max_depth':[i for i in range(2,7,2)],
              'base_estimator__min_samples_leaf':[5,10],
              'n_estimators': np.arange(400, 701, 100),
              'learning_rate':[0.1, 1]}

clf = GridSearchCV(abc, parameters,verbose=3,scoring='accuracy', n_jobs= 1,cv=5)
clf.fit(train_dev_data, train_dev_labels)
print('abc_dt',clf.best_params_,clf.best_score_)


# In[ ]:


# Increasing number of estimators, decreasing learning rate

model_name = "Boost Classifier increasing estimators, decreasing learning rate"
# Running a grid search to find the best hyperparameters for adaboost classifier with decision tree. 
abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

parameters = {'base_estimator__max_depth':[6],
              'base_estimator__min_samples_leaf':[10],
              'n_estimators': [400,500,600,700],
              'learning_rate':[0.1,0.01,0.001]}

clf = GridSearchCV(abc, parameters,verbose=3,scoring='accuracy', n_jobs= 1,cv=5)
clf.fit(train_dev_data, train_dev_labels)
print('abc_dt',clf.best_params_,clf.best_score_)


# In[ ]:


# Increasing number of estimators, decreasing learning rate

model_name = "Boost Classifier increasing estimators, decreasing learning rate"
# Running a grid search to find the best hyperparameters for adaboost classifier with decision tree. 
abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

parameters = {'base_estimator__max_depth':[6],
              'base_estimator__min_samples_leaf':[5],
              'n_estimators': [400,500,600,700],
              'learning_rate':[0.1,0.01,0.001]}

clf = GridSearchCV(abc, parameters,verbose=3,scoring='accuracy', n_jobs= 1,cv=5)
clf.fit(train_dev_data, train_dev_labels)
print('abc_dt',clf.best_params_,clf.best_score_)


# In[ ]:


# Best AdaBoost:
# abc_dt {'base_estimator__max_depth': 6, 'base_estimator__min_samples_leaf': 5, 'learning_rate': 1, 'n_estimators': 700} 0.9617142857142857
model_name = "AdaBoost Classifier with Decision Tree"
dec_tree = DecisionTreeClassifier(max_depth = 6, min_samples_leaf = 5) 
best_adaboost_model = AdaBoostClassifier(base_estimator=dec_tree, n_estimators = 700, learning_rate = 1, random_state = 0)
best_adaboost_model.fit(train_dev_data, train_dev_labels)
models[model_name]=best_adaboost_model


# In[ ]:


acc = (best_adaboost_model.predict(test_data) == test_labels).sum()/len(test_data)
print(f'Acc:{acc}')


# In[ ]:


models.keys()


# # q3.3: Evaluate Classifiers with Best Parameters Performance

# In[ ]:


def plot_roc(test_labels, predict, model_name):
  y_test = test_labels.astype(int)
  y_score = predict.astype(int)

  y_test_onehot = np.zeros((y_test.size, y_test.max()+1))
  y_test_onehot[np.arange(y_test.size),y_test] = 1
  y_score_onehot = np.zeros((y_score.size, y_score.max()+1))
  y_score_onehot[np.arange(y_score.size),y_score] = 1

  y_score, y_test = y_score_onehot, y_test_onehot
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(10):
      fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

  # Compute micro-average ROC curve and ROC area
  fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  plt.figure(figsize=(13, 8))
  plt.plot(fpr["micro"], tpr["micro"],
          label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]))
  for i in range(10):
      plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                    ''.format(i, roc_auc[i]))

  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve for '+model_name)
  plt.legend(loc="lower right")
  plt.show()


# In[ ]:


# Print and plot performance metrics for all the best models:
# knn, mlp, svc, and adaboost
for model_name, clf in models.items():
  if 'knn' in model_name:
    predict = clf.predict(1, test_data)
  else:
    predict = clf.predict(test_data)
  if model_name[:3]=="mlp":
    predict = np.argmax(predict, axis=1)
  if "svc" in model_name:
    predict = predict.astype(int)
  acc = (predict == test_labels).sum()/len(test_data)
  
  print(f'Model:{model_name}')
  print(f'Test Accuracy: {100*acc:5}%')
  print('Confusion Matrix: ')
  print(confusion_matrix(test_labels, predict))
  print(f"{metrics.classification_report(test_labels, predict)}\n")
  plot_roc(test_labels, predict,model_name)
  # plot_average(test_labels, predict,model_name)
  print('\n\n'+'*'*150+'\n\n')


# In[ ]:


models

