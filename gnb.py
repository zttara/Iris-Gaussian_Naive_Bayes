import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

#train = pd.read_csv('E:/examples/iris_GNB/train.csv')
#test = pd.read_csv('E:/examples/iris_GNB/test.csv')
datasets = pd.read_csv('E:/examples/iris_GNB/iris.csv')

selected_features = ['SepalLength','SepalWidth','PetalLength','PetalWidth']

X = datasets[selected_features]
y = datasets['Name']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

#X_train = train[selected_features]
#y_train= train['Name']

#X_test = test[selected_features]
#y_test = test['Name']

clf = GaussianNB()
clf.fit(X_train, y_train)
predict = clf.predict(X_test)
pred_result = pd.DataFrame({'real_label':y_test,'pred_label':predict})
pred_result.to_csv('E:/examples/iris_GNB/pred_result.csv',index=False)
print 'The accurary of GaussianNB on testing set is: ', clf.score(X_test, y_test)

#print(clf.predict([[ ]]))
'''
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))

print(clf_pf.predict([[ ]]))
'''