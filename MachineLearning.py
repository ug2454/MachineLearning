from sklearn import tree
features=[[300,2],[450,2],[200,8],[150,9]]
labels=["sports-car","sports-car","minivan","minivan"]
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features,labels)
print(clf.predict([[200,8]]))


