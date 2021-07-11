from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# define Gaussain NB & KNeighborsClassifier classifier
clf1 = GaussianNB()
clf2 = KNeighborsClassifier(n_neighbors=5)

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    clf1.fit(X_train, y_train)
    # calculate the print the accuracy score
    acc1 = accuracy_score(y_test,clf1.predict(X_test))

    # train the model
    clf2.fit(X_train, y_train)
    # calculate the print the accuracy score
    acc2 = accuracy_score(y_test, clf2.predict(X_test))

    if acc1 > acc2:
        acc = acc1
        clfMostAcc = "clf1"
    else:
        acc = acc2
        clfMostAcc = "clf2"
    print(f"Model trained with accuracy: {round(acc, 3)}")
    return clfMostAcc


# function to predict the flower using the model
def predict(query_data, clfMostAcc):
    x = list(query_data.dict().values())
    prediction = ""
    if clfMostAcc == "clf1":
        prediction = clf1.predict([x])[0]
    elif clfMostAcc == "clf2":
        prediction = clf2.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data,clfMostAcc):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    if clfMostAcc == "clf1":
        clf1.fit(X, y)
    elif clfMostAcc == "clf2":
        clf2.fit(X, y)
