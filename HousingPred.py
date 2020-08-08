"""
A machine learning algorithm to predict if a person owns, rents or lives in their house for free.
The training data is taken imported straight from GitHub so there is no need to download the csv file.
Joe Emmens, 08/08/2020
"""

def housing():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd
    url = "https://raw.githubusercontent.com/JoeEmmens/HousingCredit/master/german_credit_data.csv"
    credit = pd.read_csv(url, index_col="Unnamed: 0").dropna()

    credit = pd.get_dummies(credit, columns=["Sex"])
    housing_type = dict(zip(credit.Housing.unique(), [1, 2, 3]))
    credit["Housing_lbl"] = credit["Housing"].map(housing_type)

    X = credit[["Age", "Credit amount", "Job"]]
    y = credit["Housing"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    knn.score(X_test, y_test)

    user_age = float(input("What is your age? "))
    user_credit = float(input("How large was the last loan you took out? "))
    user_job = input("What type of job do you have, unskilled, skilled or highly skilled? ")
    user_residency = input("Are you a local resident, yes/no? ")

    user_job_idx = ()
    if user_job == "unskilled" and user_residency == "no":
        user_job_idx = 1
    if user_job == "unskilled" and user_residency == "yes":
        user_job_idx = 2
    if user_job == "skilled":
        user_job_idx = 3
    else:
        user_job_idx = 4

    h_prediction = knn.predict([[user_age, user_credit, user_job_idx]])
    r_o = ["rent", "own"]
    if h_prediction[0] in r_o:
        print("It is most likely that you", h_prediction[0], "your house")
    else:
        print("It is most likely you currently stay at your house for ", h_prediction[0])
