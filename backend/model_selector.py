from sklearn.ensemble import RandomForestClassifier

def get_model(problem):

    if problem == "classification":
        return RandomForestClassifier()