from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(X,y,model):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model.fit(X_train,y_train)

    preds = model.predict(X_test)

    score = accuracy_score(y_test,preds)

    return model,score