from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_mlp(X_train, y_train, X_test, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    print("MLPClassifier Accuracy:", accuracy_score(y_test, y_pred))
    print("MLPClassifier Classification Report:\n", classification_report(y_test, y_pred))
    return y_pred, mlp

def train_logistic_regression(X_train, y_train, X_test, y_test):
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred))
    print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred))
    return y_pred, log_reg