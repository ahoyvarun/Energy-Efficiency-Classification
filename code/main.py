import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import train_mlp, train_logistic_regression
from visualizations import plot_confusion_matrix

df = pd.read_excel("ENB2012_data.xlsx")
df.columns = [
    'Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
    'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution',
    'Heating Load', 'Cooling Load'
]

# Binary target: efficient = 1 if Heating Load < 15
df['Efficient'] = df['Heating Load'].apply(lambda x: 1 if x < 15 else 0)

X = df.iloc[:, :-3]
y = df['Efficient']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate MLP
y_pred_mlp, mlp_model = train_mlp(X_train_scaled, y_train, X_test_scaled, y_test)
plot_confusion_matrix(y_test, y_pred_mlp, title='MLPClassifier Confusion Matrix')

# Train and evaluate Logistic Regression
y_pred_log, log_model = train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
plot_confusion_matrix(y_test, y_pred_log, title='Logistic Regression Confusion Matrix')