from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import joblib
from dataset import X, y


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

reg = LogisticRegression().fit(X_train, y_train)
reg.score(X_test, y_test)

joblib.dump(reg, r'exam/results/logreg_model.joblib')