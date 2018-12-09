from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=.4)

n_samples, h, w = lfw_people.images.shape

X = lfw_people.data
n_features = X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

n_components = 150

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(x_train)

eigenfaces = pca.components_.reshape((n_components, h, w))

x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf = clf.fit(x_train_pca, y_train)

y_pred = clf.predict(x_test_pca)

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))



