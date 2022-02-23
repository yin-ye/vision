import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

def visualizeDataset(dataset, pair_plot=True, box_plot=False, height=2, label_column="Species"):
    if pair_plot:
        sns.pairplot(dataset.drop("Id", axis=1), hue=label_column, height=height, markers=["o", "s", "D"])
    if box_plot:
        dataset.drop("Id", axis=1).boxplot(by=label_column, figsize=(10, 5))
    plt.show()


def determineBestK(X_train, y_train, num_cross_val=5, length=50):
    # create a list of k values to test from
    k_range = list(range(1, length, 2))
    # Loop through the mean accuracy of each knn fitting and keep only the best k values
    # and accuracy
    accuracy = 0
    best_k = 0
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=num_cross_val, scoring='accuracy')
        if scores.mean() > accuracy:
            accuracy = scores.mean()
            best_k = k

    print(f'The optimal number of neighbors is {best_k}, with accuracy of {accuracy}')
    return k, accuracy


if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv("dataset/Iris.csv")
    # Split dataset into features and labels
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    X = dataset[features].values
    y = dataset['Species'].values
    # Knn cannot operate on string values, so encoding to a numerical format is necessary
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # Visualize dataset
    visualizeDataset(dataset)
    # Split the dataset into train, validate and test ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Choose Best K
    k, accuracy = determineBestK(X_train, y_train)
    # Fitting the model using pretrained k and evaluate the model
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    # Display confusion Matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(confusion_matrix, X_test, y_test)
    plt.show()
    # Calculating Model Accuracy
    accuracy = accuracy_score(y_test, y_pred) * 100

    print('Accuracy of the model:' + str(round(accuracy, 2)) + ' %.')
