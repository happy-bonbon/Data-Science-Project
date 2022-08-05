import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    # need k-fold classification
    dataset = pd.read_csv('final.csv')
    # display(dataset.head)

    features = dataset[['Maximum temperature (°C)', 'Minimum temperature (°C)',
                        'Minimum temperature (°C)', 'Evaporation (mm)', 'Sunshine (hours)']]
    classlabel = dataset['pricecategory']

    features_train, feature_test, class_train, class_test = train_test_split(
        features, classlabel, train_size=0.8, random_state=1)

    dt = DecisionTreeClassifier(
        criterion='entropy', random_state=1, max_depth=3)
    # DecisionTreeClassifier reduces the leaves of classification
    # scale the features

    dt.fit(features_train, class_train)
    predictions = dt.predict(feature_test)

    print(accuracy_score(class_test, predictions))

    fig = plt.figure(figsize=(25, 20))
    featurenames = ['Maximum temperature (°C)', 'Minimum temperature (°C)', 'Rainfall (mm)',
                    'Minimum temperature (°C)', 'Evaporation (mm)', 'Sunshine (hours)']
    tree.plot_tree(dt, feature_names=featurenames)
    # plt.show()

    # knn


if __name__ == "__main__":
    main()
