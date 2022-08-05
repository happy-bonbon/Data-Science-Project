import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

"""
The Accuracy Score with Max depth of 2 is 0.46938775510204084
The Accuracy Score with Max depth of 3 is 0.4897959183673469
The Accuracy Score with Max depth of 4 is 0.5102040816326531
The Accuracy Score with Max depth of 5 is 0.5102040816326531
The Accuracy Score with Max depth of 6 is 0.40816326530612246
The Accuracy Score with Max depth of 7 is 0.42857142857142855
The Accuracy Score with Max depth of 8 is 0.40816326530612246
The Accuracy Score with Max depth of 9 is 0.4489795918367347
The Accuracy Score with Max depth of 10 is 0.4489795918367347
"""


def main():
    # need k-fold classification
    dataset = pd.read_csv('Data/combined_detail_cleaned.csv')
    # display(dataset.head)

    # all chooen features are numerical
    features = dataset[['temperature_min','temperature_max', 'rainfall', 
                        'evaporation', 'sunshine', 'max_wind_speed', 
                        'temperature_9am', 'humidity_9am', 'cloud_9am', 
                        'wind_speed_9am', 'pressure_9am', 'temperature_3pm', 
                        'humidity_3pm', 'cloud_3pm',
                        'wind_speed_3pm', 'pressure_3pm']]
    


    classlabel = dataset['max_price_category']

    features_train, feature_test, class_train, class_test = train_test_split(
        features, classlabel, train_size=0.8, random_state=1)

    for x_times in range(2, 11):
        dt = DecisionTreeClassifier(
            criterion='entropy', random_state=1, max_depth = x_times)
        # DecisionTreeClassifier reduces the leaves of classification
        # scale the features
        dt.fit(features_train, class_train)
        predictions = dt.predict(feature_test)
        print(f'The Accuracy Score with Max depth of {x_times} is {accuracy_score(class_test, predictions)}')

    # plt.figure(figsize=(25, 20))
    # featurenames = ['temperature_min','temperature_max', 'rainfall', 
    #                     'evaporation', 'sunshine', 'max_wind_speed', 
    #                     'temperature_9am', 'humidity_9am', 'cloud_9am', 
    #                     'wind_speed_9am', 'pressure_9am', 'temperature_3pm', 
    #                     'humidity_3pm', 'cloud_3pm',
    #                     'wind_speed_3pm', 'pressure_3pm']
    # tree.plot_tree(dt, feature_names=featurenames)
    # plt.show()

    # knn


if __name__ == "__main__":
    main()
