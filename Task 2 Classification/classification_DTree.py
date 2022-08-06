import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# import matplotlib.pyplot as plt
# from IPython.display import display
# from sklearn import tree


def main():
    '''classification for Max Price Category VS Weather O(n**2)'''
    
    # read cvs files with different cleaning methods
    dataset = pd.read_csv('Data/combined_detail_cleaned.csv')
    # dataset = pd.read_csv('Data/combined_original.csv')
    # dataset = pd.read_csv('Data/combined_null_delete.csv')
    # dataset = pd.read_csv('Data/combined-detail-cleaned-AVG-temp.csv')

    # all chooen features are numerical
    features = dataset[['temperature_min','temperature_max', 'rainfall', 
                        'evaporation', 'sunshine', 'max_wind_speed', 
                        'temperature_9am', 'humidity_9am', 'cloud_9am', 
                        'wind_speed_9am', 'pressure_9am', 'temperature_3pm', 
                        'humidity_3pm', 'cloud_3pm',
                        'wind_speed_3pm', 'pressure_3pm']]
    
    classlabel = dataset['max_price_category']

    # K-fold Method (cross validation) with Classification Accuracy Formula
    k = 10
    kf = KFold(n_splits = k, shuffle = True, random_state = 88)
    
    # x_times is the max depth of the Decision Tree Classifier (parameter tuning)
    for x_times in range(2, 13):
        classification_accuracy = []
        
        # Implementation of K-fold Method
        for train_index, test_index in kf.split(features):
            
            # Split Training and Test sets
            features_train, feature_test = features.iloc[train_index, :], features.iloc[test_index, :]
            class_train, class_test = classlabel[train_index], classlabel[test_index]
            
            # Scaling the features
            scalar = preprocessing.StandardScaler().fit(features_train)
            features_train = scalar.transform(features_train)
            feature_test = scalar.transform(feature_test)
        
            # Step 1: Instantiate 
            dt = DecisionTreeClassifier(
                criterion='entropy', random_state=88, max_depth = x_times)
            
            # Step 2: Fit
            dt.fit(features_train, class_train)
            
            # Step 3: Predict
            predictions = dt.predict(feature_test)
            
            # Step 4: Evaluate
            my_score=accuracy_score(class_test, predictions)
            classification_accuracy.append(my_score)
        print(f'The AVG Decision Tree Accuracy Score with Max depth of {x_times} is {sum(classification_accuracy)/k}')
        
        #Plot the results
        # plt.figure(figsize=(25, 20))
        # featurenames = ['temperature_avg', 'rainfall', 
        #                 'sunshine']
        # tree.plot_tree(dt, feature_names=featurenames)
        # plt.show()

if __name__ == "__main__":
    main()
