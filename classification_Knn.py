import pandas as pd
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


def main():
    '''classification for Max Price Category VS Weather'''
    
    # read cvs files with different cleaning methods
    dataset = pd.read_csv('Data/combined_detail_cleaned.csv')
    # dataset = pd.read_csv('Data/combined_original.csv')
    # dataset = pd.read_csv('Data/combined_null_delete.csv')

    # all chooen features are numerical
    features = dataset[['temperature_min','temperature_max', 'rainfall', 
                        'evaporation', 'sunshine', 'max_wind_speed', 
                        'temperature_9am', 'humidity_9am', 'cloud_9am', 
                        'wind_speed_9am', 'pressure_9am', 'temperature_3pm', 
                        'humidity_3pm', 'cloud_3pm',
                        'wind_speed_3pm', 'pressure_3pm']]
    
    classlabel = dataset['max_price_category']

    # K-fold Method with Classification Accuracy Formula
    k = 10
    kf = KFold(n_splits = k, shuffle = True, random_state = 1)
    classification_accuracy = []
    
    # Implementation of K Neighbors Classifier with K-fold Method
    for train_index, test_index in kf.split(features):
        
        # Split Training and Test sets
        features_train, feature_test = features.iloc[train_index, :], features.iloc[test_index, :]
        class_train, class_test = classlabel[train_index], classlabel[test_index]
        
        # Step 1: Instantiate 
        knn = neighbors.KNeighborsClassifier(n_neighbors = 5)
        
        # Step 2: Fit
        knn.fit(features_train, class_train)
        
        # Step 3: Predict
        predictions = knn.predict(feature_test)
        
        # Step 4: Evaluate
        my_score=accuracy_score(class_test, predictions)
        classification_accuracy.append(my_score)
    print(f'The AVG Knn Accuracy Score is {sum(classification_accuracy)/k}')


if __name__ == "__main__":
    main()
