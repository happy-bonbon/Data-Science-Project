import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif


def main():
    '''classification for Max Price Category VS Weather'''
    
    # read csv file
    dataset = pd.read_csv('Data/combined_detail_cleaned.csv')

    # Selecting features
    features = dataset[[
                    'temperature_min',
                    'temperature_max',
                    'rainfall',
                    'evaporation',
                    'sunshine',
                    'max_wind_speed',
                    'temperature_9am',
                    'humidity_9am',
                    'cloud_9am',
                    'wind_speed_9am',
                    'pressure_9am',
                    'temperature_3pm',
                    'humidity_3pm',
                    'cloud_3pm',
                    'wind_speed_3pm',
                    'pressure_3pm']]
    
    classlabel = dataset['max_price_category']

    # K-fold Method (cross validation) with Classification Accuracy Formula
    k, x_times = 20, 12
    kf = KFold(n_splits = k, shuffle = True, random_state = 88)
    classification_accuracy = []
    
    # Implementation of K-fold Method
    for train_index, test_index in kf.split(features):
        
        # Split Training and Test sets
        features_train, feature_test = features.iloc[train_index, :], features.iloc[test_index, :]
        class_train, class_test = classlabel[train_index], classlabel[test_index]
        
        # Mutual Information Implementation
        features_selector = SelectKBest(mutual_info_classif, k = 3)
        features_train = features_selector.fit_transform(features_train, class_train)
        feature_test = features_selector.transform(feature_test)
        print(features_selector.scores_)
    
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
    print(f'The AVG {k}-Fold Accuracy Score with Max depth of {x_times} is {sum(classification_accuracy)/k}')
    

if __name__ == "__main__":
    main()
