import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def main():
    '''classification for Max Price Category VS Weather O(n**3) 20-fold 12-Depths'''
    
    # read csv file
    dataset = pd.read_csv('Data/combined_detail_cleaned.csv')

    # Selecting features
    features = dataset[['temperature_min','rainfall', 
                        'sunshine', 'max_wind_speed']]
    
    classlabel = dataset['max_price_category']

    # parameter tuning with diferent folds
    for k in range(10,51,10):
        # K-fold Method (cross validation) with Classification Accuracy Formula
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
            print(f'The AVG {k}-Fold Accuracy Score with Max depth of {x_times} is {sum(classification_accuracy)/k}')
        

if __name__ == "__main__":
    main()
