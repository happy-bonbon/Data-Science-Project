import pandas as pd
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import plotly.express as px

def main():
    '''classification for Max Price Category VS Weather Best: 10-fold 8-NN'''
    
    # read csv file
    dataset = pd.read_csv('Data/combined_detail_cleaned.csv')

    # Selecting features
    features = dataset[['temperature_min', 'rainfall', 
                        'evaporation', 'sunshine', 'max_wind_speed']]
    
    classlabel = dataset['max_price_category']

    # K-fold Method (cross validation) with Classification Accuracy Formula
    k, my_neighbors = 10, 8
    kf = KFold(n_splits = k, shuffle = True, random_state = 88)

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
        knn = neighbors.KNeighborsClassifier(n_neighbors = my_neighbors)
        
        # Step 2: Fit
        knn.fit(features_train, class_train)
        
        # Step 3: Predict
        predictions = knn.predict(feature_test)
        
        # Step 4: Evaluate
        
        fig = px.scatter(
        feature_test, x=0, y=1,
        color=predictions, color_continuous_scale='RdBu',
        symbol=class_test, symbol_map={'0': 'square-dot', '1': 'circle-dot'},
        labels={'symbol': 'label', 'color': 'score of <br>first class'}
        )
        fig.update_traces(marker_size=12, marker_line_width=1.5)
        fig.update_layout(legend_orientation='h')
        fig.show()
    
        my_score=accuracy_score(class_test, predictions)
        classification_accuracy.append(my_score)
    print(f'The AVG {k}-Fold Accuracy Score with {my_neighbors}-neighbors is {sum(classification_accuracy)/k}')


if __name__ == "__main__":
    main()
