
import pandas as pd
import time
from matplotlib.pylab import f
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import concurrent.futures
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed



def create_model(X_train, features, model=LinearRegression()):
    numeric_features = []
    categorical_features = []
    for feature in features:
        if X_train[feature].dtypes in ['int64', 'float64']:
            numeric_features.append(feature)
        else:
            categorical_features.append(feature)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])


def compute_score(submission):
    score = []
    submission_score = submission.to_numpy()
    for elt in submission_score:
        score.append((np.log(elt[0]) - np.log(elt[1]))**2)
    return np.sqrt(np.mean(score))


def test_tree_params():
    
    selected_features = ['GarageArea', 'GrLivArea', 'MSSubClass', 'TotalBsmtSF', 'LotFrontage', 'TotRmsAbvGrd', 'LotShape']
    all_tree = {}
    counter = 0
    max_depth = 8
    min_samples_leaf = 16
    for min_sample_split in range(2, 1000):
        df_train = pd.read_csv('resources/data/train.csv')
        y_column = 'LotArea'
        
        X_train, X_test, y_train, y_test = train_test_split(df_train.drop(y_column, axis=1), df_train[y_column], train_size=0.8, test_size=0.2, shuffle=True, random_state=0)
        
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

        model = create_model(X_train=X_train, features=selected_features, model=DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_sample_split, min_samples_leaf=min_samples_leaf))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rss = np.sum((y_test - y_pred)**2)
        all_tree[rss] = (max_depth, min_sample_split, min_samples_leaf)
        counter += 1
        print(counter, " / ", (1000-2))
    
    keys = np.sort(list(all_tree.keys()))
    print(all_tree[keys[0]])

def test_tree_bagging_feature(selected_features=None):
    data_folder = 'resources/data/'
    target = 'LotArea'
    n_bagging = 300
    random_forest_activation = True
    trunc = 0.7
    tree_params = {
        'max_features': 'sqrt'
    }
    df = pd.read_csv(data_folder + 'train.csv')
    df = df.drop(target, axis=1)
    df = df.drop('LotFrontage', axis=1)
    best_score = 0
    df = pd.read_csv(data_folder + 'train.csv')
    df[df.select_dtypes(exclude='number').columns] = df.select_dtypes(exclude='number').astype('category')
    
    X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], train_size=trunc, test_size=1-trunc, shuffle=True, random_state=0)
    X_test = X_test[selected_features]
    df = pd.concat([X_train, y_train], axis=1)

    X_trains, y_trains = [], []
    for _ in range(n_bagging):
        sample = df.sample(n=len(df), replace=True)
        X_trains.append(sample.drop(target, axis=1)[selected_features])
        y_trains.append(sample[target])
    
    bagging_models = []
    for i in range(n_bagging):
        if random_forest_activation:
            if len(selected_features) < 2:
                random_number = 1
            else:
                random_number = np.random.randint(int(np.sqrt(len(selected_features))), len(selected_features))
            tree_params = {
                'max_features' : random_number
            }
            selected_features = np.array(selected_features)
            np.random.shuffle(selected_features)
            selected_features = selected_features.tolist()
        model = create_model(X_trains[i], selected_features, DecisionTreeRegressor(**tree_params))
        bagging_models.append(model)

    bagging_predictions = []

    for i in range(n_bagging):
        bagging_models[i].fit(X_trains[i], y_trains[i])
        bagging_predictions.append(bagging_models[i].predict(X_test))

    y_pred = np.mean(bagging_predictions, axis=0)
    submission = pd.DataFrame({
        target: y_test,
        'Pred'+target: y_pred,
    })
    submission = submission.sort_index()
    score = []
    submission_score = submission.to_numpy()
    for elt in submission_score:
        score.append((np.log(elt[0]) - np.log(elt[1]))**2)
    score = np.sqrt(np.mean(score))
    return score


def test_tree_bagging_features_multithread(thread_number):
    data_folder = 'resources/data/'
    target = 'LotArea'
    n_bagging = 250
    random_forest_activation = True
    trunc = 0.7
    tree_params = {
        'max_features': 'sqrt'
    }
    localRandom = np.random.RandomState(thread_number+1)
    df = pd.read_csv(data_folder + 'train.csv')
    df = df.drop(target, axis=1)
    df = df.drop('LotFrontage', axis=1)
    all_features = df.columns.to_list()
    all_features = np.array(all_features)
    localRandom.shuffle(all_features)
    all_features = all_features.tolist()
    best_features = ['LotFrontage']
    best_score = 0
    for n in range(-1, len(all_features)):
        selected_features = best_features.copy()
        if n != -1:
            target_feature = all_features[n]
            selected_features.append(target_feature)

        df = pd.read_csv(data_folder + 'train.csv')
        df[df.select_dtypes(exclude='number').columns] = df.select_dtypes(exclude='number').astype('category')
        
        X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target], train_size=trunc, test_size=1-trunc, shuffle=True, random_state=0)
        X_test = X_test[selected_features]
        df = pd.concat([X_train, y_train], axis=1)

        X_trains, y_trains = [], []
        for _ in range(n_bagging):
                sample = df.sample(n=len(df), replace=True)
                X_trains.append(sample.drop(target, axis=1)[selected_features])
                y_trains.append(sample[target])
        
        bagging_models = []
        for i in range(n_bagging):
            if random_forest_activation:
                if len(selected_features) < 2:
                    random_number = 1
                else:
                    random_number = localRandom.randint(int(np.sqrt(len(selected_features))), len(selected_features))
                tree_params = {
                    'max_features' : random_number
                }
                selected_features = np.array(selected_features)
                localRandom.shuffle(selected_features)
                selected_features = selected_features.tolist()
            model = create_model(X_trains[i], selected_features, DecisionTreeRegressor(**tree_params))
            bagging_models.append(model)

        bagging_predictions = []

        for i in range(n_bagging):
            bagging_models[i].fit(X_trains[i], y_trains[i])
            bagging_predictions.append(bagging_models[i].predict(X_test))

        y_pred = np.mean(bagging_predictions, axis=0)
        submission = pd.DataFrame({
            target: y_test,
            'Pred'+target: y_pred,
        })
        submission = submission.sort_index()
        score = []
        submission_score = submission.to_numpy()
        for elt in submission_score:
            score.append((np.log(elt[0]) - np.log(elt[1]))**2)
        score = np.sqrt(np.mean(score))
        if score < best_score or n == -1:
            best_score = score
            best_features = selected_features.copy()
        print(thread_number, " : ", n+1, best_features, best_score)
    return (best_features, best_score)


def multithread_tree_features():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(test_tree_bagging_features_multithread, i) for i in range(100)]
        concurrent.futures.wait(futures)
        results = [f.result() for f in futures]
    for result in results:
        print(result)

def read_result_txt():
    file = open("output/results.txt", "r")
    lines = file.readlines()
    for line in lines:
        print(line, ",")

def check_result():
    result= [(['Heating', 'OverallQual', 'BsmtQual', 'LowQualFinSF', 'GarageArea', 'GrLivArea', 'MSSubClass', 'BsmtExposure', 'GarageYrBlt', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'LotShape', 'Electrical', 'PoolQC', 'BsmtFullBath', 'TotalBsmtSF', 'Foundation', 'Alley', 'Condition1', 'PoolArea', 'LotFrontage', 'CentralAir', 'Exterior2nd'], 0.25251794401596345)
 ,
(['Fireplaces', 'Neighborhood', 'GarageYrBlt', 'Street', 'BsmtQual', 'FireplaceQu', 'LotFrontage', 'KitchenQual', 'EnclosedPorch', 'MSSubClass', 'SaleType', 'GarageQual', 'Foundation', 'Condition1', 'MSZoning', 'LandSlope', 'TotalBsmtSF', 'OpenPorchSF', 'GarageCars', 'BsmtExposure'], 0.25711480385043683)
 ,
(['YearBuilt', 'HouseStyle', 'LotFrontage', 'MSSubClass', 'LandSlope', 'MiscFeature', '3SsnPorch', 'GrLivArea', 'Neighborhood', 'KitchenAbvGr', 'Condition2', 'PoolArea', 'TotalBsmtSF', 'GarageQual'], 0.2508880767896102)
 ,
(['ScreenPorch', 'GarageType', 'PavedDrive', 'GarageYrBlt', 'HalfBath', 'BsmtUnfSF', '2ndFlrSF', 'LotShape', 'LandContour', 'Neighborhood', 'MiscVal', 'GarageQual', 'BldgType', 'BsmtExposure', 'YearBuilt', 'MSZoning', 'BedroomAbvGr', 'PoolArea', 'KitchenQual', '1stFlrSF', 'OverallQual', 'TotalBsmtSF', 'LandSlope', 'LotFrontage', 'GrLivArea'], 0.24478694264814918)
 ,
(['BsmtFinType2', 'YearRemodAdd', 'LotShape', 'MiscVal', 'BldgType', 'HouseStyle', 'LandSlope', 'MSZoning', 'BedroomAbvGr', 'KitchenAbvGr', 'LotFrontage', 'Alley', 'BsmtFinSF2', 'ExterCond', 'GrLivArea', 'GarageArea', 'Functional', 'Condition2', 'Electrical', 'BsmtUnfSF', '1stFlrSF'], 0.25260589776513986)
 ,
(['1stFlrSF', '2ndFlrSF', 'GarageArea', 'MSZoning', 'CentralAir', 'LotFrontage', 'FullBath', 'GarageCars', 'Electrical', 'Neighborhood', 'PoolArea', 'HouseStyle', 'Exterior1st', 'Functional', 'TotalBsmtSF', 'LotShape', 'BsmtExposure', 'LandContour', 'BsmtFinType2', 'MSSubClass', 'GarageYrBlt'], 0.24547965172465355)
 ,
(['BsmtExposure', '2ndFlrSF', 'YearBuilt', 'PavedDrive', 'Alley', '1stFlrSF', 'Neighborhood', 'FireplaceQu', 'MiscFeature', 'RoofStyle', 'BsmtUnfSF', 'Heating', 'YearRemodAdd', 'LotShape', 'BldgType', 'GarageArea', 'MSZoning', 'HouseStyle', 'GrLivArea', 'Exterior2nd', 'LotFrontage', 'BsmtHalfBath', 'CentralAir'], 0.2450918608203056)
 ,
(['LotFrontage', 'MSZoning', 'FullBath', 'YearBuilt', 'GarageArea', 'Heating', 'LotShape', 'Electrical', 'TotRmsAbvGrd', '1stFlrSF', 'FireplaceQu', 'BedroomAbvGr', 'GrLivArea', 'PoolQC', 'BldgType', 'ScreenPorch', 'BsmtCond', 'Neighborhood'], 0.25084077252670456)
 ,
(['ScreenPorch', 'HalfBath', 'BldgType', 'Fence', 'MiscVal', 'GrLivArea', 'LotFrontage', 'YearBuilt', 'LandContour', 'Heating', 'MiscFeature', 'BsmtFinType1', 'Neighborhood', 'MSZoning', 'BsmtUnfSF', 'Street', 'YearRemodAdd', 'FullBath', 'BedroomAbvGr', 'LandSlope', 'Electrical', '2ndFlrSF', 'LotShape'], 0.24856425835408144)
 ,
(['Alley', 'GarageCars', 'Exterior2nd', 'GarageType', 'Neighborhood', 'MSZoning', 'TotalBsmtSF', 'FullBath', 'BsmtQual', 'Functional', 'LotFrontage', 'Fireplaces', 'HalfBath', 'BedroomAbvGr', 'BsmtFinType1', 'BldgType', 'GrLivArea', 'GarageQual', 'LotShape', 'GarageArea', 'MiscVal', 'RoofMatl', 'CentralAir', 'RoofStyle'], 0.25305877078275524)
 ,
(['1stFlrSF', 'Neighborhood', 'GarageQual', 'RoofMatl', 'Exterior2nd', 'YearBuilt', 'MiscVal', 'GarageCond', 'FullBath', 'LotFrontage', '3SsnPorch', 'LandSlope', 'LotShape', 'BsmtHalfBath', 'BsmtFinType1', 'Condition2', 'MSSubClass', 'GarageFinish', 'Utilities', 'TotalBsmtSF', 'BsmtCond', 'OpenPorchSF', 'TotRmsAbvGrd'], 0.2578976540671804)
 ,
(['Electrical', 'LowQualFinSF', 'LandSlope', 'HalfBath', 'GarageYrBlt', '1stFlrSF', 'GrLivArea', 'LotShape', 'CentralAir', 'OverallQual', 'FullBath', 'Neighborhood', 'BsmtHalfBath', 'Exterior1st', 'MiscVal', 'LotFrontage', 'MSZoning', 'YearBuilt', '2ndFlrSF', 'BsmtFullBath', 'MiscFeature'], 0.2414308268859102)
 ,
(['LotFrontage', 'PoolArea', 'Heating', 'MSSubClass', 'BsmtExposure', '3SsnPorch', 'Alley', 'GarageYrBlt', 'EnclosedPorch', 'LotShape', 'GrLivArea', 'MSZoning', 'GarageArea', 'Street', 'HalfBath', 'LandSlope', 'Neighborhood', 'SaleType', 'TotRmsAbvGrd', 'OverallQual', 'PoolQC', 'BsmtQual', 'WoodDeckSF', 'GarageFinish'], 0.23932705268099136)
 ,
(['MSSubClass', 'BsmtFullBath', 'LandSlope', 'MSZoning', 'Neighborhood', 'RoofStyle', 'BedroomAbvGr', 'LotFrontage', 'GrLivArea', 'Street', 'TotalBsmtSF', 'GarageFinish', 'EnclosedPorch', 'Functional', 'OverallQual', 'Exterior2nd', 'YearBuilt', 'GarageCars'], 0.24869766887604908)
 ,
(['PavedDrive', 'Heating', 'LowQualFinSF', 'MiscFeature', 'LandSlope', 'Neighborhood', 'BsmtFinType1', 'LotShape', 'MSZoning', 'Fireplaces', 'YearRemodAdd', 'GarageArea', 'RoofMatl', 'LandContour', 'LotFrontage', 'GrLivArea', 'Exterior1st', 'Alley', 'YearBuilt', 'BsmtFinType2', 'RoofStyle'], 0.2471646364355822)
 ,
(['BldgType', 'LandSlope', 'BsmtFinType2', 'LotFrontage', 'BsmtExposure', 'LandContour', 'TotalBsmtSF', 'Fence', 'Utilities', 'Alley', 'WoodDeckSF', 'BsmtCond', 'GrLivArea', 'KitchenAbvGr', 'RoofMatl', 'Neighborhood', 'MSZoning', 'Foundation'], 0.25208340383182526)
 ,
(['ScreenPorch', 'EnclosedPorch', 'BsmtFinSF2', 'MSZoning', 'BsmtExposure', 'Functional', 'Neighborhood', 'MiscFeature', 'HalfBath', 'LotConfig', 'LotShape', 'GarageQual', 'GrLivArea', 'BsmtCond', 'YearBuilt', 'KitchenAbvGr', 'BldgType', 'PoolArea', 'LotFrontage', 'GarageCars'], 0.2420413305895351)
 ,
(['1stFlrSF', 'TotalBsmtSF', 'Alley', 'BsmtQual', 'GarageYrBlt', 'Fireplaces', 'Condition1', 'MSZoning', 'BedroomAbvGr', 'PoolArea', 'Neighborhood', 'ExterQual', 'Foundation', 'LotFrontage', 'MSSubClass', 'EnclosedPorch', 'FullBath', 'LotShape', 'Exterior2nd', '2ndFlrSF', 'OverallCond', 'Heating', 'GrLivArea', 'GarageArea', 'RoofStyle'], 0.24518567356805446)
 ,
(['MoSold', 'PavedDrive', 'LandContour', 'CentralAir', 'GarageArea', 'Neighborhood', 'LotShape', 'MiscFeature', 'OverallQual', 'Functional', 'Fireplaces', 'LandSlope', 'OpenPorchSF', 'PoolArea', 'LowQualFinSF', 'BsmtFinSF2', 'LotFrontage', 'GrLivArea', '1stFlrSF', 'MSZoning', 'MSSubClass', 'TotalBsmtSF', 'BedroomAbvGr', 'GarageFinish'], 0.24396300564396117)
 ,
(['TotalBsmtSF', 'ExterQual', 'MSSubClass', 'CentralAir', 'LowQualFinSF', 'HalfBath', 'BsmtFinSF1', 'Heating', '2ndFlrSF', 'MSZoning', '3SsnPorch', 'LotShape', 'TotRmsAbvGrd', 'LandContour', 'Neighborhood', 'Fireplaces', 'OverallQual', 'LandSlope', 'SaleType', 'LotFrontage', 'GarageFinish'], 0.2509904727879763)
 ,
(['Foundation', 'LotShape', 'LotFrontage', 'Electrical', 'TotRmsAbvGrd', '1stFlrSF', 'EnclosedPorch', 'Functional', 'Alley', 'MSSubClass', 'OverallQual', 'MSZoning', 'HalfBath', 'GrLivArea', 'BsmtFullBath', 'LandSlope', 'Condition1', 'BsmtCond', 'Neighborhood', 'YearBuilt', 'Street', 'RoofStyle', 'BsmtHalfBath'], 0.24424630277666434)
 ,
(['Functional', 'TotRmsAbvGrd', 'YearBuilt', 'GrLivArea', 'MSSubClass', 'Alley', 'LotShape', 'RoofMatl', 'WoodDeckSF', 'Electrical', 'FullBath', 'GarageArea', 'LandSlope', 'Neighborhood', 'Condition1', 'BsmtCond', 'Fireplaces', 'OverallCond', 'MSZoning', '3SsnPorch', 'KitchenQual', 'MiscFeature', 'LotFrontage'], 0.24798100724753458)
 ,
(['BsmtFinSF2', 'MSSubClass', 'ExterQual', 'YearRemodAdd', 'Heating', 'MSZoning', 'GarageYrBlt', 'Fireplaces', 'LandContour', 'GarageArea', 'EnclosedPorch', 'GarageType', 'TotalBsmtSF', 'ScreenPorch', 'YearBuilt', 'Neighborhood', 'LotFrontage', 'OverallQual', 'HalfBath', 'LandSlope', 'Condition2'], 0.2517238300247459)
 ,
(['1stFlrSF', 'TotRmsAbvGrd', 'LandSlope', 'Neighborhood', 'LotShape', 'HeatingQC', 'GarageCond', 'BsmtFinType2', 'MSZoning', 'LotFrontage', 'BedroomAbvGr', 'GarageCars', 'Foundation', 'PoolArea', 'ScreenPorch', 'Functional'], 0.24663768557757818)
 ,
(['BsmtFullBath', 'BsmtFinSF2', 'HalfBath', 'BsmtCond', 'ExterCond', 'RoofMatl', 'MSZoning', 'Fireplaces', '1stFlrSF', 'MSSubClass', '2ndFlrSF', 'GrLivArea', 'PoolQC', 'Condition2', 'YearRemodAdd', 'GarageArea', 'BsmtFinSF1', 'GarageFinish', 'Street', '3SsnPorch', 'LotFrontage', 'TotalBsmtSF', 'Neighborhood', 'GarageCond', 'KitchenAbvGr', 'LandSlope', 'CentralAir'], 0.25541838081997875)
 ,
(['BsmtFinSF1', 'LotFrontage', 'GarageArea', 'MSSubClass', 'BsmtCond', 'FireplaceQu', 'LandSlope', 'Neighborhood', 'LotShape', 'MSZoning', 'MiscFeature', 'BsmtFinSF2', 'Heating', 'GarageYrBlt', 'PoolQC', 'Alley', 'ScreenPorch', 'Exterior1st', 'OpenPorchSF', '1stFlrSF'], 0.2479393026779427)
 ,
(['OverallQual', 'Exterior1st', 'Street', 'TotalBsmtSF', 'LotFrontage', 'LotShape', 'Neighborhood', 'GarageCond', 'Fireplaces', 'Condition2', 'BsmtExposure', 'CentralAir', 'Heating', 'MSZoning', 'HalfBath', 'Utilities', 'BedroomAbvGr', 'LandSlope', 'EnclosedPorch', 'GarageArea', 'GrLivArea', 'MiscFeature', 'OpenPorchSF'], 0.24934189918062502)
 ,
(['MSZoning', 'TotalBsmtSF', 'GarageArea', 'MSSubClass', 'Heating', 'Electrical', 'BedroomAbvGr', 'HouseStyle', 'PoolArea', 'MiscVal', 'GrLivArea', 'RoofMatl', 'YearRemodAdd', 'GarageCars', 'ExterQual', 'Street', 'GarageType', 'YearBuilt', '2ndFlrSF', 'YrSold', 'Neighborhood', 'BsmtFinSF1', 'RoofStyle', 'LotFrontage', 'BsmtQual', 'TotRmsAbvGrd'], 0.2596869377453567)
 ,
(['GarageQual', '3SsnPorch', 'MSZoning', 'TotRmsAbvGrd', 'ScreenPorch', '2ndFlrSF', 'Exterior2nd', 'LotConfig', 'GarageYrBlt', 'BsmtExposure', 'Neighborhood', 'LotShape', 'HeatingQC', 'MSSubClass', 'HalfBath', 'Functional', 'LandSlope', 'LotFrontage'], 0.24483052455471221)
 ,
(['Neighborhood', 'YearBuilt', 'Electrical', 'GarageCars', 'Street', 'Condition2', 'OpenPorchSF', 'OverallCond', 'Foundation', 'Condition1', 'LotShape', 'LandSlope', 'LotFrontage', 'RoofStyle', '1stFlrSF', 'BsmtCond', 'MSSubClass', 'GarageCond', 'MSZoning', 'TotRmsAbvGrd'], 0.24144475434073828)
 ,
(['TotRmsAbvGrd', 'MSSubClass', '3SsnPorch', 'KitchenAbvGr', 'LotConfig', 'YearRemodAdd', 'LotFrontage', 'EnclosedPorch', 'Neighborhood', 'ScreenPorch', 'PavedDrive', 'BsmtExposure', 'MiscVal', 'HouseStyle', '1stFlrSF', 'GarageYrBlt'], 0.26062874152734256)
 ,
(['MSZoning', 'HalfBath', 'OpenPorchSF', 'LotShape', '1stFlrSF', 'BldgType', 'Exterior2nd', 'Neighborhood', 'MiscFeature', 'FullBath', 'GrLivArea', 'YearRemodAdd', 'LotFrontage', 'Functional', '3SsnPorch', 'Fireplaces', 'Electrical', 'LandSlope', 'GarageQual', 'GarageCars', 'SaleType', 'GarageType'], 0.24744624134430257)
 ,
(['GrLivArea', 'Fireplaces', 'LotShape', 'BsmtFullBath', 'Neighborhood', 'OpenPorchSF', 'MSSubClass', 'GarageQual', 'GarageYrBlt', '1stFlrSF', 'LotFrontage', 'MiscVal', 'OverallCond', 'Foundation', '2ndFlrSF', 'FullBath', 'LandContour', 'MiscFeature', 'FireplaceQu', 'HeatingQC', 'TotRmsAbvGrd', 'MasVnrArea', 'BsmtUnfSF', 'GarageArea', 'Street', 'Functional', 'MSZoning'], 0.2428237284238823)
 ,
(['BedroomAbvGr', 'BsmtFullBath', 'Alley', 'BsmtFinType1', 'MasVnrArea', 'FullBath', 'MiscVal', 'YearBuilt', 'Fireplaces', 'Neighborhood', 'MSZoning', 'OverallCond', 'LotConfig', 'Electrical', 'GrLivArea', 'LotFrontage', 'MSSubClass', 'LotShape', '2ndFlrSF'], 0.2518119262946783)
 ,
(['Neighborhood', 'BedroomAbvGr', 'LotFrontage', '1stFlrSF', 'GarageQual', 'BsmtExposure', 'FullBath', 'BsmtQual', 'LandSlope', 'CentralAir', 'EnclosedPorch', 'LowQualFinSF', 'PavedDrive', 'LotShape', 'OverallQual', 'GrLivArea', 'RoofMatl', 'MSZoning', 'YearBuilt', 'BsmtHalfBath'], 0.2409296497255111)
 ,
(['LandSlope', 'Foundation', 'HalfBath', 'Neighborhood', 'Exterior2nd', 'LotFrontage', 'GrLivArea', 'GarageCond', 'MSZoning', 'BsmtUnfSF', 'BldgType', 'BedroomAbvGr', 'MiscFeature', 'KitchenAbvGr', 'MoSold', '1stFlrSF', 'LotShape', 'TotalBsmtSF', 'LotConfig', 'Fireplaces', 'HeatingQC', 'Street', 'BsmtExposure', 'GarageArea', 'GarageYrBlt', 'BsmtQual', 'YearBuilt', 'OpenPorchSF'], 0.24878458792223454)
 ,
(['Condition2', 'LotFrontage', 'Heating', 'BsmtFullBath', 'HouseStyle', 'BsmtFinType2', 'HalfBath', 'GarageArea', 'GrLivArea', 'BsmtQual', 'PoolArea', 'MiscVal', 'TotalBsmtSF', 'MSSubClass', 'GarageQual', 'LotShape', 'YearBuilt', 'OpenPorchSF', 'LandContour', 'BsmtExposure', 'MSZoning', 'GarageYrBlt', 'Neighborhood', 'Functional'], 0.2397638931311112)
 ,
(['Foundation', 'MSZoning', 'CentralAir', '1stFlrSF', 'ScreenPorch', 'Condition2', 'YearBuilt', 'LandSlope', 'MiscFeature', 'Utilities', 'GarageCars', 'MSSubClass', 'Electrical', 'Neighborhood', 'FireplaceQu', 'PavedDrive', 'LotFrontage'], 0.24121833600639236)
 ,
(['EnclosedPorch', 'BedroomAbvGr', 'LotShape', 'MiscFeature', 'GarageArea', 'LotFrontage', 'Condition2', 'MSSubClass', 'OpenPorchSF', '1stFlrSF', 'OverallCond', 'Fence', 'BsmtFinSF2', 'BsmtQual', 'YearBuilt', 'BsmtUnfSF', 'LandSlope', 'PoolQC', 'TotRmsAbvGrd', 'TotalBsmtSF', 'MSZoning'], 0.24844230085934765)
 ,
(['PoolArea', 'TotalBsmtSF', 'CentralAir', 'LotShape', 'BsmtHalfBath', 'TotRmsAbvGrd', 'Street', 'Neighborhood', 'YearRemodAdd', '2ndFlrSF', 'BsmtUnfSF', 'MSZoning', 'Exterior1st', 'LowQualFinSF', 'RoofMatl', 'PavedDrive', 'BsmtExposure', 'BsmtCond', 'GrLivArea', 'Condition2', '1stFlrSF', 'ScreenPorch', 'FullBath', 'MSSubClass', 'OpenPorchSF', 'FireplaceQu', 'Utilities', 'BsmtFullBath', 'LotFrontage', 'BsmtFinType2', 'ExterQual'], 0.2505727532746066)
 ,
(['OverallQual', '2ndFlrSF', '1stFlrSF', 'YearBuilt', 'PavedDrive', 'LotFrontage', 'GrLivArea', 'Neighborhood', 'MSSubClass', 'BsmtFinSF2', 'LowQualFinSF', 'HalfBath', 'BsmtExposure', 'GarageFinish'], 0.2543070039971776)
 ,
(['BsmtHalfBath', 'MSZoning', 'Heating', 'Electrical', 'Functional', 'GarageYrBlt', 'YrSold', 'Neighborhood', 'LotShape', 'Condition1', 'YearBuilt', 'GarageCars', 'LandSlope', 'TotalBsmtSF', 'Condition2', 'MSSubClass', '2ndFlrSF', 'LotFrontage', 'LandContour', 'EnclosedPorch', 'Fireplaces', 'CentralAir', 'FullBath', '1stFlrSF', 'ExterQual'], 0.2389289612040674)
 ,
(['YrSold', 'GarageYrBlt', 'HouseStyle', 'Fireplaces', 'LandContour', 'ExterCond', 'LotShape', 'MSZoning', 'GarageType', 'LotFrontage', 'GrLivArea', 'BsmtFinSF2', 'Heating', 'LandSlope', 'MSSubClass', 'Street', 'Neighborhood', '2ndFlrSF', '1stFlrSF', 'ExterQual'], 0.2445923052267965)
 ,
(['GarageYrBlt', 'LowQualFinSF', 'Fireplaces', 'LotShape', 'Condition2', 'BsmtUnfSF', 'KitchenAbvGr', 'MSSubClass', '1stFlrSF', 'TotalBsmtSF', 'MiscFeature', 'ExterQual', 'ScreenPorch', 'LandContour', 'LotFrontage', 'OverallCond', 'LandSlope', 'BsmtFinType2', 'GarageCond', 'BldgType', 'FireplaceQu', 'BedroomAbvGr', 'MSZoning', 'PavedDrive', 'LotConfig', 'Neighborhood', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'GarageFinish', 'Utilities', 'BsmtFinSF1'], 0.2477315344982999)
 ,
(['BsmtExposure', 'Electrical', 'MasVnrArea', 'GarageType', 'RoofMatl', 'RoofStyle', 'LotFrontage', 'Exterior2nd', 'MSSubClass', 'GarageYrBlt', 'GrLivArea', 'GarageCars', 'LotShape', 'BsmtCond', 'Neighborhood', 'HeatingQC', 'Functional', 'Foundation', 'Utilities', 'BedroomAbvGr', 'Condition2', 'MSZoning'], 0.2430038374916578)
 ,
(['GarageYrBlt', 'TotalBsmtSF', 'LotShape', 'GrLivArea', 'MSSubClass', 'KitchenAbvGr', 'LandContour', 'LotFrontage', 'BsmtFullBath', '2ndFlrSF', 'MSZoning', 'Fireplaces', 'Neighborhood', 'Alley', 'Street', 'Condition1', 'ExterQual', '3SsnPorch'], 0.24534016571776712)
 ,
(['ScreenPorch', 'YearBuilt', 'EnclosedPorch', 'Neighborhood', 'GarageArea', 'GrLivArea', 'GarageCars', 'RoofStyle', 'LotShape', 'LotFrontage', 'PavedDrive', 'HalfBath', 'MSZoning', 'OpenPorchSF', 'WoodDeckSF', 'PoolArea', 'BsmtFinType1', '1stFlrSF', 'BedroomAbvGr', 'KitchenAbvGr', 'Heating', 'BldgType', 'Fireplaces', 'BsmtFinType2', 'LandSlope'], 0.24363233046398386)
 ,
(['Heating', 'EnclosedPorch', 'LowQualFinSF', 'Neighborhood', 'TotalBsmtSF', 'LandContour', 'BsmtFinSF2', '2ndFlrSF', 'GrLivArea', 'YearRemodAdd', 'FireplaceQu', 'PavedDrive', 'Utilities', 'LotFrontage', 'Condition1', 'LotShape', 'MSZoning', 'GarageYrBlt', 'Foundation', 'LandSlope'], 0.24678606873006836)
 ,
(['ExterQual', 'BsmtFullBath', 'CentralAir', 'LotFrontage', 'MiscFeature', 'RoofMatl', 'MSZoning', 'MSSubClass', 'Neighborhood', 'YearBuilt', 'GarageQual', 'BsmtExposure', 'HalfBath', 'TotRmsAbvGrd', 'Condition2', 'YearRemodAdd', 'EnclosedPorch', '2ndFlrSF', 'GrLivArea', 'LotShape', 'OverallQual', 'Exterior2nd'], 0.24439919623221498)
 ,
(['CentralAir', 'PavedDrive', 'BsmtFullBath', 'YearRemodAdd', 'LotFrontage', 'GarageYrBlt', 'BsmtHalfBath', '1stFlrSF', 'Fireplaces', 'OverallCond', 'HalfBath', 'MSZoning', 'LandSlope', 'YearBuilt', 'Alley', 'ExterCond', 'WoodDeckSF', 'GarageCond', 'Neighborhood'], 0.2527320820430113)
 ,
(['1stFlrSF', 'MiscVal', 'RoofMatl', 'OverallQual', 'BldgType', 'LotShape', 'Neighborhood', 'YearBuilt', 'KitchenAbvGr', 'LandSlope', 'TotRmsAbvGrd', 'ExterCond', 'Fireplaces', 'FullBath', 'YearRemodAdd', 'LotFrontage', 'GarageArea', 'MSZoning', 'BsmtExposure'], 0.2411472321124171)
 ,
(['YrSold', 'MSZoning', 'YearBuilt', 'PavedDrive', 'Neighborhood', 'Functional', 'GarageCond', 'GarageYrBlt', 'Street', 'LandSlope', 'LotShape', 'LotFrontage', 'GarageQual', '1stFlrSF', '3SsnPorch', 'BsmtFinSF2', 'MSSubClass', 'GarageArea', 'Utilities', 'GrLivArea', 'GarageType'], 0.23293955783434328)
 ,
(['Functional', 'Street', 'LandSlope', 'LotShape', '1stFlrSF', 'GarageArea', 'Fireplaces', 'HalfBath', 'Exterior2nd', 'LotConfig', 'LotFrontage', 'Neighborhood', 'HouseStyle', 'MSZoning', 'BsmtFinSF2', 'Alley', 'GrLivArea', 'YearRemodAdd', 'MSSubClass', 'PavedDrive', 'Exterior1st', 'Electrical'], 0.2451480058092592)
 ,
(['LotShape', 'GarageCars', 'TotalBsmtSF', 'LotFrontage', 'GarageQual', 'Exterior2nd', 'Functional', '1stFlrSF', 'BsmtCond', 'GarageYrBlt', 'GarageFinish', 'Utilities', 'GarageType', 'PoolArea', 'FireplaceQu', 'GrLivArea', 'MSSubClass', 'MSZoning', 'YearBuilt', 'BsmtExposure', 'Heating'], 0.2486348595313797)
 ,
(['LotFrontage', 'RoofMatl', 'Alley', 'Condition1', 'Functional', 'BsmtFullBath', 'Street', 'Utilities', '1stFlrSF', 'LotShape', 'GarageYrBlt', 'ExterCond', 'MSZoning', 'Electrical', 'MSSubClass', 'LandContour', 'Neighborhood', 'GarageArea'], 0.23961327428386703)
 ,
(['PoolQC', 'RoofStyle', 'TotRmsAbvGrd', 'LotFrontage', 'MSZoning', 'SaleType', 'LotShape', 'BsmtExposure', 'TotalBsmtSF', '2ndFlrSF', 'MiscVal', 'BsmtFinType1', 'YearRemodAdd', 'Electrical', 'Heating', 'MSSubClass', 'Neighborhood', 'GarageArea', 'ScreenPorch', '1stFlrSF', 'Fireplaces', 'Foundation', 'GarageCond', 'LowQualFinSF'], 0.24734487648908357)
 ,
(['YearBuilt', 'BsmtUnfSF', 'LotFrontage', 'Fireplaces', 'BldgType', 'LotShape', 'GarageArea', 'Alley', 'PavedDrive', 'LowQualFinSF', 'MSSubClass', 'GarageFinish', 'GrLivArea', 'WoodDeckSF', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'ScreenPorch', 'TotRmsAbvGrd', 'RoofMatl', '2ndFlrSF', 'Electrical'], 0.25364124527902426)
 ,
(['BsmtUnfSF', 'Fireplaces', 'LotShape', 'Exterior1st', '1stFlrSF', 'GarageYrBlt', 'MSZoning', 'GrLivArea', 'Neighborhood', 'BsmtFullBath', 'LotFrontage', 'GarageType', 'GarageCars', 'EnclosedPorch', 'PoolQC', 'YearRemodAdd', 'YearBuilt', 'ExterQual', 'LandContour', 'BsmtExposure', 'BldgType'], 0.24430443576646388)
 ,
(['RoofStyle', 'HalfBath', 'GarageCars', 'PavedDrive', 'Utilities', 'YearBuilt', 'Functional', 'Neighborhood', 'MiscVal', 'LotShape', 'RoofMatl', 'LandContour', 'MSZoning', 'CentralAir', 'BsmtCond', 'TotalBsmtSF', 'BsmtFinSF2', 'LotFrontage', '3SsnPorch', 'GrLivArea', 'FullBath', 'HeatingQC'], 0.2445325333243864)
 ,
(['BldgType', 'LotShape', 'Fireplaces', 'MSZoning', 'OverallQual', 'ExterCond', 'KitchenAbvGr', 'LotConfig', 'HouseStyle', 'BedroomAbvGr', 'LotFrontage', 'GarageArea', 'RoofMatl', 'YearBuilt', 'GarageCond', 'WoodDeckSF', 'LandContour', 'Condition2', 'Electrical', 'Street', 'TotalBsmtSF', 'Neighborhood'], 0.2540887585224108)
 ,
(['MiscFeature', 'LotFrontage', 'MSZoning', 'MiscVal', 'GarageQual', '1stFlrSF', 'PavedDrive', '3SsnPorch', 'ScreenPorch', 'OverallQual', 'BldgType', 'Neighborhood', 'GrLivArea', 'RoofMatl', 'BsmtCond', 'GarageYrBlt', 'LandSlope'], 0.24603327669452388)
 ,
(['Neighborhood', 'Utilities', 'BsmtExposure', 'Electrical', 'BedroomAbvGr', '1stFlrSF', 'GrLivArea', 'FullBath', 'CentralAir', 'BldgType', 'PoolArea', 'MSZoning', 'LandSlope', 'LotShape', 'MiscFeature', 'BsmtCond', 'Alley', 'Fence', 'HouseStyle', 'PavedDrive', 'YearBuilt', 'LotFrontage', '2ndFlrSF', 'KitchenAbvGr', 'RoofStyle', 'ExterQual', 'Foundation', 'Functional'], 0.24076959582961938)
 ,
(['HouseStyle', 'GarageCars', 'Functional', 'Street', 'PoolArea', 'Heating', 'BldgType', 'FullBath', 'Foundation', 'LotShape', 'MiscVal', 'GrLivArea', 'BsmtExposure', 'GarageQual', 'LandSlope', 'MSZoning', 'Neighborhood', 'YearBuilt', 'TotalBsmtSF', 'KitchenQual', 'OpenPorchSF', 'Electrical', 'LotFrontage', 'MiscFeature', 'HalfBath', 'BsmtCond', 'Alley'], 0.23993999946252392)
 ,
(['1stFlrSF', 'LandContour', 'Electrical', 'MiscVal', 'PavedDrive', 'ScreenPorch', 'MSZoning', 'GarageArea', 'LotShape', 'GrLivArea', 'RoofMatl', 'LandSlope', 'BsmtHalfBath', 'BsmtFinSF2', 'Exterior2nd', 'EnclosedPorch', 'Neighborhood', 'HouseStyle', 'BsmtFinType1', 'GarageYrBlt', 'GarageFinish', 'LotFrontage'], 0.24453945000516386)
 ,
(['Utilities', 'KitchenAbvGr', '1stFlrSF', 'HeatingQC', 'Condition1', 'Fireplaces', 'RoofMatl', 'HalfBath', 'BsmtCond', 'LotShape', 'Functional', 'Electrical', 'LotFrontage', 'BsmtHalfBath', 'TotalBsmtSF', 'BedroomAbvGr', 'Heating', 'GrLivArea', '3SsnPorch', 'MSZoning', 'FullBath', 'HouseStyle', 'Neighborhood', 'LandContour', 'BldgType', 'GarageArea', 'RoofStyle', 'Fence', 'Alley', 'Exterior2nd'], 0.247597479882996)
 ,
(['HeatingQC', 'EnclosedPorch', 'GarageType', 'ExterQual', 'Condition1', 'KitchenAbvGr', 'Functional', 'BsmtFinType2', 'Condition2', 'GarageYrBlt', 'MSZoning', '2ndFlrSF', 'LandContour', 'GrLivArea', '3SsnPorch', 'TotalBsmtSF', 'Alley', 'Fireplaces', 'BsmtHalfBath', 'MSSubClass', 'Neighborhood', 'BsmtFinSF2', 'LotShape', 'WoodDeckSF', 'YearRemodAdd', '1stFlrSF', 'LotFrontage', 'MiscVal'], 0.24704102214323861)
 ,
(['GarageArea', 'BedroomAbvGr', 'Fireplaces', 'KitchenAbvGr', 'BsmtFullBath', 'Utilities', 'LotFrontage', 'GarageQual', 'MSZoning', 'LandContour', '1stFlrSF', 'YearBuilt', 'LandSlope', 'Neighborhood', 'MSSubClass', 'BsmtUnfSF', 'Condition2', 'ExterQual', 'GarageFinish'], 0.25504443793893655)
 ,
(['ScreenPorch', 'Condition2', '1stFlrSF', 'OpenPorchSF', 'GarageArea', 'PoolQC', 'BedroomAbvGr', 'PoolArea', 'YearRemodAdd', 'Heating', 'Alley', 'BsmtFullBath', 'OverallCond', 'Fireplaces', 'MSZoning', 'GarageFinish', 'MSSubClass', 'BsmtFinSF1', 'TotalBsmtSF', 'Neighborhood', 'LotFrontage', 'LandContour', 'Exterior1st', 'LowQualFinSF', 'BsmtFinSF2', 'LotShape', 'LandSlope'], 0.2488375455200638)
 ,
(['TotRmsAbvGrd', 'Condition1', 'MSZoning', 'Heating', 'WoodDeckSF', 'RoofStyle', 'Foundation', 'GarageArea', 'BsmtHalfBath', 'LotShape', 'OverallCond', 'BedroomAbvGr', 'LotFrontage', 'LandSlope', '1stFlrSF', 'Functional', 'EnclosedPorch', 'MSSubClass', 'KitchenQual'], 0.24959826336092342)
 ,
(['MSZoning', 'CentralAir', '2ndFlrSF', 'ExterQual', 'GarageYrBlt', 'HeatingQC', 'LotFrontage', 'Neighborhood', '3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'LotShape', '1stFlrSF', 'LandSlope', 'MiscVal', 'PoolArea', 'BedroomAbvGr'], 0.24254974907531937)
 ,
(['BsmtFinType1', 'GarageYrBlt', 'Heating', 'GarageCars', 'MSZoning', 'LotShape', 'LandContour', 'HalfBath', 'CentralAir', 'BldgType', 'RoofStyle', 'Neighborhood', 'PoolQC', 'GrLivArea', 'OpenPorchSF', 'TotalBsmtSF', 'Fireplaces', 'Exterior2nd', 'Foundation', 'LotFrontage', 'BsmtFullBath', 'BsmtFinSF2', 'Alley', '3SsnPorch', 'BsmtCond'], 0.2464957104306016)
 ,
(['TotalBsmtSF', 'LotFrontage', 'YearBuilt', 'GarageArea', 'Heating', 'Neighborhood', 'LandContour', 'OpenPorchSF', 'Condition2', '2ndFlrSF', 'LotShape', 'MSZoning', 'GarageCars', 'ScreenPorch'], 0.24130719943696483)
 ,
(['Neighborhood', 'Street', 'OverallCond', 'LandSlope', 'MSSubClass', 'BedroomAbvGr', 'BsmtUnfSF', 'LotFrontage', 'MSZoning', 'OpenPorchSF', 'ExterQual', 'PoolQC', 'Condition2', 'BsmtHalfBath', 'GarageCond', 'BsmtExposure', 'TotalBsmtSF', 'LandContour', '1stFlrSF', 'Alley', 'LotShape', 'GarageYrBlt'], 0.24338005438686233)
 ,
(['LotShape', 'MSSubClass', 'RoofMatl', 'Foundation', 'LotFrontage', 'Functional', 'BsmtUnfSF', 'SaleCondition', 'BsmtFinSF2', 'WoodDeckSF', 'PoolArea', 'Neighborhood', 'BldgType', 'LandSlope', 'RoofStyle', 'GrLivArea', 'BsmtFullBath', 'LandContour', 'KitchenAbvGr', 'CentralAir', 'GarageYrBlt', 'YearBuilt'], 0.2550089880677063)
 ,
(['BldgType', 'BsmtHalfBath', '1stFlrSF', 'GarageArea', 'EnclosedPorch', 'RoofStyle', 'BsmtFullBath', 'MSZoning', 'LotShape', 'LotFrontage', 'WoodDeckSF', 'GarageFinish', 'GarageYrBlt', 'LandContour', 'Fireplaces', 'Neighborhood', 'PoolQC', 'Condition1', 'TotalBsmtSF', 'MSSubClass', 'TotRmsAbvGrd', 'PavedDrive', 'Exterior2nd', 'Electrical', 'LotConfig', 'Alley'], 0.24952581573225982)
 ,
(['BsmtFullBath', 'BldgType', 'YearBuilt', 'BedroomAbvGr', 'KitchenAbvGr', 'BsmtFinType2', 'BsmtUnfSF', '2ndFlrSF', 'OverallQual', 'RoofMatl', 'LandSlope', 'Alley', 'HalfBath', 'LotFrontage', 'Street', 'RoofStyle', 'ScreenPorch', 'BsmtHalfBath', 'TotalBsmtSF', 'Electrical', 'BsmtCond', 'LotShape', 'BsmtExposure', 'GarageYrBlt', 'Neighborhood', 'GarageCond', 'GrLivArea'], 0.2515850624530317)
 ,
(['Fireplaces', 'MiscFeature', 'YearBuilt', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF', 'LandContour', 'HouseStyle', 'PoolArea', 'LandSlope', 'Neighborhood', 'GrLivArea', 'PoolQC', 'LotShape', 'EnclosedPorch', 'BedroomAbvGr', 'GarageQual', 'LotFrontage', 'MSZoning', 'BsmtFullBath', 'GarageYrBlt', 'Utilities', 'LowQualFinSF', 'BsmtQual', 'KitchenAbvGr', 'OpenPorchSF'], 0.24542882299728225)
 ,
(['MiscVal', 'ScreenPorch', 'BsmtUnfSF', 'MSSubClass', 'BsmtHalfBath', 'Foundation', 'KitchenAbvGr', 'WoodDeckSF', 'LotFrontage', 'YearBuilt', 'Condition1', 'RoofMatl', 'LandSlope', 'Condition2', '2ndFlrSF', 'Functional', 'KitchenQual', 'RoofStyle', 'Neighborhood', 'MSZoning'], 0.2541174521021608)
 ,
(['Neighborhood', 'TotRmsAbvGrd', 'BedroomAbvGr', 'MiscFeature', 'Condition1', 'HalfBath', 'KitchenAbvGr', 'Functional', '1stFlrSF', 'RoofMatl', 'GarageType', 'Exterior2nd', 'LotShape', 'ScreenPorch', 'MSSubClass', 'OpenPorchSF', 'ExterQual', 'Electrical', 'GarageArea', 'PoolQC', 'Exterior1st', 'GarageCars', 'BsmtExposure', 'LandContour', 'BsmtFinSF2', 'LandSlope', 'RoofStyle', 'LotFrontage', 'OverallQual', 'MSZoning'], 0.2471205201883809)
 ,
(['BsmtFinSF2', 'OverallCond', 'Fireplaces', 'SaleType', 'BsmtUnfSF', 'Neighborhood', 'SaleCondition', 'Alley', '1stFlrSF', 'MSZoning', 'LotFrontage', 'GarageYrBlt', 'EnclosedPorch', 'CentralAir', 'BsmtExposure', 'OpenPorchSF', 'LotShape', 'BldgType', 'BsmtFinType1', 'Heating', 'GrLivArea', 'HouseStyle', 'YrSold', 'ScreenPorch', 'Foundation', 'RoofStyle', 'RoofMatl', 'GarageArea'], 0.2444033803710867)
 ,
(['Foundation', 'Neighborhood', 'PoolQC', 'ExterCond', 'OverallCond', 'ExterQual', 'MiscVal', 'MSSubClass', 'GarageType', '3SsnPorch', 'Condition1', '1stFlrSF', 'Fence', 'Exterior2nd', 'LandContour', 'GarageCars', 'LotShape', 'Fireplaces', 'TotRmsAbvGrd', 'MSZoning', 'Exterior1st', 'YearRemodAdd', 'GrLivArea', 'LotFrontage', 'LandSlope', 'FireplaceQu', 'TotalBsmtSF', 'OpenPorchSF'], 0.24922380148912174)
 ,
(['MasVnrArea', 'Electrical', 'TotalBsmtSF', 'ScreenPorch', 'LowQualFinSF', 'BldgType', '1stFlrSF', 'GarageArea', 'YearBuilt', 'CentralAir', 'GarageYrBlt', 'GarageCars', 'LotFrontage', 'LotShape', 'BsmtFinType2', 'RoofStyle', 'Neighborhood', 'FireplaceQu', 'FullBath', 'BsmtFullBath', 'HouseStyle', 'Fireplaces', 'LandContour', 'OpenPorchSF', 'ExterCond', 'MSZoning'], 0.24898408813987827)
 ,
(['BldgType', 'GrLivArea', 'LandSlope', 'LotFrontage', 'BsmtExposure', 'Heating', 'BsmtFinType2', 'Foundation', 'PoolQC', 'GarageYrBlt', 'LotShape', 'Neighborhood', 'BsmtFinType1', 'MSZoning', 'BsmtFinSF1', 'KitchenQual', 'LowQualFinSF', 'TotalBsmtSF', 'GarageQual', 'LandContour', 'MiscFeature', 'PavedDrive', 'BsmtUnfSF', 'BsmtFullBath', 'EnclosedPorch', 'TotRmsAbvGrd'], 0.25005328716120667)
 ,
(['MiscVal', 'Neighborhood', 'MSZoning', 'LotFrontage', '1stFlrSF', 'GrLivArea', 'YearRemodAdd', 'GarageCond', 'Utilities', 'Fireplaces', 'PavedDrive', 'LotShape', 'LandSlope', 'YearBuilt', 'Electrical'], 0.23297180394461037)
 ,
(['TotalBsmtSF', 'FullBath', 'LotShape', 'Exterior2nd', 'GarageArea', 'GrLivArea', 'MSSubClass', 'Neighborhood', 'KitchenAbvGr', 'LotFrontage', 'RoofStyle', 'OpenPorchSF', 'Heating', 'ExterCond', 'YearBuilt', 'Fireplaces', 'GarageCond', 'MSZoning', 'LandContour', 'EnclosedPorch'], 0.2403777827516891)
 ,
(['TotalBsmtSF', 'MSZoning', 'Street', '3SsnPorch', 'LotFrontage', 'EnclosedPorch', 'MiscVal', 'OpenPorchSF', 'MiscFeature', 'Exterior2nd', 'Heating'], 0.2687309974389833)
 ,
(['MSSubClass', 'Neighborhood', 'Condition2', 'BsmtFinType2', 'PoolQC', 'LandSlope', 'GarageCars', 'MSZoning', 'Alley', 'TotalBsmtSF', 'TotRmsAbvGrd', 'Fireplaces', 'BsmtExposure', 'LotFrontage', 'KitchenAbvGr', 'GrLivArea', 'YearBuilt', 'PavedDrive', 'GarageType', '1stFlrSF', 'EnclosedPorch', 'ExterQual', 'LotShape'], 0.23741485516146507)
 ,
(['LowQualFinSF', 'Condition1', 'LandSlope', 'GarageYrBlt', 'MSZoning', 'GrLivArea', 'Electrical', 'BsmtFinSF1', 'Neighborhood', 'LotFrontage', '1stFlrSF', 'EnclosedPorch'], 0.2440461243624708)
 ,
(['Exterior1st', 'MSZoning', 'RoofStyle', 'BsmtUnfSF', 'PoolArea', 'Functional', 'LandSlope', 'LotFrontage', 'SaleCondition', 'GarageCars', 'LotShape', 'Neighborhood', 'GarageQual', 'OverallQual', 'TotalBsmtSF', 'Alley', 'Exterior2nd', 'EnclosedPorch', 'MSSubClass', 'WoodDeckSF', 'Fireplaces', 'GrLivArea', 'FullBath', 'Condition1', 'MiscFeature', 'BsmtExposure'], 0.24902596346408715)
 ,
(['BldgType', 'Street', 'BsmtCond', 'Exterior2nd', 'MSZoning', '2ndFlrSF', 'MiscFeature', 'CentralAir', 'EnclosedPorch', 'Neighborhood', 'FireplaceQu', 'TotalBsmtSF', 'GarageCars', 'GarageQual', 'Condition1', 'KitchenQual', 'LotFrontage', 'GrLivArea', 'LotShape', 'GarageArea'], 0.25850372164009966)
 ,
(['BsmtFinSF2', 'MSZoning', 'YearBuilt', 'Condition2', 'PoolArea', 'Neighborhood', 'BsmtFinType2', 'LotShape', 'GrLivArea', 'GarageYrBlt', 'KitchenAbvGr', 'RoofMatl', 'LandSlope', 'HouseStyle', 'LotFrontage', 'LandContour', 'Fireplaces', 'FullBath', 'HalfBath', 'BsmtFinSF1', 'ExterQual', 'TotalBsmtSF', 'ExterCond', 'PavedDrive', '2ndFlrSF'], 0.24218385814782079)
 ,
(['FullBath', 'OverallQual', 'Neighborhood', 'LandSlope', 'MSZoning', 'GarageYrBlt', 'Heating', 'GarageArea', 'LandContour', 'LotShape', 'CentralAir', 'MiscVal', 'EnclosedPorch', 'OpenPorchSF', '1stFlrSF', '3SsnPorch', 'GarageFinish', 'LotFrontage', 'RoofMatl'], 0.23557273016811334)
 ,
(['MSZoning', 'TotalBsmtSF', '1stFlrSF', 'BsmtFinType2', 'Fireplaces', 'ScreenPorch', 'HouseStyle', 'MSSubClass', 'Exterior1st', 'Alley', 'LotFrontage', 'OpenPorchSF', 'EnclosedPorch'], 0.263912084828834)
 ,
(['Electrical', 'Fireplaces', 'BsmtCond', 'TotalBsmtSF', 'SaleType', '2ndFlrSF', 'HeatingQC', 'LotFrontage', 'LotShape', 'BldgType', 'FullBath', 'LandContour', 'RoofStyle', 'MSZoning', 'BsmtHalfBath', 'GarageYrBlt', 'KitchenAbvGr', 'Utilities', 'PoolArea', 'Neighborhood', 'Functional', 'GrLivArea', 'Exterior2nd', 'BsmtFullBath'], 0.24369654228816875)
 ,
(['LotFrontage', 'PavedDrive', '1stFlrSF', 'BsmtHalfBath', 'GarageCars', 'Neighborhood', 'LandSlope', 'LowQualFinSF', 'KitchenAbvGr', 'EnclosedPorch', 'GarageQual', 'MSSubClass', 'TotalBsmtSF', 'YearBuilt', 'BsmtExposure', 'Condition1', 'MSZoning', 'LotShape'], 0.2376352295881898)
 ,
(['MSSubClass', 'YearRemodAdd', 'HeatingQC', 'Functional', 'Condition2', 'Neighborhood', 'LowQualFinSF', 'Street', 'LotFrontage', 'BsmtExposure', 'ExterCond', 'YearBuilt', 'MSZoning', '2ndFlrSF', '1stFlrSF', 'TotalBsmtSF'], 0.2503956855519277)
 ,
(['2ndFlrSF', 'ScreenPorch', 'RoofStyle', 'Neighborhood', 'MSSubClass', 'YearRemodAdd', 'LandSlope', 'BsmtCond', 'GarageArea', 'BldgType', 'Exterior2nd', 'BedroomAbvGr', 'LotFrontage', 'Street', 'YearBuilt', 'GarageCond', 'BsmtUnfSF', 'RoofMatl', 'GrLivArea', 'MSZoning'], 0.25381804388947016)
 ,
(['TotalBsmtSF', 'MSZoning', 'LotFrontage', 'MiscFeature', '2ndFlrSF', 'GarageCars', 'Condition2', 'GarageYrBlt', 'HouseStyle', 'Heating', 'GarageCond', 'GrLivArea', 'BsmtCond', 'KitchenAbvGr', 'GarageArea', 'Condition1', 'GarageType', 'OverallCond', 'Exterior1st', 'LandSlope', 'Neighborhood', 'LandContour', 'LotShape', 'BsmtFinSF1', 'MSSubClass'], 0.24390586353297294)
 ,
(['Fireplaces', 'BsmtCond', 'Condition1', 'MiscVal', 'RoofStyle', 'GarageQual', 'ScreenPorch', 'Heating', 'LandContour', 'MasVnrArea', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'Foundation', 'LotFrontage', 'RoofMatl', 'FireplaceQu', 'KitchenAbvGr', 'GarageCars', 'MSZoning', 'YearBuilt', 'LotShape', 'MSSubClass', 'LowQualFinSF', 'BsmtHalfBath', 'OpenPorchSF', 'BsmtExposure', 'Neighborhood', 'OverallQual', '3SsnPorch'], 0.24913518448890093)
 ,
(['TotalBsmtSF', 'GarageCars', 'LotFrontage', 'BsmtExposure', 'Alley', 'BldgType', 'OpenPorchSF', 'MasVnrArea', 'PavedDrive', 'OverallCond', 'LotShape', 'BsmtUnfSF', 'Street', 'BsmtFinSF2', 'Electrical', 'SaleType', 'LandContour', 'Foundation', 'GarageQual', 'Neighborhood', 'FireplaceQu', 'Exterior1st', 'BsmtFullBath', 'YearBuilt', '2ndFlrSF', 'GarageYrBlt', 'GrLivArea', 'MSZoning', '1stFlrSF'], 0.25125870596195626) ]
    dico = {}
    for tuple in result:
        dico[tuple[1]] = tuple[0]
    keys = list(dico.keys())
    keys.sort()
    for key in keys:
        if key <0.24:
            print(dico[key], ",")

def check_features():
    list_selected_features = [['YrSold', 'MSZoning', 'YearBuilt', 'PavedDrive', 'Neighborhood', 'Functional', 'GarageCond', 'GarageYrBlt', 'Street', 'LandSlope', 'LotShape', 'LotFrontage', 'GarageQual', '1stFlrSF', '3SsnPorch', 'BsmtFinSF2', 'MSSubClass', 'GarageArea', 'Utilities', 'GrLivArea', 'GarageType'] ,
['MiscVal', 'Neighborhood', 'MSZoning', 'LotFrontage', '1stFlrSF', 'GrLivArea', 'YearRemodAdd', 'GarageCond', 'Utilities', 'Fireplaces', 'PavedDrive', 'LotShape', 'LandSlope', 'YearBuilt', 'Electrical'] ,
['FullBath', 'OverallQual', 'Neighborhood', 'LandSlope', 'MSZoning', 'GarageYrBlt', 'Heating', 'GarageArea', 'LandContour', 'LotShape', 'CentralAir', 'MiscVal', 'EnclosedPorch', 'OpenPorchSF', '1stFlrSF', '3SsnPorch', 'GarageFinish', 'LotFrontage', 'RoofMatl'] ,
['MSSubClass', 'Neighborhood', 'Condition2', 'BsmtFinType2', 'PoolQC', 'LandSlope', 'GarageCars', 'MSZoning', 'Alley', 'TotalBsmtSF', 'TotRmsAbvGrd', 'Fireplaces', 'BsmtExposure', 'LotFrontage', 'KitchenAbvGr', 'GrLivArea', 'YearBuilt', 'PavedDrive', 'GarageType', '1stFlrSF', 'EnclosedPorch', 'ExterQual', 'LotShape'] ,
['LotFrontage', 'PavedDrive', '1stFlrSF', 'BsmtHalfBath', 'GarageCars', 'Neighborhood', 'LandSlope', 'LowQualFinSF', 'KitchenAbvGr', 'EnclosedPorch', 'GarageQual', 'MSSubClass', 'TotalBsmtSF', 'YearBuilt', 'BsmtExposure', 'Condition1', 'MSZoning', 'LotShape'] ,
['BsmtHalfBath', 'MSZoning', 'Heating', 'Electrical', 'Functional', 'GarageYrBlt', 'YrSold', 'Neighborhood', 'LotShape', 'Condition1', 'YearBuilt', 'GarageCars', 'LandSlope', 'TotalBsmtSF', 'Condition2', 'MSSubClass', '2ndFlrSF', 'LotFrontage', 'LandContour', 'EnclosedPorch', 'Fireplaces', 'CentralAir', 'FullBath', '1stFlrSF', 'ExterQual'] ,
['LotFrontage', 'PoolArea', 'Heating', 'MSSubClass', 'BsmtExposure', '3SsnPorch', 'Alley', 'GarageYrBlt', 'EnclosedPorch', 'LotShape', 'GrLivArea', 'MSZoning', 'GarageArea', 'Street', 'HalfBath', 'LandSlope', 'Neighborhood', 'SaleType', 'TotRmsAbvGrd', 'OverallQual', 'PoolQC', 'BsmtQual', 'WoodDeckSF', 'GarageFinish'] ,
['LotFrontage', 'RoofMatl', 'Alley', 'Condition1', 'Functional', 'BsmtFullBath', 'Street', 'Utilities', '1stFlrSF', 'LotShape', 'GarageYrBlt', 'ExterCond', 'MSZoning', 'Electrical', 'MSSubClass', 'LandContour', 'Neighborhood', 'GarageArea'] ,
['Condition2', 'LotFrontage', 'Heating', 'BsmtFullBath', 'HouseStyle', 'BsmtFinType2', 'HalfBath', 'GarageArea', 'GrLivArea', 'BsmtQual', 'PoolArea', 'MiscVal', 'TotalBsmtSF', 'MSSubClass', 'GarageQual', 'LotShape', 'YearBuilt', 'OpenPorchSF', 'LandContour', 'BsmtExposure', 'MSZoning', 'GarageYrBlt', 'Neighborhood', 'Functional'] ,
['HouseStyle', 'GarageCars', 'Functional', 'Street', 'PoolArea', 'Heating', 'BldgType', 'FullBath', 'Foundation', 'LotShape', 'MiscVal', 'GrLivArea', 'BsmtExposure', 'GarageQual', 'LandSlope', 'MSZoning', 'Neighborhood', 'YearBuilt', 'TotalBsmtSF', 'KitchenQual', 'OpenPorchSF', 'Electrical', 'LotFrontage', 'MiscFeature', 'HalfBath', 'BsmtCond', 'Alley']]
    
    score_per_selected_features = {}
    n=0
    for selected_features in list_selected_features:
        list_score = []
        for i in range(10):
            n += 1
            score = test_tree_bagging_feature(selected_features)
            list_score.append(score)
            print(f"{n} : {score}")
        score = np.mean(list_score)
        min_score = min(list_score)
        max_score = max(list_score)
        print(n, "step", score, min_score, max_score)
        score_per_selected_features[score] = (selected_features, score, min_score, max_score)
    keys = list(score_per_selected_features.keys())
    keys.sort()
    for key in keys:
        print(score_per_selected_features[key])

def create_model_feature(X_train, features, model=LinearRegression()):
    numeric_features = []
    categorical_features = []
    for feature in features:
        if X_train[feature].dtypes in ['int64', 'float64']:
            numeric_features.append(feature)
        else:
            categorical_features.append(feature)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        #target encoding
        ('ordinal', OrdinalEncoder())
        
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

def test_features_with_rf_tree(X_train, y_train, i):
    tree_random_forest_params = {
    'max_features': None,
    'max_depth': 30000000,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'criterion': 'absolute_error',
    'random_state': None,
    'n_estimators': 250,
}
    model = create_model_feature(X_train, X_train.columns, model=RandomForestRegressor(**tree_random_forest_params))
    model.fit(X_train, y_train)
    feature_importances = model.named_steps['model'].feature_importances_
    print(i)
    return feature_importances

def test1():
    df = pd.read_csv("resources/data/train.csv")
    X_train = df.drop(columns=["LotArea"])
    y_train = df["LotArea"]
    features_importance = []
    features_importance = Parallel(n_jobs=12)(delayed(test_features_with_rf_tree)(X_train, y_train, i) for i in range(500))
    importance = []
    for i in range(len(features_importance[0])):
        importance.append(0)
    for feature_importances in features_importance:
        for i in range(len(feature_importances)):
            importance[i] += feature_importances[i]
    for i in range(len(importance)):
        importance[i] = importance[i]/500
    print(importance)
    #importance = [0.007403803565433918, 0.09110052305870887, 0.014883894075927707, 0.004515320437581002, 0.011882267323720476, 0.015464192992072115, 0.006432360387420164, 0.019681554146906904, 0.012500233451181246, 0.010573054762885881, 0.015578078262966392, 0.07288239146322364, 0.003798780086202962, 0.0001612672609095274, 0.060552465586924044, 0.005386132844171138, 0.0005743234089777623, 0.016151909332642007, 0.0017922559227573846, 0.007439068776486565, 9.596728888551513e-05, 0.01841581248408139, 0.006917794318772814, 0.02407987391371366, 0.001607439625767609, 0.01577430342408301, 0.01857402455455625, 0.00840802131392414, 0.004573204317756743, 0.0006016047343756035, 0.0019939537081934368, 0.0007947226112441143, 0.00025717300280187627, 0.009155245837160908, 0.009285398307187182, 0.01814394031877057, 0.003416265400539913, 0.0003225378362165558, 0.012436875997842036, 0.003921322166829808, 0.0003151795320980706, 0.005620213046507053, 0.2113520981118186, 0.02162475858982621, 0.0024315541007667763, 6.381630880060748e-05, 0.07790595631198521, 0.001982472125984123, 0.004511454954471939, 0.02726325806815238, 0.01441248556291255, 0.017953392161690937, 0.0022154505962294333, 0.0012046726344878652, 0.0018233144400710954, 0.0023671696765350495, 0.0020336636617961116, 0.0008707529451664953, 0.001732065631152149, 0.003590306968454279, 0.0046446527671454305, 0.004206849714997077, 0.015532921256999063, 0.00047353800545869927, 0.0004862718867232451, 0.003017072016762937, 0.002871968673606045, 0.0053828147191479145, 0.0044577862329092, 0.004815347860769192, 0.0008268550090366471, 0.0006573481217810826, 0.009459849760827442, 0.00016273031389314925, 0.0009766515146030312, 0.00016562096450064854, 0.0008099090960971762, 0.0022484183460322674]
    features_imp = {}
    features = X_train.columns.to_list()
    for i in range(len(importance)):
        features_imp[importance[i]] = features[i]
    keys = list(features_imp.keys())
    keys.sort(reverse=True)
   
    features_lst = []
    for key in keys:
        features_lst.append(features_imp[key])
    print(features_lst)

if __name__ == "__main__":
    pass