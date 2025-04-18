#!/usr/bin/env python3

# Libs unittest
import unittest
from unittest.mock import patch
from unittest.mock import Mock

# Utils libs
import os
import shutil
import dill as pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, KBinsDiscretizer, Binarizer, PolynomialFeatures, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, _VectorizerMixin
from sklearn.feature_selection import SelectKBest, SelectorMixin
from sklearn.impute import SimpleImputer
from sklearn.exceptions import NotFittedError
from ynov import utils
from ynov.models_training import utils_models
from ynov.models_training.classifiers import model_rf_classifier, model_dense_classifier, model_xgboost_classifier
from ynov.models_training.regressors import model_rf_regressor, model_dense_regressor, model_xgboost_regressor
from ynov.preprocessing import preprocess

# Disable logging
import logging
logging.disable(logging.CRITICAL)


def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


class UtilsModelsTests(unittest.TestCase):
    '''Main class to test all functions in utils_models.py'''

    # On evite les prints de tqdm
    pd.Series.progress_apply = pd.Series.apply


    def setUp(self):
        '''SetUp fonction'''
        # On se place dans le bon répertoire
        # Change directory to script directory
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)


    def test01_normal_split(self):
        '''Test de la fonction ynov.models_training.utils_models.normal_split'''
        # Vals à tester
        input_test = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']})
        test_size = 0.2
        train, test = utils_models.normal_split(input_test, test_size=test_size)
        self.assertEqual(train.shape[0], input_test.shape[0] * (1 - test_size))
        self.assertEqual(test.shape[0], input_test.shape[0] * test_size)

        # Vérification des inputs
        with self.assertRaises(ValueError):
            utils_models.normal_split(input_test, test_size=1.2)
        with self.assertRaises(ValueError):
            utils_models.normal_split(input_test, test_size=-0.2)


    def test02_stratified_split(self):
        '''Test de la fonction ynov.models_training.utils_models.stratified_split'''
        # Vals à tester
        input_test = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'],
                                   'col2': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]})
        test_size = 0.5
        col = 'col2'
        train, test = utils_models.stratified_split(input_test, col, test_size=test_size)
        self.assertEqual(train.shape[0], (input_test.shape[0] - 1) * (1 - test_size))
        self.assertEqual(test.shape[0], (input_test.shape[0] - 1) * test_size)
        self.assertEqual(train[train[col] == 0].shape[0], input_test[input_test[col] == 0].shape[0] * (1 - test_size))
        self.assertEqual(test[test[col] == 0].shape[0], input_test[input_test[col] == 0].shape[0] * test_size)
        self.assertEqual(train[train[col] == 2].shape[0], 0)
        self.assertEqual(test[test[col] == 2].shape[0], 0)

        # Vérification du type du/des input(s)
        with self.assertRaises(TypeError):
            utils_models.stratified_split(input_test, [8])
        with self.assertRaises(ValueError):
            utils_models.stratified_split(input_test, col, test_size=1.2)
        with self.assertRaises(ValueError):
            utils_models.stratified_split(input_test, col, test_size=-0.2)


    def test03_remove_small_classes(self):
        '''Test de la fonction ynov.models_training.utils_models.remove_small_classes'''
        # Vals à tester
        input_test = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'],
                                   'col2': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]})
        test_size = 0.2
        col = 'col2'

        result_df = utils_models.remove_small_classes(input_test, col, min_rows=2)
        self.assertEqual(result_df[result_df[col] == 0].shape[0], input_test[input_test[col] == 0].shape[0])
        self.assertEqual(result_df[result_df[col] == 1].shape[0], input_test[input_test[col] == 1].shape[0])
        self.assertEqual(result_df[result_df[col] == 2].shape[0], 0)

        result_df = utils_models.remove_small_classes(input_test, col, min_rows=5)
        self.assertEqual(result_df[result_df[col] == 0].shape[0], 0)
        self.assertEqual(result_df[result_df[col] == 1].shape[0], input_test[input_test[col] == 1].shape[0])
        self.assertEqual(result_df[result_df[col] == 2].shape[0], 0)

        # Vérification du type du/des input(s)
        with self.assertRaises(TypeError):
            utils_models.remove_small_classes(input_test, [8])
        with self.assertRaises(ValueError):
            utils_models.remove_small_classes(input_test, col, min_rows=0)


    def test04_display_train_test_shape(self):
        '''Test de la fonction ynov.models_training.utils_models.display_train_test_shape'''
        # Vals à tester
        df = pd.DataFrame({'col1': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'],
                           'col2': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2]})

        # Fonctionnement nominal
        utils_models.display_train_test_shape(df, df)
        utils_models.display_train_test_shape(df, df, df_shape=10)


    def test05_preprocess_model_multilabel(self):
        '''Test de la fonction ynov.models_training.utils_models.preprocess_model_multilabel'''
        # Création d'un jeu de données
        df = pd.DataFrame({'x_col': ['test', 'toto', 'titi'], 'y_col': [(), ('x1', 'x2'), ('x3', 'x4', 'x1')]})
        df_expected = pd.DataFrame({'x_col': ['test', 'toto', 'titi'], 'y_col': [(), ('x1', 'x2'), ('x3', 'x4', 'x1')],
                                    'x1': [0, 1, 1], 'x2': [0, 1, 0], 'x3': [0, 0, 1], 'x4': [0, 0, 1]})
        subset_classes = ['x2', 'x4']
        df_subset_expected = pd.DataFrame({'x_col': ['test', 'toto', 'titi'], 'y_col': [(), ('x1', 'x2'), ('x3', 'x4', 'x1')],
                                           'x2': [0, 1, 0], 'x4': [0, 0, 1]})

        # Fonctionnement nominal
        df_mlb, classes = utils_models.preprocess_model_multilabel(df, 'y_col')
        self.assertEqual(sorted(classes), ['x1', 'x2', 'x3', 'x4'])
        pd.testing.assert_frame_equal(df_mlb, df_expected, check_dtype=False)

        # Fonctionnement argument classes
        df_mlb, classes = utils_models.preprocess_model_multilabel(df, 'y_col', classes=subset_classes)
        self.assertEqual(sorted(classes), sorted(subset_classes))
        pd.testing.assert_frame_equal(df_mlb, df_subset_expected, check_dtype=False)

        # Vérification du type du/des input(s)
        with self.assertRaises(TypeError):
            utils_models.preprocess_model_multilabel(df, [42])


    def test06_load_pipeline(self):
        '''Test de la fonction ynov.models_training.utils_models.load_pipeline'''

        # Creation fake pipeline
        pipeline_dir = os.path.join(utils.get_pipelines_path(), 'fake_pipeline_dir')
        pipeline_path = os.path.join(pipeline_dir, 'pipeline.pkl')
        preprocess_str = 'fake_pipeline'
        remove_dir(pipeline_dir)
        os.makedirs(pipeline_dir)
        fake_pipeline = ColumnTransformer([('fake_pipeline', FunctionTransformer(lambda x: x * 2), ['toto', 'titi'])])
        df = pd.DataFrame({'toto': [1, 2, 3], 'tata': [4, 5, 6], 'titi': [7, 8, 9]})
        fake_pipeline.fit(df) # On fit , même si pas vraiment besoin

        # Save pipeline
        pipeline_dict = {'preprocess_pipeline': fake_pipeline, 'preprocess_str': preprocess_str}
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline_dict, f)

        # Reload
        new_pipeline, new_preprocess_str = utils_models.load_pipeline(pipeline_dir='fake_pipeline_dir')
        # On fait qqs tests
        np.testing.assert_equal(new_pipeline.transform(df), fake_pipeline.transform(df))
        self.assertEqual(new_preprocess_str, preprocess_str)

        # Pareil, mais avec chemin
        new_pipeline, new_preprocess_str = utils_models.load_pipeline(pipeline_dir=pipeline_dir, is_path=True)
        # On fait qqs tests
        np.testing.assert_equal(new_pipeline.transform(df), fake_pipeline.transform(df))
        self.assertEqual(new_preprocess_str, preprocess_str)
        remove_dir(pipeline_dir)

        # On fait pareil avec pipeline_dir = None, i.e. backup no preprocess
        new_pipeline, new_preprocess_str = utils_models.load_pipeline(pipeline_dir=None)
        # Attention, on doit fit la pipeline
        new_pipeline.fit(df)
        # On fait qqs tests
        np.testing.assert_equal(new_pipeline.transform(df), np.array(df))
        self.assertEqual(new_preprocess_str, 'no_preprocess')

        # Vérification raises
        with self.assertRaises(FileNotFoundError):
            utils_models.load_pipeline(pipeline_dir='tototo')
        with self.assertRaises(FileNotFoundError):
            utils_models.load_pipeline(pipeline_dir='./tototo', is_path=True)
        with self.assertRaises(NotFittedError):
            pipeline, _ = utils_models.load_pipeline(pipeline_dir=None)
            pipeline.transform(df)


    def test07_load_model(self):
        '''Test de la fonction ynov.models_training.utils_models.load_model'''

        # Données pour apprentissage
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, 2, -6, 3], 'col_2': [2, -1, -8, 3, 12, 2]})
        x_test = pd.DataFrame({'col_1': [-2, 8, 0, -1, -2, -9], 'col_2': [5, -1, 3, 6, -5, 2]})
        y_train_classification = pd.Series([0, 0, 0, 1, 1, 1])
        y_train_regression = pd.Series([-3, -2, -8, 5, 6, 5])
        model_dir = os.path.join(utils.get_models_path(), 'test_model')
        model_name = 'test_model_name'
        batch_size = 8
        epochs = 3
        patience = 5
        early_stopping_rounds = 3

        ####################################################

        # Tests sur un fake model - RF classifier
        remove_dir(model_dir)
        model = model_rf_classifier.ModelRFClassifier(model_dir=model_dir, model_name=model_name)
        model.fit(x_train, y_train_classification)
        model.save()

        # Reload
        new_model, new_config = utils_models.load_model(model_dir='test_model')
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))

        # Pareil, mais avec chemin
        new_model, new_config = utils_models.load_model(model_dir=model_dir, is_path=True)
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))
        remove_dir(model_dir)

        ####################################################

        # Tests sur un fake model - RF regressor
        remove_dir(model_dir)
        model = model_rf_regressor.ModelRFRegressor(model_dir=model_dir, model_name=model_name)
        model.fit(x_train, y_train_regression)
        model.save()

        # Reload
        new_model, new_config = utils_models.load_model(model_dir='test_model')
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))

        # Pareil, mais avec chemin
        new_model, new_config = utils_models.load_model(model_dir=model_dir, is_path=True)
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))
        remove_dir(model_dir)

        ####################################################

        # Tests sur un fake model - Dense classifier
        remove_dir(model_dir)
        model = model_dense_classifier.ModelDenseClassifier(model_dir=model_dir, model_name=model_name,
                                                            batch_size=batch_size, epochs=epochs, patience=patience)
        model.fit(x_train, y_train_classification)
        model.save()

        # Reload
        new_model, new_config = utils_models.load_model(model_dir='test_model')
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_config['batch_size'], batch_size)
        self.assertEqual(new_config['epochs'], epochs)
        self.assertEqual(new_config['patience'], patience)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))
        self.assertEqual([list(_) for _ in new_model.predict_proba(x_test)], [list(_) for _ in model.predict_proba(x_test)])

        # Pareil, mais avec chemin
        new_model, new_config = utils_models.load_model(model_dir=model_dir, is_path=True)
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_config['batch_size'], batch_size)
        self.assertEqual(new_config['epochs'], epochs)
        self.assertEqual(new_config['patience'], patience)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))
        self.assertEqual([list(_) for _ in new_model.predict_proba(x_test)], [list(_) for _ in model.predict_proba(x_test)])
        remove_dir(model_dir)

        ####################################################

        # Tests sur un fake model - Dense regressor
        remove_dir(model_dir)
        model = model_dense_regressor.ModelDenseRegressor(model_dir=model_dir, model_name=model_name,
                                                          batch_size=batch_size, epochs=epochs, patience=patience)
        model.fit(x_train, y_train_regression)
        model.save()

        # Reload
        new_model, new_config = utils_models.load_model(model_dir='test_model')
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_config['batch_size'], batch_size)
        self.assertEqual(new_config['epochs'], epochs)
        self.assertEqual(new_config['patience'], patience)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))

        # Pareil, mais avec chemin
        new_model, new_config = utils_models.load_model(model_dir=model_dir, is_path=True)
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_config['batch_size'], batch_size)
        self.assertEqual(new_config['epochs'], epochs)
        self.assertEqual(new_config['patience'], patience)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))
        remove_dir(model_dir)

        ####################################################

        # Tests sur un fake model - XGboost classifier
        remove_dir(model_dir)
        model = model_xgboost_classifier.ModelXgboostClassifier(model_dir=model_dir, model_name=model_name,
                                                                xgboost_params={'n_estimators': 5},
                                                                early_stopping_rounds=early_stopping_rounds)
        model.fit(x_train, y_train_classification)
        model.save()

        # Reload
        new_model, new_config = utils_models.load_model(model_dir='test_model')
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_config['early_stopping_rounds'], early_stopping_rounds)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))
        self.assertEqual([list(_) for _ in new_model.predict_proba(x_test)], [list(_) for _ in model.predict_proba(x_test)])

        # Pareil, mais avec chemin
        new_model, new_config = utils_models.load_model(model_dir=model_dir, is_path=True)
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_config['early_stopping_rounds'], early_stopping_rounds)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))
        self.assertEqual([list(_) for _ in new_model.predict_proba(x_test)], [list(_) for _ in model.predict_proba(x_test)])
        remove_dir(model_dir)

        ####################################################

        # Tests sur un fake model - XGboost regressor
        remove_dir(model_dir)
        model = model_xgboost_regressor.ModelXgboostRegressor(model_dir=model_dir, model_name=model_name,
                                                              xgboost_params={'n_estimators': 5},
                                                              early_stopping_rounds=early_stopping_rounds)
        model.fit(x_train, y_train_regression)
        model.save()

        # Reload
        new_model, new_config = utils_models.load_model(model_dir='test_model')
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_config['early_stopping_rounds'], early_stopping_rounds)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))

        # Pareil, mais avec chemin
        new_model, new_config = utils_models.load_model(model_dir=model_dir, is_path=True)
        # On fait qqs tests
        self.assertEqual(new_config['model_name'], model_name)
        self.assertEqual(new_config['early_stopping_rounds'], early_stopping_rounds)
        self.assertEqual(new_model.model_name, model_name)
        self.assertEqual(list(new_model.predict(x_test)), list(model.predict(x_test)))
        remove_dir(model_dir)

        ####################################################

        # Vérification raises
        with self.assertRaises(FileNotFoundError):
            utils_models.load_model(model_dir='tototo')
        with self.assertRaises(FileNotFoundError):
            utils_models.load_model(model_dir='./tototo', is_path=True)


    def test08_get_columns_pipeline(self):
        '''Test de la fonction ynov.models_training.utils_models.get_columns_pipeline'''
        # Set fake pipelines
        numeric_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
        text_pipeline = make_pipeline(CountVectorizer(), SelectKBest(k=5))
        transformers = [
            ('num', numeric_pipeline, ['Age', 'SibSp', 'Parch', 'Fare']),
            ('cat', cat_pipeline, ['Pclass', 'Sex', 'Embarked']), # Pour convertir une colonne en 'category' -> df["A"].astype("category")
            ('text', text_pipeline, 'Name'), # CountVectorizer possible sur une seule colonne à la fois
        ]
        pipeline1 = ColumnTransformer(transformers, remainder='drop')
        pipeline2 = ColumnTransformer(transformers, remainder='passthrough')

        # Fake DataFrame to fit the pipelines
        df = pd.read_csv('test_dataset.csv', sep=',', encoding='utf-8')
        y = df['Survived']
        X = df.drop('Survived', axis=1)
        pipeline1.fit(X, y)
        pipeline2.fit(X, y)

        # Fonctionnement nominal
        columns_in1, mandatory_columns1 = utils_models.get_columns_pipeline(pipeline1)
        columns_in2, mandatory_columns2 = utils_models.get_columns_pipeline(pipeline2)
        self.assertEqual(set(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']), set(columns_in1))
        self.assertEqual(set(['Age', 'SibSp', 'Parch', 'Fare', 'Pclass', 'Sex', 'Embarked', 'Name']), set(mandatory_columns1))
        self.assertEqual(set(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']), set(columns_in2))
        self.assertEqual(set(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']), set(mandatory_columns2))


    def test09_apply_pipeline(self):
        '''Test de la fonction ynov.models_training.utils_models.apply_pipeline'''
        # Set fake pipelines
        numeric_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
        cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
        text_pipeline = make_pipeline(CountVectorizer(), SelectKBest(k=5))
        transformers = [
            ('num', numeric_pipeline, ['Age', 'SibSp', 'Parch', 'Fare']),
            ('cat', cat_pipeline, ['Pclass', 'Sex', 'Embarked']), # Pour convertir une colonne en 'category' -> df["A"].astype("category")
            ('text', text_pipeline, 'Name'), # CountVectorizer possible sur une seule colonne à la fois
        ]
        pipeline1 = ColumnTransformer(transformers, remainder='drop')
        pipeline2 = ColumnTransformer(transformers, remainder='passthrough')

        # Fake DataFrame to fit the pipelines
        df = pd.read_csv('test_dataset.csv', sep=',', encoding='utf-8')
        y = df['Survived']
        X = df.drop('Survived', axis=1)
        pipeline1.fit(X, y)
        pipeline2.fit(X, y)

        # Fonctionnement nominal
        preprocessed_df1 = utils_models.apply_pipeline(X, pipeline1)
        self.assertEqual(preprocessed_df1.shape, (100, 17))
        preprocessed_df2 = utils_models.apply_pipeline(X, pipeline2)
        self.assertEqual(preprocessed_df2.shape, (100, 20))

        # Avec colonnes en trop
        df['toto'] = 5
        preprocessed_df1 = utils_models.apply_pipeline(df, pipeline1)
        self.assertEqual(preprocessed_df1.shape, (100, 17))
        preprocessed_df2 = utils_models.apply_pipeline(df, pipeline2)
        self.assertEqual(preprocessed_df2.shape, (100, 20))

        # Avec missing optional column
        new_X = X.drop('PassengerId', axis=1)
        preprocessed_df1 = utils_models.apply_pipeline(new_X, pipeline1)
        self.assertEqual(preprocessed_df1.shape, (100, 17))
        with self.assertRaises(ValueError):
            preprocessed_df2 = utils_models.apply_pipeline(new_X, pipeline2)

        # Avec missing mendatory column
        new_X = X.drop('Age', axis=1)
        with self.assertRaises(ValueError):
            preprocessed_df1 = utils_models.apply_pipeline(new_X, pipeline1)
        with self.assertRaises(ValueError):
            preprocessed_df2 = utils_models.apply_pipeline(new_X, pipeline2)


    def test10_predict(self):
        '''Test de la fonction ynov.models_training.utils_models.predict'''

        # Données pour apprentissage
        x_train_1 = pd.DataFrame({'col_1': [-5, 5, -5, 5, 5, 5] * 10})
        x_train_2 = pd.DataFrame({'col_1': [-5, -5, -5, 5, 5, 5] * 10, 'col_2': [0, 0, 0, 0, 0, 0] * 10})
        x_test_1 = pd.DataFrame({'col_1': [-5, 5, -5, 5, -5, 5]})
        x_test_1_solo = pd.DataFrame({'col_1': [-5]})
        x_test_2 = pd.DataFrame({'col_1': [-5, 5, -5, 5, -5, 5], 'col_2': [0, 0, 0, 0, 0, 0]})
        x_test_2_solo = pd.DataFrame({'col_1': [5], 'col_2': [0]})
        y_train_classification = pd.Series([0, 0, 0, 1, 1, 1] * 10)
        y_train_classification_multi = pd.DataFrame({'test': [1, 1, 1, 0, 0, 0] * 10, 'toto': [0, 0, 0, 1, 1, 1] * 10})
        y_train_regression = pd.Series([-10, -10, -10, 10, 10, 10] * 10)
        y_test_classification = [0, 1, 0, 1, 0, 1]
        y_test_classification_multi = [('test',), ('toto',), ('test',), ('toto',), ('test',), ('toto',)]
        y_test_1_classification_solo = 0
        y_test_2_classification_solo = 1
        y_test_1_classification_multi_solo = ('test',)
        y_test_2_classification_multi_solo = ('toto',)
        model_dir = os.path.join(utils.get_models_path(), 'test_model')
        model_name = 'test_model_name'

        ################
        # RF Classification - monolabel - 1

        # Creation fake model
        remove_dir(model_dir)
        model = model_rf_classifier.ModelRFClassifier(model_dir=model_dir, model_name=model_name)
        model.fit(x_train_1, y_train_classification)
        model.save()

        # Fonctionnement nominal
        model, model_conf = utils_models.load_model(model_dir='test_model')
        self.assertEqual(utils_models.predict(x_test_1, model), y_test_classification)
        self.assertEqual(utils_models.predict(x_test_1_solo, model), y_test_1_classification_solo)
        remove_dir(model_dir)

        ################
        # RF Classification - monolabel - 2

        # Creation fake model
        remove_dir(model_dir)
        model = model_rf_classifier.ModelRFClassifier(model_dir=model_dir, model_name=model_name)
        model.fit(x_train_2, y_train_classification)
        model.save()

        # Fonctionnement nominal
        model, model_conf = utils_models.load_model(model_dir='test_model')
        self.assertEqual(utils_models.predict(x_test_2, model), y_test_classification)
        self.assertEqual(utils_models.predict(x_test_2_solo, model), y_test_2_classification_solo)
        remove_dir(model_dir)

        ################
        # RF Classification - multilabel - 1

        # Creation fake model
        remove_dir(model_dir)
        model = model_rf_classifier.ModelRFClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
        model.fit(x_train_1, y_train_classification_multi)
        model.save()

        # Fonctionnement nominal
        model, model_conf = utils_models.load_model(model_dir='test_model')
        self.assertEqual(utils_models.predict(x_test_1, model), y_test_classification_multi)
        self.assertEqual(utils_models.predict(x_test_1_solo, model), y_test_1_classification_multi_solo)
        remove_dir(model_dir)

        ################
        # RF Classification - multilabel - 2

        # Creation fake model
        remove_dir(model_dir)
        model = model_rf_classifier.ModelRFClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
        model.fit(x_train_2, y_train_classification_multi)
        model.save()

        # Fonctionnement nominal
        model, model_conf = utils_models.load_model(model_dir='test_model')
        self.assertEqual(utils_models.predict(x_test_2, model), y_test_classification_multi)
        self.assertEqual(utils_models.predict(x_test_2_solo, model), y_test_2_classification_multi_solo)
        remove_dir(model_dir)

        ################
        # RF Regression - 1

        # Creation fake model
        remove_dir(model_dir)
        model = model_rf_regressor.ModelRFRegressor(model_dir=model_dir, model_name=model_name)
        model.fit(x_train_1, y_train_regression)
        model.save()

        # Fonctionnement nominal
        model, model_conf = utils_models.load_model(model_dir='test_model')
        # On check juste le type, compliqué avec de la regression ...
        self.assertEqual(type(utils_models.predict(x_test_1, model)), list)
        self.assertTrue(isinstance(utils_models.predict(x_test_1_solo, model), (np.floating, float)))
        remove_dir(model_dir)

        ################
        # RF Regression - 2

        # Creation fake model
        remove_dir(model_dir)
        model = model_rf_regressor.ModelRFRegressor(model_dir=model_dir, model_name=model_name)
        model.fit(x_train_2, y_train_regression)
        model.save()

        # Fonctionnement nominal
        model, model_conf = utils_models.load_model(model_dir='test_model')
        # On check juste le type, compliqué avec de la regression ...
        self.assertEqual(type(utils_models.predict(x_test_2, model)), list)
        self.assertTrue(isinstance(utils_models.predict(x_test_2_solo, model), (np.floating, float)))
        remove_dir(model_dir)


    def test11_predict_with_proba(self):
        '''Test de la fonction ynov.models_training.utils_models.predict_with_proba'''

        # Données pour apprentissage
        x_train_1 = pd.DataFrame({'col_1': [-5, 5, -5, 5, 5, 5] * 10})
        x_train_2 = pd.DataFrame({'col_1': [-5, -5, -5, 5, 5, 5] * 10, 'col_2': [0, 0, 0, 0, 0, 0] * 10})
        x_test_1 = pd.DataFrame({'col_1': [-5, 5, -5, 5, -5, 5]})
        x_test_1_solo = pd.DataFrame({'col_1': [-5]})
        x_test_2 = pd.DataFrame({'col_1': [-5, 5, -5, 5, -5, 5], 'col_2': [0, 0, 0, 0, 0, 0]})
        x_test_2_solo = pd.DataFrame({'col_1': [5], 'col_2': [0]})
        y_train_classification = pd.Series([0, 0, 0, 1, 1, 1] * 10)
        y_train_classification_multi = pd.DataFrame({'test': [1, 1, 1, 0, 0, 0] * 10, 'toto': [0, 0, 0, 1, 1, 1] * 10})
        y_train_regression = pd.Series([-10, -10, -10, 10, 10, 10] * 10)
        y_test_classification = [0, 1, 0, 1, 0, 1]
        y_test_classification_multi = [('test',), ('toto',), ('test',), ('toto',), ('test',), ('toto',)]
        y_test_1_classification_solo = 0
        y_test_2_classification_solo = 1
        y_test_1_classification_multi_solo = ('test',)
        y_test_2_classification_multi_solo = ('toto',)
        model_dir = os.path.join(utils.get_models_path(), 'test_model')
        model_name = 'test_model_name'

        ################
        # RF Classification - monolabel - 1

        # Creation fake model
        remove_dir(model_dir)
        model = model_rf_classifier.ModelRFClassifier(model_dir=model_dir, model_name=model_name)
        model.fit(x_train_1, y_train_classification)
        model.save()

        # Fonctionnement nominal
        model, model_conf = utils_models.load_model(model_dir='test_model')
        pred, proba = utils_models.predict_with_proba(x_test_1, model)
        self.assertEqual(pred, y_test_classification)
        self.assertEqual(len(proba), len(y_test_classification))
        self.assertEqual(sum([round(p) for p in proba]), len(y_test_classification))
        pred, proba = utils_models.predict_with_proba(x_test_1_solo, model)
        self.assertEqual(pred, y_test_1_classification_solo)
        self.assertTrue(proba >= 0.5)
        remove_dir(model_dir)

        ################
        # RF Classification - monolabel - 2

        # Creation fake model
        remove_dir(model_dir)
        model = model_rf_classifier.ModelRFClassifier(model_dir=model_dir, model_name=model_name)
        model.fit(x_train_2, y_train_classification)
        model.save()

        # Fonctionnement nominal
        model, model_conf = utils_models.load_model(model_dir='test_model')
        pred, proba = utils_models.predict_with_proba(x_test_2, model)
        self.assertEqual(pred, y_test_classification)
        self.assertEqual(len(proba), len(y_test_classification))
        self.assertEqual(sum([round(p) for p in proba]), len(y_test_classification))
        pred, proba = utils_models.predict_with_proba(x_test_2_solo, model)
        self.assertEqual(pred, y_test_2_classification_solo)
        self.assertTrue(proba >= 0.5)
        remove_dir(model_dir)

        ################
        # RF Classification - multilabel - 1

        # Creation fake model
        remove_dir(model_dir)
        model = model_rf_classifier.ModelRFClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
        model.fit(x_train_1, y_train_classification_multi)
        model.save()

        # Fonctionnement nominal
        model, model_conf = utils_models.load_model(model_dir='test_model')
        pred, proba = utils_models.predict_with_proba(x_test_1, model)
        self.assertEqual(pred, y_test_classification_multi)
        self.assertEqual(len(proba), len(y_test_classification_multi))
        self.assertEqual(sum([round(p[0]) for p in proba]), sum([len(_) for _ in y_test_classification_multi]))
        pred, proba = utils_models.predict_with_proba(x_test_1_solo, model)
        self.assertEqual(pred, y_test_1_classification_multi_solo)
        self.assertTrue(proba[0] >= 0.5)
        remove_dir(model_dir)

        ################
        # RF Classification - multilabel - 2

        # Creation fake model
        remove_dir(model_dir)
        model = model_rf_classifier.ModelRFClassifier(model_dir=model_dir, model_name=model_name, multi_label=True)
        model.fit(x_train_2, y_train_classification_multi)
        model.save()

        # Fonctionnement nominal
        model, model_conf = utils_models.load_model(model_dir='test_model')
        pred, proba = utils_models.predict_with_proba(x_test_2, model)
        self.assertEqual(pred, y_test_classification_multi)
        self.assertEqual(len(proba), len(y_test_classification_multi))
        self.assertEqual(sum([round(p[0]) for p in proba]), sum([len(_) for _ in y_test_classification_multi]))
        pred, proba = utils_models.predict_with_proba(x_test_2_solo, model)
        self.assertEqual(pred, y_test_2_classification_multi_solo)
        self.assertTrue(proba[0] >= 0.5)
        remove_dir(model_dir)

        ################
        # RF Regression

        # Creation fake model
        remove_dir(model_dir)
        model = model_rf_regressor.ModelRFRegressor(model_dir=model_dir, model_name=model_name)
        model.fit(x_train_1, y_train_regression)
        model.save()

        with self.assertRaises(ValueError):
            pred, proba = utils_models.predict_with_proba(x_test_1, model)

        remove_dir(model_dir)


    def test12_search_hp_cv_classifier(self):
        '''Test de la fonction ynov.models_training.utils_models.search_hp_cv_classifier'''
        # Défintion des variables pour un fonctionnement nominal
        x_train = pd.DataFrame({'col_1': [-5, -1, 0, 2, -6, 3] * 10, 'col_2': [2, -1, -8, 3, 12, 2] * 10})
        y_train_mono = pd.Series([0, 0, 0, 1, 1, 1] * 10)
        y_train_multi = pd.DataFrame({'y1': [0, 0, 0, 1, 1, 1] * 10, 'y2': [1, 0, 0, 1, 1, 1] * 10})
        model_cls = model_rf_classifier.ModelRFClassifier
        model_params_mono = {'multi_label': False}
        model_params_multi = {'multi_label': True}
        hp_params = {'rf_params': [{'max_depth': 5, 'n_estimators': 50}, {'max_depth': 10, 'n_estimators': 25}], 'multiclass_strategy': ['ovo', 'ovr']}
        kwargs_fit_mono = {'x_train': x_train, 'y_train': y_train_mono}
        kwargs_fit_multi = {'x_train': x_train, 'y_train': y_train_multi}

        # Fonctionnement nominal
        n_splits = 5
        model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, hp_params, "accuracy", kwargs_fit_mono, n_splits=n_splits)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        csv_path = os.path.join(model.model_dir, f"hyper_params_results.csv")
        json_path = os.path.join(model.model_dir, f"hyper_params_tested.json")
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        with open(csv_path, 'r', encoding="utf-8") as f:
            self.assertEqual(sum(1 for line in f), n_splits * len(hp_params) + 1)
        remove_dir(model.model_dir)

        # Fonctionnement multi
        n_splits = 3
        model = utils_models.search_hp_cv_classifier(model_cls, model_params_multi, hp_params, "f1", kwargs_fit_multi, n_splits=n_splits)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        csv_path = os.path.join(model.model_dir, f"hyper_params_results.csv")
        json_path = os.path.join(model.model_dir, f"hyper_params_tested.json")
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        with open(csv_path, 'r', encoding="utf-8") as f:
            self.assertEqual(sum(1 for line in f), n_splits * len(hp_params) + 1)
        remove_dir(model.model_dir)

        # On check les différentest fonctions de scoring...
        n_splits = 2
        model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, hp_params, "precision", kwargs_fit_mono, n_splits=n_splits)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        csv_path = os.path.join(model.model_dir, f"hyper_params_results.csv")
        json_path = os.path.join(model.model_dir, f"hyper_params_tested.json")
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        with open(csv_path, 'r', encoding="utf-8") as f:
            self.assertEqual(sum(1 for line in f), n_splits * len(hp_params) + 1)
        remove_dir(model.model_dir)

        n_splits = 2
        model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, hp_params, "recall", kwargs_fit_mono, n_splits=n_splits)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        csv_path = os.path.join(model.model_dir, f"hyper_params_results.csv")
        json_path = os.path.join(model.model_dir, f"hyper_params_tested.json")
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        with open(csv_path, 'r', encoding="utf-8") as f:
            self.assertEqual(sum(1 for line in f), n_splits * len(hp_params) + 1)
        remove_dir(model.model_dir)

        # Fonction de scoring custom
        def custom_func(test_dict: dict):
            return (test_dict['Precision'] + test_dict['Recall']) / 2
        n_splits = 2
        model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, hp_params, custom_func, kwargs_fit_mono, n_splits=n_splits)
        self.assertFalse(model.trained)
        self.assertEqual(model.nb_fit, 0)
        csv_path = os.path.join(model.model_dir, f"hyper_params_results.csv")
        json_path = os.path.join(model.model_dir, f"hyper_params_tested.json")
        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))
        with open(csv_path, 'r', encoding="utf-8") as f:
            self.assertEqual(sum(1 for line in f), n_splits * len(hp_params) + 1)
        remove_dir(model.model_dir)


        # Check des erreurs
        with self.assertRaises(TypeError):
            model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, hp_params, 5, kwargs_fit_mono, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, hp_params, 'toto', kwargs_fit_mono, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv_classifier(model_cls, {'toto': True}, hp_params, 'accuracy', kwargs_fit_mono, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, hp_params, 'accuracy', {'toto': True, 'y_train': y_train_mono}, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, hp_params, 'accuracy', {'toto': True, 'x_train': x_train}, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, hp_params, 'accuracy', {'toto': True, 'x_train': x_train}, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv_classifier(model_cls, {**model_params_mono, **{'toto': True}}, {**hp_params, **{'toto': [False]}}, 'accuracy', kwargs_fit_mono, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, {'toto': True}, 'accuracy', kwargs_fit_mono, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, {'toto': [1, 2], 'titi': [3]}, 'accuracy', kwargs_fit_mono, n_splits=n_splits)
        with self.assertRaises(ValueError):
            model = utils_models.search_hp_cv_classifier(model_cls, model_params_mono, hp_params, 'accuracy', kwargs_fit_mono, n_splits=1)


# Execution des tests
if __name__ == '__main__':
    # Start tests
    unittest.main()