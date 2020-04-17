import os
import pandas as pd
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(__file__))
project_source = os.path.dirname(project_root)


class Preprocessor:
    """
    This class accepts the following arguments:
    path: The path to the dataset with the URLs, their resolutions, and tasks
    max_width: The maximum resolution width allowed for images e.g. 3000 pixels
    max_height: The maximum resolution height allowed for images e.g. 3000 pixels
    ch: The channels not allowed for the images (Currently Grayscale, RGB, RGBA allowed)
    top: How many classes will be allowed as a maximum for each task e.g. top 2000

    High level functionality:
    1) Retrieve dataset and remove images acording to specified characteristics
    2) Split the dataset in train, validation and test set
    3) Return the sunbsets and save the test set for future use
    """
    def __init__(self, min_el):
        self.max_width = 3000
        self.max_height = 3000
        self.NOT_USED_NUMBER_OF_CHANNELS = 2
        self.min_el = min_el

    def retrieve_from_dataset(self):
        """
        Functionality:
        1) Retrieve the dataset from the specified path
        2) Count the number of classes of each task and add them to the dataset
        3) Filter out entries from classes that have a count bellow 5
        4) Filter out entries that exceed specified width,height,number of channels
        5) Drop NaN values
        """
        data = pd.read_csv(project_source + '/data/dataset/capstone_url_v1.csv')
        data = data[data["width"] <= self.max_width]
        data = data[data["height"] <= self.max_height]
        data = data[data["channels"] != self.NOT_USED_NUMBER_OF_CHANNELS]
        data = data.dropna()
        data['class_comb'] = data['brand_id'].astype(str) + '_' + data['category_id'].astype(str) + '_' + data['product_line_id'].astype(str)
        class_counts = data['class_comb'].value_counts().rename('class_counts')
        data = data.merge(class_counts.to_frame(), left_on='class_comb', right_index=True)
        data = data[data.class_counts >= self.min_el]
        return data

    def split_train_val(self, data):
        """
        Functionality:
        1) Get URL as "id" and task_id as "label" from the dataset
        2) Train: 0.84 Validation: 0.16
        """
        x_train, x_val, y_train, y_val = train_test_split(data[["site_image_url"]].values, data[['brand_id', 'category_id', 'product_line_id', 'barcoded_product_id', 'class_comb']]
                                                          .astype(str).values, test_size=0.16, random_state=42,
                                                          shuffle=True, stratify=data[['class_comb']])

        train = pd.DataFrame({'id': x_train.ravel(), 'brand_id': [row[0] for row in y_train], 'category_id': [row[1] for row in y_train],
                              'product_line_id': [row[2] for row in y_train], 'barcoded_product_id': [row[3] for row in y_train], 'class_comb': [row[4] for row in y_train]})
        valid = pd.DataFrame({'id': x_val.ravel(), 'brand_id': [row[0] for row in y_val], 'category_id': [row[1] for row in y_val],
                              'product_line_id': [row[2] for row in y_val], 'barcoded_product_id': [row[3] for row in y_val]})
        return train, valid

    def split_train_test(self, train):
        """
        Functionality:
        1) Get URL as "id" and task_id as "label" from the dataset
        2) Train: 0.84 Test: 0.16
        """
        x_train, x_test, y_train, y_test = train_test_split(train[["id"]].values, train[['brand_id', 'category_id', 'product_line_id', 'barcoded_product_id']].astype(str)
                                                            .values, test_size=0.2, random_state=42, shuffle=True,
                                                            stratify=train[["class_comb"]])

        train = pd.DataFrame({'id': x_train.ravel(), 'brand_id': [row[0] for row in y_train],
                              'category_id': [row[1] for row in y_train], 'product_line_id': [row[2] for row in y_train], 'barcoded_product_id': [row[3] for row in y_train]})
        test = pd.DataFrame({'id': x_test.ravel(), 'brand_id': [row[0] for row in y_test],
                             'category_id': [row[1] for row in y_test], 'product_line_id': [row[2] for row in y_test], 'barcoded_product_id': [row[3] for row in y_test]})
        return train, test

    def save_sets(self, train, valid, test):
        """
        Save the test set of each task to data directory for future use
        """
        print("Brand Training classes: ", train['brand_id'].value_counts().shape[0],
              "\n Category Validation classes: ", valid['brand_id'].value_counts().shape[0],
              "\nProduct line Testing classes: ", test['brand_id'].value_counts().shape[0])

        print("Brand Training classes: ", train['category_id'].value_counts().shape[0],
              "\nCategory Validation classes: ", valid['category_id'].value_counts().shape[0],
              "\nTesting classes: ", test['category_id'].value_counts().shape[0])

        print("Brand Training classes: ", train['product_line_id'].value_counts().shape[0],
              "\nCategory Validation classes: ", valid['product_line_id'].value_counts().shape[0],
              "\nProduct line Testing classes: ", test['product_line_id'].value_counts().shape[0])

        print("Brand Training classes: ", train['barcoded_product_id'].value_counts().shape[0],
              "\nCategory Validation classes: ", valid['barcoded_product_id'].value_counts().shape[0],
              "\nProduct line Testing classes: ", test['barcoded_product_id'].value_counts().shape[0])

        train.to_csv(project_source + '/data/classification/train/train_' + str(self.min_el) + '.csv', index=False, encoding='utf-8-sig')
        valid.to_csv(project_source + '/data/classification/valid/valid_' + str(self.min_el) + '.csv', index=False, encoding='utf-8-sig')
        test.to_csv(project_source + '/data/classification/test/test_' + str(self.min_el) + '.csv', index=False, encoding='utf-8-sig')

    def save_matching_sets(self, train, valid, test):
        """
        Save the test set of each task to data directory for future use
        """

        valid = valid[valid['barcoded_product_id'].isin(train['barcoded_product_id'].unique())]
        test = test[test['barcoded_product_id'].isin(train['barcoded_product_id'].unique())]

        print("Brand Training classes: ", train['brand_id'].value_counts().shape[0],
              "\n Category Validation classes: ", valid['brand_id'].value_counts().shape[0],
              "\nProduct line Testing classes: ", test['brand_id'].value_counts().shape[0])

        print("Brand Training classes: ", train['category_id'].value_counts().shape[0],
              "\nCategory Validation classes: ", valid['category_id'].value_counts().shape[0],
              "\nTesting classes: ", test['category_id'].value_counts().shape[0])

        print("Brand Training classes: ", train['product_line_id'].value_counts().shape[0],
              "\nCategory Validation classes: ", valid['product_line_id'].value_counts().shape[0],
              "\nProduct line Testing classes: ", test['product_line_id'].value_counts().shape[0])

        print("Brand Training classes: ", train['barcoded_product_id'].value_counts().shape[0],
              "\nCategory Validation classes: ", valid['barcoded_product_id'].value_counts().shape[0],
              "\nProduct line Testing classes: ", test['barcoded_product_id'].value_counts().shape[0])

        train.to_csv(project_source + '/data/matching/train/train_' + str(self.min_el) + '.csv', index=False, encoding='utf-8-sig')
        valid.to_csv(project_source + '/data/matching/valid/valid_' + str(self.min_el) + '.csv', index=False, encoding='utf-8-sig')
        test.to_csv(project_source + '/data/matching/test/test_' + str(self.min_el) + '.csv', index=False, encoding='utf-8-sig')

    def get_sets(self):
        # Execute the above functions in order
        data = self.retrieve_from_dataset()
        train, valid = self.split_train_val(data)
        train, test = self.split_train_test(train)
        self.save_sets(train, valid, test)
        self.save_matching_sets(train, valid, test)
