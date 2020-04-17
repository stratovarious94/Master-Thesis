import os
import pandas as pd

from src.statistics import helpers

project_root = os.path.dirname(os.path.dirname(__file__))
project_source = os.path.dirname(project_root)


def show():
    """
    Utility function that reads the datasets produced from the crawler for statistical purposes.
    Then it proceeds to display some figures in order to give insights about the dataset.
    """
    data_stats = pd.read_csv(project_source + '/data/dataset/capstone_stats.csv')
    data_complete = pd.read_csv(project_source + '/data/dataset/capstone_complete.csv')
    data_url = pd.read_csv(project_source + '/data/dataset/capstone_url_v1.csv')

    helpers.broken_links_per_site(data_stats)
    helpers.broken_links_by_site_type(data_stats)
    helpers.images_per_directory_no_of_images(data_url)
    helpers.same_barcoded_id(data_complete)
    helpers.diff_res_per_year(data_url, data_complete)
    helpers.diff_res_per_site(data_url, data_complete)
