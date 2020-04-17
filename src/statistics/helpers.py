import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


def transform_test_y(data_complete):
    """
    Utility function that accepts a dataset and converts the task columns from numerical to string.

    :param data_complete: A dataset enriched with additional information from the crawler
    :return: The columns containing the y_labels in string form
    """
    data_complete["brand_id"] = data_complete["brand_id"].astype(int).astype(str)
    data_complete["category_id"] = data_complete["category_id"].astype(int).astype(str)
    data_complete["product_line_id"] = data_complete["product_line_id"].astype(int).astype(str)


def broken_links_per_site(data_stats):
    """
    Figure that shows which sites have the most broken links, prohibiting us from downloading their images.

    :param data_stats: A dataset enriched with additional information from the crawler
    """
    by_site = data_stats.groupby(["site_id", "status"]).agg({"product_id": "count", "site_product_url": "first"}).reset_index().rename(columns={"product_id": "count"})
    plt.figure(figsize=(20, 15))
    sns.barplot(x="site_id", y="count", data=by_site, hue="status", order=data_stats.site_id.value_counts().index)
    plt.show()


def broken_links_by_site_type(data_stats):
    """
    Figure that shows which site type (supermarket or pharmacy) has got the most broken links, prohibiting us from downloading their images.

    :param data_stats: A dataset enriched with additional information from the crawler
    """
    by_site_type = data_stats.groupby(["site_type_id", "status"]).agg({"product_id": "count"}).reset_index().rename(columns={"product_id": "count"})
    sns.barplot(x="site_type_id", y="count", data=by_site_type, hue="status")
    plt.show()
    dt_temp = data_stats[data_stats["site_type_id"] == 1]
    print("Broken links from Pharmacies: ", dt_temp[dt_temp["status"] == False].shape)
    dt_temp = data_stats[data_stats["site_type_id"] == 165]
    print("Broken links from Supermarkets: ", dt_temp[dt_temp["status"] == False].shape)


def images_per_directory_no_of_images(data_url):
    """
    Figure that shows products that contain more than one image_url.

    :param data_url: A dataset enriched with additional information from the crawler
    """
    by_product_id = data_url.product_id.value_counts().value_counts()
    plt.figure(figsize=(20, 15))
    sns.barplot(x=by_product_id.index.astype(str).tolist(), y=np.log(by_product_id.values.tolist()), order=by_product_id.index.astype(str).tolist())
    plt.show()
    print(by_product_id)


def same_barcoded_id(data_complete):
    """
    Figure that shows which products have got the same barcode (It means they are identical but from different stores).

    :param data_complete: A dataset enriched with additional information from the crawler
    """
    by_barcoded_id = data_complete.barcoded_product_id.value_counts().value_counts()
    plt.figure(figsize=(20, 15))
    sns.barplot(x=by_barcoded_id.index.astype(str).tolist(), y=np.log(by_barcoded_id.values.tolist()), order=by_barcoded_id.index.astype(str).tolist())
    plt.show()
    print("Top 10\n", by_barcoded_id[:10], "\nLast 10\n", by_barcoded_id[-10:], "\nProducts without match: ", len(by_barcoded_id[by_barcoded_id == 1]))


def diff_res_per_year(data_url, data_complete):
    """
    Figure that shows the number of different resolutions in the images downloaded by year.

    :param data_url, data_complete: Datasets enriched with additional information from the crawler
    """
    by_date_res = pd.merge(data_url, data_complete[['product_id', 'last_sale']], on="product_id")
    by_date_res['last_sale'] = by_date_res['last_sale'].apply(lambda date: str(date).split('-', 1)[0])
    return by_date_res.groupby(["last_sale"]).agg({"last_sale": "first", "resolution": lambda x: x.nunique()})


def diff_res_per_site(data_url, data_complete):
    """
    Figure that shows the number of different resolutions in the images downloaded by website.

    :param data_url, data_complete: A dataset enriched with additional information from the crawler
    """
    by_site_res = pd.merge(data_url, data_complete[['product_id', 'site_id']], on="product_id")
    return by_site_res.groupby(["site_id"]).agg({"site_id": "first", "resolution": lambda x: x.nunique()}).sort_values("resolution", ascending=False)[:10]
