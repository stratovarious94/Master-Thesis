import numpy as np
import pandas as pd
import requests
import shutil
import os
from PIL import Image, ImageFile
import time
from threading import Thread
ImageFile.LOAD_TRUNCATED_IMAGES = True

project_root = os.path.dirname(os.path.dirname(__file__))

data = pd.read_csv(project_root + "/capstone-project-2019/data/dataset/dataset.csv", low_memory=False)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data = data.dropna(subset=['site_image_url'])
data = data[data['site_image_url'].str.startswith("{http")]
data = data.applymap(str)

img_path = project_root + "/capstone-project-2019/data/images"
print('Root Folder: ' + img_path)


def retrieve_leftover_url(row):
    """
    Requests all image URLs associated with a product in the dataframe and saves them in a local directory called images.
    The subdirectory name name will be the product id and the image name [img0-...].png

    :param row: A row in the dataframe representing a distinct product
    """
    # Declare a path in which all the images will be stored
    path = project_root + "/capstone-project-2019/data/images/" + str(row['product_id'])
    print('Downloaded: ' + path)
    # If I created a productId directory in fail attempt, it will be removed
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

    # Iterate multiple images if that's the case
    split_url = row["site_image_url"][1:-1].rsplit('/', 1)
    base_url = split_url[0]
    for i, end_url in enumerate(split_url[1].split(",")):
        url = base_url + "/" + end_url
        # make a request to the image
        try:
            time.sleep(1)
            r = requests.get(url)
            with open(path + "/" + "img" + str(i) + ".png", 'wb') as f:
                f.write(r.content)
        except (requests.exceptions.RequestException, OSError) as e:
            print(e, "\nproduct ID: ", row['product_id'], "\nURL: ", url)


existing_folders = []
for root, dirs, files in list(os.walk(img_path))[1:]:
    existing_folders.append(root.rsplit("/", 1)[1])

data_to_download = data[~data["product_id"].isin(set(existing_folders))]


def retrieve_leftover_url_multithread(site_id):
    data_to_download[data_to_download["site_id"] == site_id].apply(lambda row: retrieve_leftover_url(row), axis=1)


threads = []
for site_id in data_to_download.site_id.unique():
    process = Thread(target=retrieve_leftover_url_multithread, args=[site_id])
    process.start()
    threads.append(process)

for process in threads:
    process.join()

# Delete corrupted files
for root, dirs, files in os.walk(project_root + "/capstone-project-2019/data/images"):
    for file in files:
        try:
            img = Image.open(root + "/" + file)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', root + "/" + file)
            os.remove(root + "/" + file)

# Remove empty folders
for root, dirs, files in list(os.walk(project_root + "/capstone-project-2019/data/images"))[1:]:
    # folder example: ('FOLDER/3', [], ['file'])
    if not files:
        print(root)
        os.rmdir(root)

img_list = []
for root, dirs, files in os.walk(img_path):
    for file in files:
        img_list.append((root.rsplit('/', 1)[1], root + "/" + file))
img_list2 = [item[0] for item in img_list]
stats_df = data
stats_df['status'] = stats_df["product_id"].isin(img_list2)
stats_df.to_csv(project_root + '/capstone-project-2019/data/dataset/capstone_stats.csv', index=False, encoding='utf-8-sig')


def create_datasets(attempt):
    """
    Creates complete datasets for statistical purposes and a core dataset containing the urls of the images in the local folder.

    :param attempt: The version of the dataset that the user wants created
    """
    df = pd.read_csv(project_root + "/capstone-project-2019/data/dataset/capstone_stats.csv")

    # URL dataset
    img_list = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            img_list.append((root.rsplit('/', 1)[1], root + "/" + file))
    url_df = pd.DataFrame(img_list, columns=['product_id', 'site_image_url'])
    url_df = pd.merge(url_df.applymap(str), df[['product_id', 'brand_id', 'category_id', 'product_line_id', 'barcoded_product_id']].applymap(int).applymap(str), on='product_id')
    url_df.to_csv(project_root + '/capstone-project-2019/data/dataset/capstone_url' + attempt + '.csv', index=False)

    print('done')

    url_matching_df = pd.DataFrame(img_list, columns=['product_id', 'site_image_url'])
    url_matching_df = pd.merge(url_matching_df.applymap(str), df[['product_id', 'barcoded_product_id']].applymap(int).applymap(str), on='product_id')
    url_matching_df.to_csv(project_root + '/capstone-project-2019/data/dataset/matching_url' + attempt + '.csv', index=False)

    # None broken entries dataset
    complete_df = df[df["product_id"].isin(set(url_df['product_id']))]
    complete_df.to_csv(project_root + '/capstone-project-2019/data/dataset/capstone_complete' + attempt + '.csv', index=False, encoding='utf-8-sig')

    print('done')

    # Get statistics dataset
    stats_df = df
    stats_df['status'] = df["product_id"].isin(set(url_df['product_id']))
    stats_df.to_csv(project_root + '/capstone-project-2019/data/dataset/capstone_stats' + attempt + '.csv', index=False, encoding='utf-8-sig')

    print('done')


create_datasets('')

data_url = pd.read_csv(project_root + '/capstone-project-2019/data/dataset/capstone_url.csv')


def add_resolution(data_url,ver):
    """
    Adds the resolution in the dataset that contains the URLs

    :param data_url: The URL of a single row in the dataset
    :param ver: Version of the dataset to expand resolutions
    :return:
    """
    data_url['resolution'] = data_url['site_image_url'].apply(lambda el: np.array(Image.open(el)).shape)
    data_url.to_csv(project_root + '/capstone-project-2019/data/dataset/capstone_url_res'+ver+'.csv', index=False, encoding='utf-8-sig')


add_resolution(data_url, '')

data_url = pd.read_csv(project_root + '/capstone-project-2019/data/dataset/capstone_url_res.csv')


def expand_resolution(data_url, ver):
    """
    Split the resolution (tuple) in width, height, channels for convenience and save the final core dataset
    """
    data_url["width"]=data_url["resolution"].apply(lambda el: int(el[1:-1].split(",")[0]) if len(el)>2 else 1)
    data_url["height"]=data_url["resolution"].apply(lambda el: int(el[1:-1].split(",")[1]) if len(el[1:-1].split(","))>1 else 1)
    data_url["channels"]=data_url["resolution"].apply(lambda el: int(el[1:-1].split(",")[2]) if len(el[1:-1].split(","))>2 else 1)
#     for img in data_url[data_url["channels"]==4]["site_image_url"]:
#         Image.open(img).convert('RGBA').save(img)
    data_url.to_csv(project_root + '/capstone-project-2019/data/dataset/capstone_url_v1'+ver+'.csv', index=False, encoding='utf-8-sig')


expand_resolution(data_url, '')
