## How to run the project

### Input Data

---

In order to start the project, a file named dataset.csv should be placed in 
the /data/dataset directory. The csv file can have the following fields:

* product_id*: Unique identifier for each product  
* product_name: Title of the product 	  
* last_sale: Last time the product was sold in year-month-day format  
* site_id: Unique identifier for each retailer's site  
* site_type_id: Discerns pharmacies and Supermarkets  
* brand_id*, category_id*, product_line_id*: The three tasks used for the classification   
* site_product_url: URL pinpointing the product in the retailer's site  
* site_image_url*: URL of the product's image in the retailer's site  
* barcoded_product_id*: A unique barcode is given to each product. Used for matching identical products  
* barcoded_product_name:  The title associated with the barcode  


*Means that the field is mandatory

### The crawler.py file

---

The file crawler.py is a script that exists separately from the project for the sake of
downloading the images from the links in dataset.csv. It stores the downloaded 
images in the /data/images directory. It can be stopped and rerun without overwriting
any of the existing images in the directory. It will also create the following files 
inside the /data/dataset directory:
  
* capstone_complete.csv  
* capstone_stats.csv  
* capstone_url.csv  
* capstone_url_res.csv  
* matching_url.csv  
* capstone_url_v1.csv

All datasets apart from the last one (capstone_url_v1.csv) exist purely for the extraction
of statistical insights and can be safely deleted if there is not enough space.

### capstone_url_v1.csv

---

If all the steps described above are done right, capstone_url_v1.csv should look
like this:

![Alt text](../sample2.png?raw=true "Title")

* product_id: Unique identifier for each product  
* site_image_url: Absolute path that points to the /data/images directory 
* brand_id, category_id, product_line_id: The three tasks used for the classification  
* height, width, channel: The three sizes of the image  
* barcoded_product_id: A unique barcode is given to each product. Used for matching
identical products  

If you already posses a dataset that looks like this, you can put it in the
/data/dataset directory as capstone_url_v1.csv in order to skip the running of crawler.py
which is time consuming.

---

You are now ready to run the project by executing cmd_interface.py desribed in the
[execution section](Execution.md)
