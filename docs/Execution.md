### Cmd interface

The cmd interface accepts commands of the style: 

action -[arg1] -[arg2] -[arg3] -[arg4] -[arg5]

The arguments are numbers that have the following functionalities:

## Arg1
0: Creates train, validation and test sets using the dataset placed in the 
/dataset folder. It is a special command and does only need Arg5.
So, a legal command would be: action -0 -100.

1: Trains the specified network for classification and stores the model

2: Evaluates the specified model and stores its probabilities and predicted classes

3: Uses a model previously trained for classification to extract embeddings from the images

4: Performs product matching and stores the predicted classes

5: Displays general statistics about the dataset given. No further arguments are needed.
It also requires the datasets created previously for statistical purposes to be intact.

6: Displays statistics about the classification process

7: Displays statistics about the matching process

## Arg2

0: Inception ResNet V2

1: NasNet Large

2: DenseNet

3: Efficientnet B3 (The most lightweight)

4: EfficientNet B5 (The best results)

## Arg3

0: Brand

1: Category

2: Product Line

3: Barcoded Product ID (warning: Will consume a lot of GPU memory)

4: Multitask (Performs classification for both brand, category, product line)

## Arg4

Batch size. Preferred to be a power of 2, e.g 32

## Arg5

Least amount of images allowed per class per task. For example Brand might have
1000 classes of products. If we input -10, only the classes that have more images 
will be picked from the dataset for Brand. 

If -0 is inputed the dataset will remain as is.

---

## Examples

Lets see the commands in action:

action -0 -10

Will produce train, valid and test sets with the restriction that each
of the three tasks must have only classes with 10+ images

action -1 -4 -2 -32 -10 

Will train EfficientNet B5 to classify Product Line with 
batch size of 32. It will request a dataset with at least 10 images per class

action -3 -3 -4 -32 -100 

Will use pretrained EfficientNet B3 Multiclass model 
to extract embeddings with batch size of 32. It will request a dataset with 
at least 100 images per class

action -6 -0 -2 -32 -0 

Will display classification statistics about an Inception ResNet V2 model 
trained for Product Line. It will request a dataset without restrictions in classes
