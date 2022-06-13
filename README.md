*See the [Workshop Webpage](https://www.primate-cognition.eu/de/veranstaltungen/bridging-the-technological-gap-workshop.html) for the context of this tutorial.* <br>
*See the [Workshop ML part Confluence Page](https://ccp-eva.atlassian.net/wiki/external/231442020/NzM0MTJiYzVlZjk5NDJlZWJiYzY3ZTNhZWQyZDhlOTU?atlOrigin=eyJpIjoiMmJlM2Q0NzY3MTM2NGI0NGIyMmIzYjMxMDc3Y2RhNGMiLCJwIjoiYyJ9) for initial installation steps and further interactive content.*

# Introduction

The gap between what is achieve in the computer vision field, and the computer vision tools used in other fields, exists because of the lack of communication between the two fields. Thanks to this workshop and the proposed hand-on session, we hope to fill this gap in an funny and interactive way.

In this tutorial, we shall solve a simple classification task with one provided dataset, and if you have time, your own dataset! Let's hope you can do the same with a more serious scientific question in your future projects!

# Installation

Before coming to the workshop, make sure to set your computer or borow a computer. 64bit computer is requiered. GPU is advised. See the [Workshop ML part Confluence Page](https://ccp-eva.atlassian.net/wiki/external/231442020/NzM0MTJiYzVlZjk5NDJlZWJiYzY3ZTNhZWQyZDhlOTU?atlOrigin=eyJpIjoiMmJlM2Q0NzY3MTM2NGI0NGIyMmIzYjMxMDc3Y2RhNGMiLCJwIjoiYyJ9).

Then in your terminal, follow the following steps:

``` bash
# Clone baseline repo via command line or manually using the browser: download zip file and extract file.
git clone https://github.com/ccp-eva/tuto_classification.git
cd tuto_classification
```

You should have downloaded the github repo. This repo should contain several python file. In order to all have the same installation environment, we will use conda that you have installed previously, and the provided `.yml` files

``` bash
# Create the conda environment (you can change the name and location)
conda env create --prefix ./env --file environment_with_versions.yml # or environment.yml
# Activate the conda environment
conda activate ./env
```

# The first steps

<!--
For organization:
``` bash
# to create md5sums files
find -type f -exec md5sum "{}" + > MD5SUMS
# to create env .yml files
conda env export --name ME22_env --file environment_with_versions.yml
conda env export --name ME22_env --from-history --file environment.yml
```
-->


The next steps may be done before the woorkshop and any bug/difficulties may be reported to the organizers.

``` bash
# Extract frames from videos
python 1_create_database.py
# Split your databse
python 2_split_database.py
# Run classification on you database split framewisely
python 3_cnn_classification.py my_dataset_framewise_split
# Run classification on you database split videowisely
python 3_cnn_classification.py my_dataset_framewise_split
```

These steps should create several folders:
- a log folder for script 1 and 2.
- a few images to represent your datasets.

- cnn_classification_output folder for script 3. It should contain a folder for each call of the function. The latest should contain the weights of the model your just trained on the given dataset, and some images representing its performance.

Finally, use one of the output folder of script 3 to run the demo app. It shall use your camera and indicate the infered class of your model.

```
# Run your app
python 4_run_app.py [path_of_you_trained_model_folder]

```


# Further steps

Further steps will be conducted at the workshop. If time allows it, we may create our own dataset and our own task we would like to solve.

Thank you for your participation!

