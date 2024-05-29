# Exploring the Use of Enhanced SWAD Towards Building Learned Models that Generalize Better to Unseen Sources

Code repository available with my masters thesis for testing SWAD (Stochastic Weight Averaging Densely) a machine learning algorithm that aims at increasing a 
deep learned models accuracy on unseen sources. Specifically in this work a focus is placed on a novel Chest X-Ray dataset.
In addition to the testing of the SWAD algorithm an alteration to the SWAD algorithm called SWAD-S is also tested on this
chest X-ray dataset as well as some other well known datasets such as PACS. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

Clone this repository to your machine.
You can load in any dataset you want as long as it creates the sets the code is looking for:
train_x, train_y, val_x, val_y, test_seen_x, test_seen_y, test_unseen_x, test_unseen_y

Currently the code uses the load_CXR_data function to load data from a folder called "data"
where there are sub folders titles train, valid, test-seen, test-unseen

### Prerequisites

Install Anaconda (or miniconda as I find that to be easier)
Included in this repository is a environment file tf_gpu.yaml that contains
the packages and versions needed to run this project. Most important is the
tensorflow-gpu package that will allow the training to be done on a GPU.


### Installing


Creating Anaconda environment from yaml file:

```
conda env create -f tf_gpu.yaml
```
Activate the environment:

```
conda activate tf_gpu
```

## Start training

```
python train_cxr.py
```


## Built With

* [Tensorflow](https://www.tensorflow.org/) - Machine leanring framework used
* [Open CV](https://opencv.org/get-started/) - Image processing library used for loading images and image manipulation

## Authors

* **Brandon Weinhofer** - *Initial work* - [adopolis23](https://github.com/adopolis23)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details

