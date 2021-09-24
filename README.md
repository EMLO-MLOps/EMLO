# Example: Versioning 

Datasets and ML model getting started
[versioning tutorial](https://dvc.org/doc/tutorials/versioning).


### To run model
> python train.py


### Project Skeleton

> git clone https://github.com/iterative/example-versioning.git
> cd example-versioning


### Virtual Environment
> python3 -m venv .env
> source .env/bin/activate
> pip install -r requirements.txt


### Install & Initialize DVC (MAC)
> brew install dvc


## First Model Version

> dvc get https://github.com/iterative/dataset-registry \
          tutorials/versioning/data.zip
> unzip -q data.zip
> rm -f data.zip


#### Data Directory Structure
data
├── train
│   ├── dogs
│   │   ├── dog.1.jpg
│   │   ├── ...
│   │   └── dog.500.jpg
│   └── cats
│       ├── cat.1.jpg
│       ├── ...
│       └── cat.500.jpg
└── validation
   ├── dogs
   │   ├── dog.1001.jpg
   │   ├── ...
   │   └── dog.1400.jpg
   └── cats
       ├── cat.1001.jpg
       ├── ...
       └── cat.1400.jpg
       
       
### Add data with DVC
> dvc add data


### Add trained model weights
> dvc add model.h5


### Commit Current State
> git add data.dvc model.h5.dvc metrics.csv .gitignore
> git commit -m "First model, trained with 1000 images"
> git tag -a "v1.0" -m "model v1.0, 1000 images"


## Second Model Version
> dvc get https://github.com/iterative/dataset-registry \
          tutorials/versioning/new-labels.zip
> unzip -q new-labels.zip
> rm -f new-labels.zip


### Commit second version
> git add data.dvc model.h5.dvc metrics.csv
> git commit -m "Second model, trained with 2000 images"
> git tag -a "v2.0" -m "model v2.0, 2000 images"


## Switching between workspace versions

#### Full Workspace Checkout
> git checkout v1.0
> dvc checkout

#### Checkout specific data
> git checkout v1.0 data.dvc
> dvc checkout data.dvc


## Automated Capturing
> dvc run -n train -d train.py -d data \
          -o model.h5 -o bottleneck_features_train.npy \
          -o bottleneck_features_validation.npy -M metrics.csv \
          python train.py
