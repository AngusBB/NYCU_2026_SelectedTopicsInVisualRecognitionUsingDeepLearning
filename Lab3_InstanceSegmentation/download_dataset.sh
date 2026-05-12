#!/bin/bash

pip install gdown

if [ ! -f "hw3-data-release.tar" ]; then
    gdown "https://drive.google.com/uc?id=1uCnJ3LrsBHOeQoJDoe4Yg8H32VuQJodv" -O "hw3-data-release.tar"
fi

if [ ! -d "hw3-data-release" ]; then
    tar -xf "hw3-data-release.tar"
fi

mkdir -p "data"
mv "train" "data/train"
mv "test_release" "data/test"
mv "test_image_name_to_ids.json" "data/test_image_name_to_ids.json"