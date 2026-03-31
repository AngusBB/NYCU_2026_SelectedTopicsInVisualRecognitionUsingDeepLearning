#!/bin/bash
if [ ! -f cv_hw1_data.tar ]; then
    gdown "https://drive.google.com/uc?id=1vxiXJHUo6ZPGxBGXwrsSutOpqfJ6HN9D" -O cv_hw1_data.tar
fi

if [ ! -d cv_hw1_data ]; then
    tar -xf cv_hw1_data.tar
fi
