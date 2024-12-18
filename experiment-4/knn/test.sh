#!/bin/bash

# Remake
make -C src clean
make -C src all

# Clear or initialize data
> data.txt

cd src

# 定义参数范围
k_values=(10 20)
num_classes_values=(10 20)
num_training_samples_values=(10000 100000 1000000)
num_new_samples_values=(100 1000 10000)

# 遍历所有组合并执行
for k in "${k_values[@]}"; do
  for num_classes in "${num_classes_values[@]}"; do
    for num_training_samples in "${num_training_samples_values[@]}"; do
      for num_new_samples in "${num_new_samples_values[@]}"; do
        echo "Executing: ./knn-exec $k $num_classes $num_training_samples $num_new_samples"
        ./knn-exec $k $num_classes $num_training_samples $num_new_samples
      done
    done
  done
done
