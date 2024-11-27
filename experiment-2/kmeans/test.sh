#!/bin/bash

# 文件名：run_kmeans_serial.sh

echo "----------------------------------------------------------------------------------"

# init data
echo "Init data..."
> data.txt

# Remake
echo "Building files..."
cd src
make clean && make

echo "----------------------------------------------------------------------------------"

# 定义参数范围
s_values=(10000 100000 1000000)
k_values=(2 4 8 16)
n_values=(1 2 4 8 16 32 64)

# 循环遍历所有参数组合
for s in "${s_values[@]}"; do
    for k in "${k_values[@]}"; do
        echo "Running: ./kmeans-serial -s $s -k $k"
        ./kmeans-serial -s $s -k $k
        echo "-----------------------------------------"
    done
done

echo "----------------------------------------------------------------------------------"

# 循环遍历所有参数组合
for n in "${n_values[@]}"; do
    for s in "${s_values[@]}"; do
        for k in "${k_values[@]}"; do
            echo "Running: ./kmeans-parallel -s $s -k $k -n $n"
            ./kmeans-parallel -s $s -k $k -n $n
            echo "-----------------------------------------"
        done
    done
done

echo "----------------------------------------------------------------------------------"
