#!/bin/bash

# Remake
make -C src clean
make -C src all

# Clean data
> data.txt

cd src

# 定义取值范围
np_values=(1 2 4 8 16)
dim_values=(1024 2048 4096 8192)
timesteps_values=(50 100 200 500)

# 循环遍历所有组合
for np in "${np_values[@]}"; do
  for dim in "${dim_values[@]}"; do
    for timesteps in "${timesteps_values[@]}"; do
      echo "Running mpirun -np $np ./cg $dim $timesteps"
      mpirun -np $np ./cg $dim $timesteps 2> >(grep -v -E 'Authorization|yaksa')
    done
  done
done
