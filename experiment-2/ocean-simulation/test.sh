#!/bin/bash

echo "----------------------------------------------------------------------------------"

# init data
echo "Init data..."
> data.txt

# Remake
echo "Building files..."
cd src
make clean && make

echo "----------------------------------------------------------------------------------"


# Declare the possible values for d, t, and n
d_values=(18 66 258 1026 4098)
t_values=(4 16 64 256)
n_values=(1 2 4 8 16 32 64)

# First, run all serial_ocean combinations
echo "Running serial_ocean..."
for d in "${d_values[@]}"; do
  for t in "${t_values[@]}"; do
    echo "Running ./serial_ocean -d $d -t $t"
    ./serial_ocean -d "$d" -t "$t"
    echo "-----------------------------------------"
  done
done

echo "----------------------------------------------------------------------------------"

# After serial_ocean is done, run all omp_ocean combinations
echo "Running omp_ocean..."
for n in "${n_values[@]}"; do
  for t in "${t_values[@]}"; do
    for d in "${d_values[@]}"; do
      echo "Running ./omp_ocean -d $d -t $t -n $n"
      ./omp_ocean -d "$d" -t "$t" -n "$n"
      echo "-----------------------------------------"
    done
  done
done

echo "----------------------------------------------------------------------------------"
