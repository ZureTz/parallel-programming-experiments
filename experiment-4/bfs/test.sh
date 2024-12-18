#!/bin/bash

# Remake
make -C src clean
make -C src all

# Clear or initialize data
> data.txt

cd src

# execution using different arguments
./bfs-exec 0 100000 1000000
./bfs-exec 0 100000 10000000
./bfs-exec 0 100000 100000000
./bfs-exec 0 1000000 1000000
./bfs-exec 0 1000000 10000000
./bfs-exec 0 1000000 100000000
./bfs-exec 0 10000000 10000000
./bfs-exec 0 10000000 100000000
./bfs-exec 0 10000000 100000000
