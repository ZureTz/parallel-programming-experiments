#!/bin/bash

# Remake
make -C src clean
make -C src all

# Clear or initialize data
> data.txt

cd src

# Define arrays for test parameters
vertices=(100000 1000000 10000000 50000000)
edges=(1000000 10000000 100000000 500000000)
start_vertex=0

# Nested loops for combinations
for v in "${vertices[@]}"; do
    for e in "${edges[@]}"; do
        # Only run if edges are more than vertices (valid graph)
        if [ $e -gt $v ]; then
            echo "Testing with vertices=$v edges=$e"
            ./bfs-exec $start_vertex $v $e
            sleep 1  # Small delay between tests
        fi
    done
done

# Additional specific test cases for larger graphs
large_cases=(
    "100000000 1000000000"
    "100000000 2000000000"
)

for case in "${large_cases[@]}"; do
    read -r v e <<< "$case"
    echo "Testing large graph with vertices=$v edges=$e"
    ./bfs-exec $start_vertex $v $e
    sleep 2  # Longer delay for larger tests
done
