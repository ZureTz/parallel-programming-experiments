#! /bin/bash

# Recommeneded usage: ./test.sh | tee test-result.log

echo '------------------------------------------------------------------'
echo 'Making again ...'
make clean
make all
echo 'Remake finished'
echo '------------------------------------------------------------------'

echo ''

echo '------------------------------------------------------------------'
echo 'Shellsort Performance: '
for arrayLength in 5000000 10000000 30000000; do
  echo '---------------------------------'
  echo "arrayLength = $arrayLength"
  echo "Without parallelism performance:"
  ./shellsort -s ${arrayLength}
  echo '---------------------------------'
  echo "With parallelism performance:"
  for numThreads in 1 2 4 8 16 32 64; do
    echo "Using $numThreads threads: "
    ./shellsort-parallel -s ${arrayLength} -n ${numThreads}
  done
  echo '---------------------------------'
done
echo '------------------------------------------------------------------'
echo 'Radixsort Performance: '
for arrayLength in 5000000 10000000 30000000; do
  echo '---------------------------------'
  echo "arrayLength = $arrayLength"
  echo "Without parallelism performance:"
  ./radixsort -s ${arrayLength}
  echo '---------------------------------'
  echo "With parallelism performance:"
  for numThreads in 1 2 4 8 16 32 64; do
    echo "Using $numThreads threads: "
    ./radixsort-parallel -s ${arrayLength} -n ${numThreads}
  done
  echo '---------------------------------'
done
echo '------------------------------------------------------------------'
