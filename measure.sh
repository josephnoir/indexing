#!/bin/bash

for i in $(seq 20000 20000 200000); do
  if [[ ! -f "./data$i.txt" ]]; then
    echo "Generating input file data$i.txt"
    ./build/bin/generator -a $i -t 65535 > data$i.txt;
  fi
done

for i in $(seq 20000 20000 200000); do
  if [[ -f "./measurement$i.txt" ]]; then
    echo "File \"measurement$i.txt\" exists, deleting it."
    rm -f "./measurement$i.txt"
  fi
  ./aggregate.sh -e ./build/bin/phases -f ./data$i.txt -i 512 -o measurement$i.txt;
done
