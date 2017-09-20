#!/bin/bash

#FROM=2000000
#STEP=2000000
#TO=20000000

#FROM=1000
#STEP=1000
#TO=1000000


#for i in $(seq -f %.0f $FROM $STEP $TO); do
##for i in 20000 200000 2000000 20000000; do
#  if [[ ! -f "./data$i.txt" ]]; then
#    echo "Generating input file data$i.txt"
#    ./build/bin/generator -a $i > data$i.txt;
#  fi
#done

#for i in $(seq -f %.0f $FROM $STEP $TO); do
#for i in 20000 100000 2000000 20000000; do
echo "Measuring indexing VAST"
for i in 20000 100000 $(seq -f %.0f 250000 250000 20000000); do
  #if [[ -f "./measurement$i.txt" ]]; then
    #echo "File \"measurement$i.txt\" exists, deleting it."
    #rm -f "./measurement$i.txt"
  #fi
  for j in $(seq 1 1 10); do
    rm -f "./data$i.txt"
    ./build/bin/generator -a $i > data$i.txt;
    ./aggregate.sh -e ./build/bin/vst -f ./data$i.txt -i 1 -o measurement-vast-$i-$j.txt;
    #./aggregate.sh -e ./build/bin/phases -f ./data$i.txt -i 1 -o measurement$i.txt;
  done
done

echo "Measuring indexing with OpenCL actors"
for i in 20000 100000 $(seq -f %.0f 250000 250000 20000000); do
  #if [[ -f "./measurement$i.txt" ]]; then
    #echo "File \"measurement$i.txt\" exists, deleting it."
    #rm -f "./measurement$i.txt"
  #fi
  for j in $(seq 1 1 10); do
    rm -f "./data$i.txt"
    ./build/bin/generator -a $i > data$i.txt;
    #./aggregate.sh -e ./build/bin/vst -f ./data$i.txt -i 1 -o measurement-vast-$i-$j.txt;
    ./aggregate.sh -e ./build/bin/phases -f ./data$i.txt -i 1 -o measurement-opencl-$i-$j.txt;
  done
done

## Next three things are just for the 2 to 20 million measuremnet with all steps
#for i in $(seq -f %.0f 2000000 2000000 20000000); do
  #paste measurement$i-1.txt measurement$i-2.txt measurement$i-3.txt measurement$i-4.txt measurement$i-5.txt measurement$i-6.txt measurement$i-7.txt measurement$i-8.txt measurement$i-9.txt measurement$i-10.txt > aggregate-$i.txt
#done

#for i in $(seq -f %.0f 2000000 2000000 20000000); do
  #awk '{print ($1+$2+$3+$4+$5+$6+$7+$8+$9+$10)/10}' aggregate-$i.txt > mean-$i.txt
#done

#paste mean-20000000.txt mean-18000000.txt mean-16000000.txt mean-14000000.txt mean-12000000.txt mean-10000000.txt mean-8000000.txt mean-6000000.txt mean-4000000.txt mean-2000000.txt > measurements20m.txt


#### OLD CODE ####

#FROM=1000
#STEP=1000
#TO=16000

#for i in $(seq -f %.0f $FROM $STEP $TO); do
#  if [[ ! -f "./data$i.txt" ]]; then
#    echo "Generating input file data$i.txt"
#    ./build/bin/generator -a $i > data$i.txt;
#  fi
#done

#for i in $(seq -f %.0f $FROM $STEP $TO); do
#  if [[ -f "./measurement$i.txt" ]]; then
#    echo "File \"measurement$i.txt\" exists, deleting it."
#    rm -f "./measurement$i.txt"
#  fi
#  ./aggregate.sh -e ./build/bin/phases -f ./data$i.txt -i 10 -o measurement$i.txt;
#done
