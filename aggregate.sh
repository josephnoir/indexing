#!/bin/bash

# Executes a script in an interation and builds rows from the output values

usage="\
Usage: $0 -e EXECUTABLE [OPTIONS]
 Options:
  -e | --executable             Binary to execute (required).
  -f | --input-file             File with input data (passed using -f).
  -i | --iterations             Iterations to execute.
  -o | --output-file            File the output data is written to.

  -h | --help                   Print this text.
"

ITERATIONS=1
ARGUMENT_TEXT="-f"
TRUCANATOR=" us"
EXECUTABLE="../build/bin/phases"

while [[ $# -gt 1 ]]
do
key="$1"

case $key in
  -e|--executable)
  EXECUTABLE="$2"
  shift # past argument
  ;;
  -f|--input-file)
  INPUTFILE="$2"
  shift # past argument
  ;;
  -i|--iterations)
  ITERATIONS="$2"
  shift # past argument
  ;;
  -o|--output-file)
  OUTPUTFILE="$2"
  shift # past argument
  ;;
  -h|--help)
  echo "${usage}" 1>&2
  exit 1
  ;;
  *)
  echo "Unknown argument: $key".
  echo "${usage}" 1>&2
  exit 1
  ;;
esac
shift # past argument or value
done

# Don't try to run without an executable
if [[ -z "$EXECUTABLE" ]]; then
  echo "An execuatable to perform the tests is required."
  echo "${usage}" 1>&2
  exit 4
fi

# Check if executable exists
if [[ ! -f $EXECUTABLE ]]; then
  echo "$EXECUTABLE does not exist."
  echo "${usage}" 1>&2
  exit 4
fi

# Create arguments if input file exists
if [[ ! -z "$INPUTFILE" ]]; then
  if [[ ! -f $INPUTFILE ]]; then
    echo "$INPUTFILE is not a valid file."
    echo "${usage}" 1>&2
    exit 4
  fi
  ARGUMENTS="$ARGUMENT_TEXT $INPUTFILE"
fi

# Create a default output file if no name is given
if [[ -z "$OUTPUTFILE" ]]; then
  DATE=`date +%Y-%m-%d-%H-%M-%S`
  OUTPUTFILE="measurements-$DATE.txt"
  echo "writing to $OUTPUTFILE"
fi

ROWS=0
# for i in {1.."$ITERATIONS"}; do
for ((i = 1; i <= $ITERATIONS; i++)); do
  # echo "$i $EXECUTABLE $ARGUMENTS"
  # RESULT="$RESULT $($EXECUTABLE $ARGUMENTS)"
  # echo $RESULT
  if [[ $i -eq 1 ]]; then
    RESULT="$($EXECUTABLE $ARGUMENTS)"
    SPLITS=$(echo $RESULT | tr "us" "\n")
    for j in $SPLITS; do
      ((ROWS+=1))
    done
    # echo "$ROWS rows"
  else
    RESULT="$RESULT $($EXECUTABLE $ARGUMENTS)"
  fi
done

ROW=0
SPLITS=$(echo $RESULT | tr "$TRUCANATOR" "\n")
for VAL in $SPLITS; do
  LINE=OUTPUT_ROW_$ROW 
  eval TMP=\$$LINE
  # Prevent space at the start of each row
  if [[ -z "$TMP" ]]; then
    TMP="$VAL"
  else
    TMP="$TMP $VAL"
  fi
  eval $LINE="\$TMP"
  ((ROW+=1))
  ROW=$(($ROW % $ROWS))
done


for ((i = 0; i < $ROWS; i++)); do
  LINE=OUTPUT_ROW_$i
  eval RES=\$$LINE
  echo $RES >> $OUTPUTFILE
  #SUM=0
  #for VAL in $RES; do
    #((SUM+=$VAL))
  #done
  #MEAN=$(echo "scale=3; $SUM / $ITERATIONS" | bc)
  #echo "$MEAN" >> $OUTPUTFILE
done

