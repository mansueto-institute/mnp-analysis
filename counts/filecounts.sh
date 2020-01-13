#!/bin/bash

cd /project2/bettencourt/mnp/prclz/data/complexity

# https://unix.stackexchange.com/a/260636
total=0
files=0
find . -type f -name "*.csv" | while read FILE; do 
  count=$(grep -c ^ < "$FILE")
  let total=total+count
  let files=files+1
  echo $files
done 
net_blocks=$((total-files)) # adjust for headers
echo "num GADMs : ${files}"
echo "num blocks: ${net_blocks}"
