#!/bin/bash

#set -e
#set -x

root_path=/dynamic_batch/ee/
for file_name in "$@"
do
	echo ${file_name}
	# Array of values for i
	values=(1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32)

	# Loop through each value and execute the Python script
	for i in "${values[@]}"
	do
		python ${root_path}models/test/test_${file_name}.py "$i"
	done
	
done
