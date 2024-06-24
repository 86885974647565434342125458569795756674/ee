#!/bin/bash

#set -e
#set -x

root_path=/dynamic_batch/ee/
for file_name in "$@"
do
	# Array of values for i
	values=(2 4 8 16 32)

	# Loop through each value and execute the Python script
	for i in "${values[@]}"
	do
		python ${root_path}models/test/test_${file_name}.py "$i"
	done
	
done
