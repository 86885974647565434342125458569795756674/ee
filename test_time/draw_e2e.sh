#!/bin/bash

set -e
set -x

root_path=/dynamic_batch/triton-multi-modal-serving
for file_name in "$@"
do
	if [ -f "${root_path}/${file_name}_time.txt" ]; then
		rm -rf "${root_path}/${file_name}_time.txt"
	fi
	# Array of values for i
	values=(1 2 4 8 10 12 14 16 18 20 22 24 26 28 30 32)

	# Loop through each value and execute the Python script
	for i in "${values[@]}"
	do
		python ${root_path}/demos/test_${file_name}.py "$i"
	done

	python ${root_path}/test_time/draw.py ${file_name}

	mv ${root_path}/${file_name}_time.png ${root_path}/../
done
