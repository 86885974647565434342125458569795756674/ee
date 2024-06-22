#!/bin/bash

python_pids=$(ps -ef | grep 'python' | awk '{print $2}')

for pid in $python_pids
do
	echo "Killing python process with PID: $pid"
	kill -9 $pid
done

