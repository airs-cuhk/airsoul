#!/bin/bash

function run_diff {
nohup python evaluate_diff.py \
	--task_para ~/Data3/CartPole/CP_Train_$1/ \
	--data ~/Data3/CartPole/CP_Test_$2/record_0$3/ \
	--output dist_"$1$2$3" > log.caldist."$1$2$3" &
}
function run_diff_num {
	echo "Start Processing $1"
	run_diff C B $1
	run_diff D B $1
	run_diff E B $1
	run_diff C C $1
	run_diff D C $1
	run_diff E C $1
}

run_diff_num $1
