#!/bin/bash

for repeat in $(seq 0 1000);
do
for env in Copy-v0 DuplicatedInput-v0 RepeatCopy-v0 Reverse-v0;
do
    for tau in `seq 0.1 0.1 1.0`;
    do
	sbatch run_generalTsallisv2_q_2.sh $env 5000 $tau
    done
done
done


for repeat in $(seq 0 1000);
do
for env in Copy-v0 DuplicatedInput-v0 RepeatCopy-v0 Reverse-v0;
do
    for tau in `seq 0.01 0.01 0.1`;
    do
	sbatch run_generalTsallisv2_q_2.sh $env 5000 $tau
    done
done
done


for repeat in $(seq 0 1000);
do
for env in Copy-v0 DuplicatedInput-v0 RepeatCopy-v0 Reverse-v0;
do
    for tau in `seq 0.001 0.001 0.01`;
    do
	sbatch run_generalTsallisv2_q_2.sh $env 5000 $tau
    done
done
done


for repeat in $(seq 0 1000);
do
for env in Copy-v0 DuplicatedInput-v0 RepeatCopy-v0 Reverse-v0;
do
    for tau in `seq 0.0001 0.0001 0.001`;
    do
	sbatch run_generalTsallisv2_q_2.sh $env 5000 $tau
    done
done
done


for repeat in $(seq 0 1000);
do
for env in Copy-v0 DuplicatedInput-v0 RepeatCopy-v0 Reverse-v0;
do
    for tau in `seq 0.00001 0.00001 0.0001`;
    do
	sbatch run_generalTsallisv2_q_2.sh $env 5000 $tau
    done
done
done