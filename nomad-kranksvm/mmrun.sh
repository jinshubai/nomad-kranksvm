#!/bin/bash
./split -p 3 -e 0.001 train.txt
./sendtotrain.py machinefile train.txt temp_dir
mpirun -n 3 --machinefile machinefile ./train -t 4 /home/jing/dis_data/train.txt.sub
./splittopredict.py machinefile test.txt
mpirun -n 3 --machinefile machinefile ./predict /home/jing/dis_data/test.txt.sub /home/jing/model/train.txt.sub.model MQ2007.txt
