#!/bin/bash
./split -p 3 -e 0.001 bodyfat_scale_qid
./sendtotrain.py machinefile bodyfat_scale_qid temp_dir
mpirun -n 3 --machinefile machinefile ./train -t 2 /home/jing/dis_data/bodyfat_scale_qid.sub 
