#!/bin/bash
./train -t 8 MQ2008/train.txt 
./predict MQ2008/test.txt /home/jing/model/train.txt.model a.txt
