#!/bin/bash

STARTTIME=$(date +%s)
python3 all.py a | tee model/a.out
ENDTIME=$(date +%s)
echo "A takes $((($ENDTIME - $STARTTIME) / 60)) minutes..."

STARTTIME=$(date +%s)
python3 all.py b | tee model/b.out
ENDTIME=$(date +%s)
echo "A takes $((($ENDTIME - $STARTTIME) / 60)) minutes..."

STARTTIME=$(date +%s)
python3 all.py c | tee model/c.out
ENDTIME=$(date +%s)
echo "A takes $((($ENDTIME - $STARTTIME) / 60)) minutes..."
