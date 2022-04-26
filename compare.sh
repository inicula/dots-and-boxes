#!/bin/bash

for i in {1..20}
do
        python3 main.py --non-interactive | tail -n1
done
