#!/bin/sh
conda env create -f environment.yml
conda init bash
conda activate parallelMatricesCalcs
i=1
while [ $i -le 32 ]; do
  python main.py $i
  i=$(( i + 1 ))
done

python plotter.py
rm -f ./results.csv