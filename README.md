# The source code for "PISD: A linear complexity distance beats dynamic time warping on time series classification and clustering" accepted by Engineering Applications of Artificial Intelligence Journal (IF 7.5).


# Method

The process to calculate PISD between two time series Q and C. We detect PIPs, followed by extracting PISs and PCSs for Q and C. Then, we compute SubDist between each PIS and its PCS, where SQi is PSubDist(QPIS_i,CPCS_i) and SCi is PSubDist(CPIS_i,QPCS_i). Finally, we use those PSubDist to calculate the PISD of Q and C

![alt text](https://github.com/tmtuan1307/PISD/blob/main/img/pisd3.jpg)

This chart compares the average rank and prediction time for a single time series of all the tested methods. It is clear that our PISD is the first method to be both quicker and more accurate than DTWCV, the most common variant of DTW.

![alt text](https://github.com/tmtuan1307/PISD/blob/main/img/tvsr2.jpg)


# Dataset

Dataset: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

Please download dataset and put it in \dataset\. 

# Dependencies: 

Python 3.8 and above

Numpy 1.19.2 and above

Scipy 1.7.2 and above

Please install the below library before running the code.

pip install numpy

pip install scipy

pip install scikit-learn

# Usage: 

You can run the command: 

Original Version PISD:
```
python original_pisd/pisd.py --dataset_pos 2
python original_pisd/pisd_f.py --dataset_pos 2
```

or

```
python original_pisd/pisd.py --dataset_name ArrowHead
python original_pisd/pisd_f.py --dataset_name ArrowHead
```


Speed Up Version PISD:

```
python speedup_pisd/pisd.py --dataset_pos 2
python speedup_pisd/pisd_f.py --dataset_pos 2
```

or

```
python speedup_pisd/pisd.py --dataset_name ArrowHead
python speedup_pisd/pisd_f.py --dataset_name ArrowHead
```

k-PISA:

```
python k_pisa/k_pisa.py --dataset_pos 2
```

or

```
python k_pisa/k_pisa.py --dataset_name ArrowHead
```

# Classification Result
You can see the full results on 112 UCR datasets in `results/`
