# Scalable and Globally Optimal Generalized L_1 k-center Clustering via Constraint Generation in Mixed Integer Linear Programming


This is the official implementation of our paper [camera ready pdf here](/AAAI_2023_ChembuA.pdf) accepted to be presented at AAAI 2023. (Link will be updated soon)

If you would like to cite this work, a sample bibtex citation is as following:


```
@inproceedings{chembu:aaai23,
	author = {Aravinth Chembu and Scott Sanner and Hassan Khurram and Akshat Kumar},
	title = {Scalable and Globally Optimal Generalized L1 k-center Clustering via Constraint Generation in Mixed Integer Linear Programming},
	year = {2023},
	booktitle = {Proceedings of the 37th {AAAI} Conference on Artificial Intelligence ({AAAI-23})},
	address = {Washington D.C., USA}
}
```

## Setup 

First, install the requirements

```
pip install -r requirements.txt
```

Note that using Gurobi requires an [academic](https://www.gurobi.com/academia/academic-program-and-licenses/) or [evaluation](https://www.gurobi.com/downloads/request-an-evaluation-license/) license.



Second, setup the directory to run the codes

* Save the contents of the downloaded and unzipped folders into a single parent folder *your_folder*
* Contents include:
    * *kc-opt* folder containing all the codes needed to run the experiments 
    * *Datasets* folder containing data files for the real datasets 
* From within *your_folder/kc-opt*, run the config.py to make directories to save the results


<br>

## Experiments with synthetic data


All the experiments with synthetic data including plots used in the paper, construction of the three synthetic datasets *Norm, Norm-Imb* and *Norm-Out*, and results from these datasets from the baselines and our kc-Opt and kc-OptOut are in the Jupyter-notebooks organized in the [SyntheticDataExpts](/SyntheticDataExpts/) folder.

* [Constraint_generation_example](/SyntheticDataExpts/Constraint_generation_example.ipynb) for Figure 2
* [IntroPlots_UnbalancedData](/SyntheticDataExpts/IntroPlots_UnbalancedData.ipynb) for Figure 1
* [OutliersHandle_PlotExample](/SyntheticDataExpts/OutliersHandle_PlotExample.ipynb) for Figure 6
* [Syn1_WS_NORM_CompareAlgosPlots](/SyntheticDataExpts/Syn1_WS_NORM_CompareAlgosPlots.ipynb), [Syn1_WS_NORM_MinMaxClustering](/SyntheticDataExpts/Syn1_WS_NORM_MinMaxClustering.ipynb) for Figures 3, 4 (a), 4 (b), 7, 8 and Figure 9 
* [Syn2_NORM_Imb_ComparePlot](/SyntheticDataExpts/Syn2_NORM_Imb_ComparePlot.ipynb) for Figure 11
* [Syn3_Norm_Out_Dataset1_Plot](/SyntheticDataExpts/Syn3_Norm_Out_Dataset1_Plot.ipynb), [Syn3_Norm_Out_Dataset2](/SyntheticDataExpts/Syn3_Norm_Out_Dataset2.ipynb) for Figure 5
* [Syn4_WS_NORM_AnisotropicPlots](/SyntheticDataExpts/Syn4_WS_NORM_AnisotropicPlots.ipynb) for Figure 10

<br>

## Experiments with real datasets

All the experiments for real datasets can be executed with the codes in [RealDatasetExpts](/RealDatasetExpts/) and the saved results can be extracted and organized with the [AllResultsCollect_toLatex](/RealDatasetExpts/AllResultsCollect_toLatex.ipynb) Jupyter-notebook.

