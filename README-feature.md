

<!--
 * @Author: Rui Wang
 * @Date: 2025-02-10 22:44:41
 * @LastModifiedBy: Rui Wang
 * @LastEditTime: 2025-07-17 14:55:54
 * @Email: wang.rui@nyu.edu
 * @FilePath: /TopoDockQ/README-feature.md
 * @Description: 
-->
# TopoDockQ-Feature
This Python package generates persistent combinatorial Laplacian-based features for the TopoDockQ model.

## Requirements

- python=3.8.1
- numpy<=1.24.3
- pandas<=2.2.0
- scikit-learn=1.3.0
- gudhi=3.8.0

## Install and Build

```bash
conda env create -f environment.yml
```

## Run Code

#### Example usage:

```bash
python -m main --pdb_id PDBID --model_id MODELID --bins BINS --filtration FILTRATION --file_path FILEPATH --saving_path SAVINGPATH
```

- PDBID: A string, the Protein Data Bank ID. For example, 4k38
- MODELID: An int, the ID of the model. For example, 44.
- BINS: A string records bins' starting and ending points in a list format.
- FILTRATION: A string, records various distance-based filtration values. 
- FILEPATH: A string shows the directory of the protein-peptide data.
- SAVINGPATH: A string, indicates where the output feature file will be saved. 

####  An Example to run:

```bash
python -m main --pdb_id 4k38 --model_id 44 --bins "[0, 2.,  2.25,  2.5 ,  2.75,  3.  ,  3.25,  3.5 ,  3.75,  4., 4.25,  4.5 ,  4.75,  5.]"  --filtration "[0, 2.,  2.25,  2.5 ,  2.75,  3.  ,  3.25,  3.5 ,  3.75,  4., 4.25,  4.5,  4.75,  5.]"  --file_path ./data/interface_files --saving_path ./feature
```

#### Output file

Features will be saved in `./feature`, named `feature_4k38_ranked_44_sp_interface.npy`. 


## Documentation 

Documentation for TopoDockQ can be found [here](https://github.com/XDaiNYU/TopoDockQ).

## Citing
#### Please use the BibTeX entry to cite TopoDockQ:

```bibtex
@article{dai2025topological,
  title={Topological Deep Learning for Enhancing Peptide-Protein Complex Prediction},
  author={Dai, Xuhang and Wang, Rui and Zhang, Yingkai},
  journal={in review},
  year={2025}
}
```

#### Please use the BibTex entry to cite Persistent combinatorial Laplacians:

```bibtex
@article{wang2021hermes,
  title={HERMES: Persistent spectral graph software},
  author={Wang, Rui and Zhao, Rundong and Ribando-Gros, Emily and Chen, Jiahui and Tong, Yiying and Wei, Guo-Wei},
  journal={Foundations of data science (Springfield, Mo.)},
  volume={3},
  number={1},
  pages={67},
  year={2021},
  publisher={NIH Public Access}
}
```

## References
- R. Wang, R. Zhao, E. Ribando-Gros, J. Chen, Y. Tong, and G.-W. Wei. [HERMES: Persistent spectral graph software](https://www.aimsciences.org/article/doi/10.3934/fods.2021006), _Foundations of Data Science_, 2021.
- R. Wang, D. D. Nguyen, and G.-W. Wei. [Persistent spectral graph](https://users.math.msu.edu/users/weig/paper/p243.pdf), _International Journal for Numerical Methods in Biomedical Engineering_, page e3376, 2020.
