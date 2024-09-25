# Virtual Node Graph Neural Network for Full Phonon Prediction

This repository provides the implementation of the Virtual Node Graph Neural Network (VGNN) for full phonon prediction in materials science. VGNN is designed to address the challenges in phonon prediction using graph neural networks.

<p align="center">
  <img src="assets/vgnn.png" width="500">
</p>

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Tutorial](#tutorial)
3. [Using Pre-trained Models](#using-pre-trained-models)
4. [Training the Model](#training-the-model)
5. [Citation](#citation)
6. [References](#references)
7. [Data Availability](#data-availability)

---

## 1. Environment Setup

To set up the environment locally and run the code:

1. Clone the repository:
	```bash
	git clone https://github.com/RyotaroOKabe/phonon_prediction.git
	cd phonon_prediction_
	```

2. Create a virtual environment:
	```bash
	conda create -n pdos python=3.9
	conda activate pdos
	```

3. Install the required dependencies:
	```bash
	pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
	```
	Replace `${TORCH}` and `${CUDA}` with your specific versions (e.g., `cpu`, `cu118` for CUDA 11.8, and `2.0.0` for PyTorch 2.0). For example:
	```bash
	pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-2.0.0+cu118.html
	```

4. Start Jupyter Notebook and open the notebooks:
	```bash
	jupyter notebook
	```

---

## 2. Tutorial

You can follow the tutorial by running the provided Jupyter notebooks:

- `tutorial_VVN.ipynb`
- `tutorial_MVN.ipynb`
- `tutorial_kMVN.ipynb`

To use a trained model from scratch, specify the `pretrained_name` in the code.

Note: Tutorial codes from a previous version are stored in the `./previous_codes/` folder. To run them, move the files to the parent directory and use `tutorial_XXX_previous.ipynb`.

---

## 3. Using Pre-trained Models

To run phonon predictions on your own materials using the pre-trained models, use these notebooks:

- `cif_VVN.ipynb`
- `cif_MVN.ipynb`
- `cif_kMVN.ipynb`

Store your CIF files in the `./cif_folder/`. The Jupyter notebooks will load the materials from this folder and perform the prediction. Seekpath automatically suggest the high-symmetry path for kMVN. Use the `idx_out` variable to specify which material to plot the results for.

---

## 4. Training the Model

To train the models from scratch, use the following commands:

- Train the VVN model:
	```bash
	python train_vvn.py
	```

- Train the MVN model for Gamma phonon prediction:
	```bash
	python train_mvn.py
	```

- Train the k-MVN model for phonon band structure prediction:
	```bash
	python train_kmvn.py
	```

---

## 5. Citation

If you find this code or dataset useful, please cite the following paper:

```bibtex
@article{okabe2024virtual,
  title={Virtual node graph neural network for full phonon prediction},
  author={Okabe, Ryotaro and Chotrattanapituk, Abhijatmedhi and Boonkird, Artittaya and Andrejevic, Nina and Fu, Xiang and Jaakkola, Tommi S and Song, Qichen and Nguyen, Thanh and Drucker, Nathan and Mu, Sai and others},
  journal={Nature Computational Science},
  pages={1--10},
  year={2024},
  publisher={Nature Publishing Group US New York}
}

```

## 6. References
**Architecture:** Zhantao Chen, Nina Andrejevic, *et al.* "Virtual Node Graph Neural Network for Full Phonon
Prediction." Adv. Sci. 8, 2004214 (2021). https://onlinelibrary.wiley.com/doi/10.1002/advs.202004214.    

**E(3)NN:** Mario Geiger, Tess Smidt, Alby M., Benjamin Kurt Miller, *et al.* Euclidean neural networks: e3nn (2020) v0.4.2. https://doi.org/10.5281/zenodo.5292912.

**seekpath:** Y. Hinuma, G. Pizzi, Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on crystallography, Comp. Mat. Sci. 128, 140 (2017)  https://seekpath.readthedocs.io/en/latest/index.html.   

**Dataset:** Guido Petretto, Shyam Dwaraknath, Henrique P. C. Miranda, Donald Winston, *et al.* "High-throughput Density-Functional Perturbation Theory phonons for inorganic materials." (2018) figshare. Collection. https://doi.org/10.6084/m9.figshare.c.3938023.v1

## 7. Data Availability Statement
The data that support the findings of this study are openly available in GitHub at https://github.com/RyotaroOKabe/phonon_prediction. The $\Gamma$-phonon database generated with the MVN method is available at https://osf.io/k5utb/
