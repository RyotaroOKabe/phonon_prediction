# Virtual Node Graph Neural Network for Full Phonon Prediction
We present the virtual node graph neural network (VGNN) to address the challenges in phonon prediction. 

#### Work from a local installation
To work from a local copy of the code:

1. Clone the repository:
	> `git clone https://github.com/RyotaroOKabe/phonon_prediction.git`

	> `cd phonon_prediction_`

2. Create a virtual environment for the project:
	> `conda create -n pdos python=3.9`

	> `conda activate pdos`

3. Install all necessary packages:
	> `pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html`

	where `${TORCH}` and `${CUDA}` should be replaced by the specific CUDA version (e.g. `cpu`, `cu102`) and PyTorch version (e.g. `1.10.0`), respectively. For example:

	> `pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html`

4. Run `jupyter notebook` and open `VVN.ipynb` or `kMVN.ipynb`.

### References
**Publication:** Zhantao Chen, Nina Andrejevic, Tess Smidt, *et al.* "Virtual Node Graph Neural Network for Full Phonon
Prediction." Journal. (2023): XXXX. "URL".

**E(3)NN:** Mario Geiger, Tess Smidt, Alby M., Benjamin Kurt Miller, *et al.* Euclidean neural networks: e3nn (2020) v0.4.2. https://doi.org/10.5281/zenodo.5292912.

**Dataset:** Guido Petretto, Shyam Dwaraknath, Henrique P. C. Miranda, Donald Winston, *et al.* "High-throughput Density-Functional Perturbation Theory phonons for inorganic materials." (2018) figshare. Collection. https://doi.org/10.6084/m9.figshare.c.3938023.v1

### Data Availability Statement
The data that support the findings of this study are openly available in GitHub at https://github.com/RyotaroOKabe/phonon_prediction. The $\Gamma$-phonon database generated with the MVN method is available at https://osf.io/k5utb/ 

