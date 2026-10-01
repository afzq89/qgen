# PHSM-QGEN

This repository provides a simple workflow for estimating along-the-path kinetic and thermodynamic properties from molecular simulation:

An example workflow is demonstrated in `rate_qgen-notebook.ipynb`.

The notebook is currently using unbiased simulation data. 

## Available rate estimators 

The following generator estimators can be selected in the notebook:

- `DA`: diagonal adjustment
- `WA`: Weighted Adjustment
- `PA`: polynomial adjustment
- `CWO`: Component-Wise Optimization
- `QOG`: Quasi-optimization of the Generator
- `KL`: Kalbfleisch and Lawless estimator from deeptime package
- `CVE`:  Crommelin and Vanden-Eijnden estimator from deeptime package
- `MLE-pi`: maximum-likelihood estimator with a fixed free-energy profile

The existing estimator implementations in `generators.py` and `estimator.py` are used directly.

## Installation

Create the Conda environment:

conda env create -f QGEN.yml
conda activate QGEN

## Compile PHSM

Compile PHSM before running the notebook:

cd phsm
make
cd ..

A C++11-compatible compiler is required.

## Example

An example Müller–Brown demonstration data is provided:

```text
example/colvars.npy
example/centers2D.txt
```

The `colvars.npy` stored data as following

```text
x-coordinate, y-coordinate, weight
```
Weights are all initially equal. 

`centers2D.txt` contains the initial pathway.

## Running the notebook

Run the cells in `rate_qgen-notebook.ipynb` in order.

The main input section specifies:

- coordinate and initial-path files
- PHSM executable
- number of PHSM centers
- `SIGMA`
- `DELTA` through `R_values`
- `THICKNESS` through `tang_values`
- `EPSILON`
- PHSM convergence parameters
- temperature and simulation time step
- lag times
- number of bootstrap samples
- initial and target MFPT states
- generator estimator

## Output

Results are written as:

```text
rate_results/<estimator>_<R>_<thickness>/
```

The main output files are:

```text
fe_<R>_<thickness>_<lag>.txt
D_<R>_<thickness>_<lag>.txt
mfpt_<R>_<thickness>.txt
cluster.npy
phsm.path
phsm.smooth
phsm.smooth.status
```

The notebook also plots:

- free-energy profiles for all lag times
- diffusion profiles for all lag times
- MFPT using FE and 1D diffusion versus lag
- Poisson MFPT versus lag

## Notes

PHSM assignments equal to `-1` are treated as trajectory separators. Transitions are not counted across rejected configurations.


## References
