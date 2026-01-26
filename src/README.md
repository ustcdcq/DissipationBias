# Steering Active-Colloid Assembly by Biasing Dissipation



This repository implements a **clone algorithm** to bias the **dissipation tendency** of a colloidal self-assembly system, enabling **directed self-assembly**.



Core idea: by tuning the bias factor **α (alpha)**, trajectories are reweighted according to their dissipation, and a **cloning/pruning** mechanism adjusts the copy number of trajectories. This amplifies either **energy-avoiding** (low-dissipation) or **energy-seeking** (high-dissipation) pathways, thereby steering the assembly outcome (e.g., disorder/stripe/trimer).



***



## ✨ Features

- **Clone algorithm** for trajectory reweighting
- Pathway selection and directed assembly controlled by the bias parameter **α**

***



## 📦 Dependencies

- [CUDA](https://developer.nvidia.com/cuda-downloads) ≥ 11.2

- [Make](https://www.gnu.org/software/make/) ≥ 4.2.1

-  gcc (GCC) 9.3.1

- OS: CentOS Linux 7 (Core) 7.9.2009

- OVITO ≥ 3.13

***



## 🚀 Usage



Clone the repository and compile the code:

```bash
git clone https://github.com/ustcdcq/Dissipation.git

cd src

make -j8

./CC
```

# 📺 Demo

You can load the following files from the **`Output`** directory into `OVITO` to visualize the simulation trajectory.

file format:`A_40.0_Ks_220.0_clone_id.dump`

