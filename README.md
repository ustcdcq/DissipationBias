# Steering Active-Colloid Assembly by Biasing Dissipation



This repository implements a **clone algorithm** to bias the **dissipation tendency** of a colloidal self-assembly system, enabling **directed self-assembly**.



Core idea: by tuning the bias factor **α (alpha)**, trajectories are reweighted according to their dissipation, and a **cloning/pruning** mechanism adjusts the copy number of trajectories. This amplifies either **energy-avoiding** (low-dissipation) or **energy-seeking** (high-dissipation) pathways, thereby steering the assembly outcome (e.g., disorder/stripe/trimer).



***



## ✨ Features

- **Clone algorithm** for trajectory reweighting

- **Pathway selection and directed assembly** controlled by the bias parameter **(\alpha)**

- **Rich active-colloid assembly configurations** (e.g., disorder/stripe/trimer)

***



## 📦 Dependencies

- [CUDA](https://developer.nvidia.com/cuda-downloads) ≥ 11.2
- [Make](https://www.gnu.org/software/make/) ≥ 4.2.1
- gcc (GCC) 9.3.1
- OS: CentOS Linux 7 (Core) 7.9.2009
- OVITO ≥ 3.13

***



## 🚀 Usage



Clone the repository and compile the code:

```bash
git clone https://github.com/ustcdcq/Dissipation.git

cd src

#prepare Input and Output directory 
mkdir Input && mkdir Gen && cd Gen && mkdir Input && mkdir Output

#complie 
make -j8

#run
./CC
```

## 📺 Demo

You can tune $\alpha$ and the **`total number of replicas`** directly in `main.cu`.

To visualize the simulation trajectories, load the dump files in `/Gen/Output/` into **OVITO.**

Example output:alpha_minus10_A40_Ks220.gif


<div align="center">



 <img src="./media/alpha_down_for_40_220.gif" width="500"/>



</div>


You can find the clone-information file `info_cloneid.txt` from `/Gen/Output` directory. The file is formatted as follows: 

- column 1: `dissipation`
- column 2:`clone number`
- column 3: `father id` 
- column 4: `disorder ratio`
- column 5: `stripe ratio`
- column 6: `trimer ratio`

## 🧾 Information

A rich variety of structures with distinct functions can be generated through complex nonequilibrium self-assembly, yet directing the system toward a desired target state remains challenging because multiple dynamical pathways may coexist and fluctuations can determine the outcome. In this work, we propose a thermodynamic control principle for nonequilibrium targeted assembly in which tuning the dissipation tendency modulates the frequency and intensity of local rearrangements, thereby reshaping assembly pathways and enabling directional self-assembly when competing structures dissipate differently. 

Using the assembly of active core–corona colloids as a platform, we demonstrate two representative capabilities enabled by this principle: (i) Inducing ordered target configurations from disordered structures; (ii) Directionally selecting among competing assembly pathways.