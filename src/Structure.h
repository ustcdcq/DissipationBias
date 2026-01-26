#ifndef STRUCTURE
#define STRUCTURE

#include "cuda_runtime.h"
#include "MacroDefinition.h"
#include <curand_kernel.h>
#include <cstdio>
#include <cstring>
/************************************************************************/
/* Basic struct;                                                        */
/************************************************************************/
typedef struct{
	float p[2];
	float p0[2];
	int num0[3];
	int  identity;  //particle identity : mainly for FENE or BEND calculation

	float dx[2];
}molP;
/******************/
typedef struct{
	float f[2];
}molF;
/******************/
typedef struct{
	float phi, phi0;
}molO;
/******************/
typedef struct{
	int front;     //point to the memory sequence;
	int next;
}Node;
/******************/
typedef struct{
	int cell_pos[2];
}Information;
/******************/
typedef struct{
     float s[2]; 
}S;
/************************************************************************/
/* Cell struct                                                         */
/************************************************************************/
class Cell{
public:
//Global parameters
	float box_l[2];
	int   cellsPerDim[2];
	int   cells;

	float From_min; // for frequency counting;
	float To_max;
	int   Bins;

    void Initializ_cell_parameters();
//simulation parameters  
	int   *cell_head ;
	int   *lock_flag ;
	float *device_local_density ;
	float *device_local_vx ;
	float *device_local_vy ;
	float *host_local_density ;
	float *host_local_vx ;
	float *host_local_vy ;

	int   *host_frequency_count ;

	Cell* self ;

public:
//simulation function;
	void MemoryAllocation();
	void MemoryDestroy();
	void MemoryDestroyforcopy();
	void Initialize_simulation_parameters();
	void DataTransmit_DeviceToHost();
	void Output(char *flag);

//overloadFunction
	void operator=(Cell& c);
	void Toself();

	//__host__ __device__ Cell();
	//__host__ __device__ ~Cell();
};
/************************************************************************/
/* Particle strcut;                                                     */
/************************************************************************/
class Particle{
public:
    //Global parameters
	int GeShu;
	float Activity;
	float Da;
	float tau;
	float Sg, Sg_tot, B1, B1_tot, B2, B2_tot, B3, B3_tot, B4, B4_tot;

	//some controlled flag
	int if_input_file;

	long int device_time_seed;
	long int host_time_seed;

	void Initialize_particle_parameters();
   //simulation parameters
	molP     *device_pos ;
	molF     *device_force ;
	molF* device_active_force ;
	molF* device_conven_force ;

	float* dtheta;



	molO     *device_ori ;
	Node     *device_force_cell_node ;
	Node     *device_density_cell_node ;
	Information *device_force_cell_info ;
	Information *device_density_cell_info ;
	curandState *device_seed ;
	float    *device_rand ;  //for the generation of initial configuration;

	molP     *host_pos;
	molF     *host_force;
	molO     *host_ori;
	//for the property check;
#ifdef CHECK_MIN_DIST
	float    *host_min_dist_check ;
	float    *device_min_dist_check ;
	int      *device_min_dist_check_flag ;
#endif
	int* divergency;
	int* host_div ;
	float* ratio ;

	//Sprodution
	S* S_gpu ;
	S* S_cpu ;
	//GPU pointer
	Particle* self ;

public:
    //simulation function;
	void Initialize_node();
	void Initialize_check();
	//Allocate and destroy the memory for all the variable;
	void MemoryAllocation();
	void MemoryDestroy();
	//Input and output the pos, ve and force;
	void InputFile();
	void InputFile2(char *flag);
	void InputFile3(char* flag);
	void OutputFile(char *flag);
	void OutputFile2(char* flag);
	void OutputFile3(char* flag);
	void OutputFileLammpsTraj(const char* flag, long long timestep,
		double xlo, double xhi,
		double ylo, double yhi,
		double zlo, double zhi);
	//Transmit the data between host and device
	void DataTransmit_HostToDevice();
	void DataTransmit_DeviceToHost();
	//construct
	Particle();
	~Particle();
	//calc sum of all particles epr
	float calc_sumepr(int n_step);
	void calc_ratio();
	void operator=(Particle& p);
	void Toself();
	void MemoryDestroyforcopy();
};


/************************************************************************/
/* Interaction parameters struct                                        */
/************************************************************************/
class IP{
public:
	//LJ interaction parameters
#ifdef LJ
	float sigma;
	float epsilon;
	float r_offset;
	float r_cut;
#endif

#ifdef HARMONIC
	float Harmonic_K;
	float Harmonic_offset;
	float Harmonic_cut;

	float Harmonic_K2;
	float Harmonic_offset2;
	float Harmonic_cut2;
#endif
	IP* self ;
public:
	void Toself();
	void Initialize_interaction_parameters();
	void memory_allocation();
	void memory_destroy();
	//__host__ __device__ IP();
	//__host__ __device__ ~IP();
};


/************************************************************************/
/* System parameters struct                                             */
/************************************************************************/
class SP{
public:
	float temperature;
	float gamma ;
	float timeStep;

	int n_cycle ;
	int n_per_cycle ;
	int n_cycle_warm ;
	int n_per_cycle_warm ;

	int n_cycle_density_sampling;
	SP* self ;

public:
	void Toself();
	void memory_destroy();
	void memory_allocation();
	void Initialize_system_parameters();
	//__host__ __device__ SP();
	//__host__ __device__ ~SP();
};

#endif

