#include "RandomNumber.h"
#include "MacroDefinition.h"
#include "Structure.h"
#include <curand_kernel.h>
#include "cuda_runtime.h"
//#include <curand.h>
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "curand.lib")

__global__ void Device_set_seed (Particle colloid)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x ;

	if (tid < colloid.GeShu*ENSEMBLE)
	{
		curand_init(tid + colloid.device_time_seed, 0, 0, colloid.device_seed+tid);
	}
}

void Generate_Random_Number(Particle colloid)
{
	curandGenerator_t gen;  
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) ;  
	curandSetPseudoRandomGeneratorSeed(gen, colloid.host_time_seed) ;  
	curandGenerateUniform(gen, colloid.device_rand, colloid.GeShu*3*ENSEMBLE) ;  
	curandDestroyGenerator(gen) ;  
}
