#include "PropertyCheck.h"
#include "MacroDefinition.h"
#include "Structure.h"
#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include "Tools.h"
#include "cuda_runtime.h"

#ifdef CHECK_MIN_DIST

__global__ void Initialize_check_min_dist(Particle colloid)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < colloid.GeShu*ENSEMBLE)
	{
		colloid.device_min_dist_check[tid] = 999.0f;
		colloid.device_min_dist_check_flag[tid] = 0;
	}
}

float Check_min_dist(Particle& colloid)
{
	cudaMemcpy( colloid.host_min_dist_check, colloid.device_min_dist_check, colloid.GeShu * ENSEMBLE * sizeof(float), cudaMemcpyDeviceToHost) ;

	Initialize_check_min_dist <<<Calc_blocks(colloid.GeShu)*ENSEMBLE, threadsPerBlock>>> (colloid);

	/*for (int i=0;i<colloid.GeShu*ENSEMBLE;i++)
	{
		if (colloid.host_min_dist_check[i] == 0)
		{
			printf("wrong is %d\n", i);
		}
	}*/
	thrust::sort (colloid.host_min_dist_check, colloid.host_min_dist_check+colloid.GeShu*ENSEMBLE);
	return colloid.host_min_dist_check[0];
}

#endif


void Check_divergency(Particle& colloid)
{
	//cudaDeviceSynchronize();
	cudaMemcpy(colloid.host_div, colloid.divergency, sizeof(int), cudaMemcpyDeviceToHost);

	for (int i=0;i<ENSEMBLE;i++)
	{
		if (colloid.host_div[i] != -1)
		{
			printf("the particle %d in ensemble %d divergency\n", colloid.divergency[i], i);
			exit(0);
		}
	}
}

