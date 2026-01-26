#ifndef PROPERTY_CHECK
#define PROPERTY_CHECK

#include "Structure.h"
#include <stdio.h>


#ifdef CHECK_MIN_DIST

float Check_min_dist(Particle& colloid);

__global__ void Initialize_check_min_dist(Particle colloid);


//device function;
__device__ inline int LOCK_CHECK(Particle colloid, int tid)
{
	return atomicExch(&colloid.device_min_dist_check_flag[tid],1);
}

__device__ inline int UNLOCK_CHECK(Particle colloid, int tid)
{
	return atomicExch(&colloid.device_min_dist_check_flag[tid],0);

}

__device__ inline void Compare_And_Exchange(Particle& colloid, int tid, float r)
{

	int cnt = 0;
	bool succeed = false;

	int locked = atomicExch(colloid.device_min_dist_check_flag + tid, 1);
	while (!succeed) {
		if (!locked) {
			if (r < colloid.device_min_dist_check[tid])
			{
				colloid.device_min_dist_check[tid] = r;
			}
			__threadfence();
			colloid.device_min_dist_check_flag[tid] = 0;
			succeed = true;
		}
		else {
			locked = atomicExch(colloid.device_min_dist_check_flag + tid, 1);
		}
	}
}
#endif

void Check_divergency(Particle& colloid);

#endif