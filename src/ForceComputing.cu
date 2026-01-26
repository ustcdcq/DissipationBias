#include "ForceComputing.h"
#include "MacroDefinition.h"
#include "Structure.h"
#include "cuda_runtime.h"
#include <math.h>
#include "PropertyCheck.h"
#include "RandomNumber.h"
#include "Tools.h"

#ifdef LJ
__device__ void force_LJ(Particle& colloid,
						 IP& interact_para,
				         int tid,
						 int tid_compare,
						 float dist[2])
{
	float r2 = get_length_square(dist);
	float r6i;
	float ff;

	if (r2 < interact_para.r_cut * interact_para.r_cut)
	{
		r6i = 1.0f / (r2*r2*r2);
		ff  = 48.0f*interact_para.epsilon*r6i*(r6i-0.5f)/r2;
#pragma unroll
		for(int i=0; i<2; i++)
		{
			atomicAdd(&colloid.device_force[tid].f[i], ff * dist[i]);
		}
	}

}
#endif

#ifdef HARMONIC
__device__ void force_harmonic(Particle& colloid, 
	                           IP& interact_para,
							   int tid,
							   int tid_compare,
							   float dist[2])
{
	float r2 = get_length_square(dist);
	float r  = sqrtf(r2);
	float delta_r ;
	float ff;

#ifdef CHECK_MIN_DIST
	Compare_And_Exchange(colloid, tid, r);
#endif
	/*if (tid == 8181)
	{
		printf("tid_compare is %d and r is %f\n", tid_compare, r);
	}*/
	if (r2 < interact_para.Harmonic_cut* interact_para.Harmonic_cut)
	{
	    delta_r = r - interact_para.Harmonic_offset;
		ff  = -interact_para.Harmonic_K * delta_r;
#pragma unroll
		for(int i=0; i<2; i++)
		{
			atomicAdd(&colloid.device_force[tid].f[i], ff * dist[i]);
		}
	}
}

__device__ void force_harmonic2(Particle& colloid,
	IP& interact_para,
	int tid,
	int tid_compare,
	float dist[2])
{
	float r2 = get_length_square(dist);
	float r = sqrtf(r2);
	float delta_r;
	float ff;

#ifdef CHECK_MIN_DIST
	Compare_And_Exchange(colloid, tid, r);
#endif
	/*if (tid == 8181)
	{
		printf("tid_compare is %d and r is %f\n", tid_compare, r);
	}*/
	if (r2 < interact_para.Harmonic_cut2 * interact_para.Harmonic_cut2)
	{
		delta_r = r - interact_para.Harmonic_offset2;
		ff = -interact_para.Harmonic_K2 * delta_r;
#pragma unroll
		for (int i = 0; i < 2; i++)
		{
			atomicAdd(&colloid.device_force[tid].f[i], ff * dist[i]);
		}
	}
}
#endif

__global__ void compute_nonbond_force(Particle* colloid, Cell* force_cell, IP* interact_para, int initial_flag)
{
	int offset = threadIdx.x + blockIdx.x*blockDim.x;
    int tid  = int(offset/9);
	int ensemble_id = int(tid/colloid->GeShu);
	int id_in_27cells = offset%9;

	if ( tid < colloid->GeShu*ENSEMBLE)
	{
		int cell_x = int(id_in_27cells/3)-1;
		int cell_y = (id_in_27cells%3)-1;
		int cell_dist;
        int tid_compare;
		float dist[2];

		cell_x = colloid->device_force_cell_info[tid].cell_pos[0] + cell_x;
		cell_y = colloid->device_force_cell_info[tid].cell_pos[1] + cell_y;

		check_cell_pos_overflow(&cell_x, force_cell->cellsPerDim[0]);
		check_cell_pos_overflow(&cell_y, force_cell->cellsPerDim[1]);

		//find the entrance of the specific cell;
		cell_dist = get_total_index(cell_x, cell_y, force_cell->cellsPerDim[1]);
		cell_dist += force_cell->cells * ensemble_id;
		tid_compare = force_cell->cell_head[cell_dist];

		while(tid_compare != -1)
		{
			if (tid == tid_compare)
			{
				tid_compare = colloid->device_force_cell_node[tid_compare].next;
				continue;
			}else{
				get_reduced_distance(*colloid, *force_cell, tid, tid_compare, dist);

            	if (initial_flag == 1 )
            	{
					force_harmonic(*colloid, *interact_para, tid, tid_compare, dist);
				}else{
					
					
					force_LJ(*colloid, *interact_para, tid, tid_compare, dist);
					
					force_harmonic2(*colloid, *interact_para, tid, tid_compare, dist);
				
				}
				
				tid_compare = colloid->device_force_cell_node[tid_compare].next;
			}
		}

		colloid->device_conven_force[tid].f[0] = colloid->device_force[tid].f[0];
		colloid->device_conven_force[tid].f[1] = colloid->device_force[tid].f[1];
	}

}

__global__ void compute_active_force(Particle* colloid)
{
	int tid =  threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < colloid->GeShu*ENSEMBLE)
	{
		colloid->device_active_force[tid].f[0] += colloid->Activity * cosf(colloid->device_ori[tid].phi);
		colloid->device_active_force[tid].f[1] += colloid->Activity * sinf(colloid->device_ori[tid].phi);

		colloid->device_force[tid].f[0] += colloid->Activity * cosf(colloid->device_ori[tid].phi);
		colloid->device_force[tid].f[1] += colloid->Activity * sinf(colloid->device_ori[tid].phi);
	}
}

__global__ void compute_reset(Particle* colloid)
{
        int tid =  threadIdx.x + blockIdx.x*blockDim.x;

        if (tid < colloid->GeShu*ENSEMBLE)
        {
                colloid->device_pos[tid].p0[0] = 10.0f;
                colloid->device_pos[tid].p0[1] = 10.0f;
				colloid->device_pos[tid].num0[0] = 0;
				colloid->device_pos[tid].num0[1] = 0;
        }
}

__global__ void assign(Particle* colloid, Cell* force_cell)
{
    int tid =  threadIdx.x + blockIdx.x*blockDim.x;
	float qq1;
	float dist[2];

	if (tid < colloid->GeShu * ENSEMBLE)
	{
		for (int tid_compare = 0; tid_compare < colloid->GeShu * ENSEMBLE; tid_compare++)
		{
			if (tid != tid_compare)
			{
				get_reduced_distance(*colloid, *force_cell, tid, tid_compare, dist);
				qq1 = get_length(dist);
				if (qq1 < colloid->device_pos[tid].p0[0])
				{
					colloid->device_pos[tid].p0[1] = colloid->device_pos[tid].p0[0];
					colloid->device_pos[tid].num0[1] = colloid->device_pos[tid].num0[0];
					colloid->device_pos[tid].p0[0] = qq1;
					colloid->device_pos[tid].num0[0] = tid_compare;
				}
				else if (qq1 < colloid->device_pos[tid].p0[1])
				{
					colloid->device_pos[tid].p0[1] = qq1;
					colloid->device_pos[tid].num0[1] = tid_compare;
				}
			}

		}
	}
}

__global__ void assign2(Particle* colloid, Cell* force_cell)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	float qq1, qq2, qq3;
	float dist[2], dist2[2];

	if (tid < colloid->GeShu * ENSEMBLE)
	{
		get_reduced_distance(*colloid, *force_cell, colloid->device_pos[tid].num0[0], colloid->device_pos[tid].num0[1], dist);
		qq1 = get_length(dist);
		//get_angle(colloid->device_pos[tid].p0[0], colloid->device_pos[tid].p0[1], qq1, qq2);

		qq2 = (colloid->device_pos[tid].p0[0] * colloid->device_pos[tid].p0[0] + colloid->device_pos[tid].p0[1] * colloid->device_pos[tid].p0[1] - qq1 * qq1) / 2.0f/ colloid->device_pos[tid].p0[0]/ colloid->device_pos[tid].p0[1];
		//colloid->device_ori[tid].phi0 = qq2;
		colloid->device_pos[tid].num0[2] = 1;
		if (colloid->device_pos[tid].p0[1] <= 1.4f)
		{
			if (qq2 < -0.94)	colloid->device_pos[tid].num0[2] = 3;
			if ((qq2 > 0.342) && (qq2 < 0.6428))	colloid->device_pos[tid].num0[2] = 2;
		}
	}
}

__global__ void compute_stochastic_force(Particle* colloid, SP* system_para)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	float ext_ran = sqrtf(2.0*system_para->temperature * system_para->gamma / system_para->timeStep);

	if (tid < colloid->GeShu*ENSEMBLE)
	{
		colloid->device_force[tid].f[0] += ext_ran * curand_normal(colloid->device_seed+tid);
		colloid->device_force[tid].f[1] += ext_ran * curand_normal(colloid->device_seed+tid);
		//colloid.device_pos[tid].p0[0] = curand_normal(colloid.device_seed+tid);
        //colloid.device_pos[tid].p0[1] = curand_normal(colloid.device_seed+tid);
	}
}

void Computing_Forces(Particle& colloid, Cell& force_cell, IP& interact_para, SP& system_para,int initial_flag)       
{
	if (initial_flag <= 1)
	{
		cudaMemset(colloid.device_force, 0, colloid.GeShu * ENSEMBLE * sizeof(molF));
		cudaMemset(colloid.device_conven_force, 0, colloid.GeShu * ENSEMBLE * sizeof(molF));
		compute_nonbond_force << <Calc_blocks(colloid.GeShu) * 9 * ENSEMBLE, threadsPerBlock >> > (colloid.self, force_cell.self, interact_para.self, initial_flag);


		if (initial_flag == 0) 
		{
			cudaMemset(colloid.device_active_force, 0, colloid.GeShu * ENSEMBLE * sizeof(molF));
			compute_active_force << <Calc_blocks(colloid.GeShu * ENSEMBLE), threadsPerBlock >> > (colloid.self);
			compute_stochastic_force << <Calc_blocks(colloid.GeShu) * ENSEMBLE, threadsPerBlock >> > (colloid.self, system_para.self);
		}
	}
	else {
		compute_reset << <Calc_blocks(colloid.GeShu * ENSEMBLE), threadsPerBlock >> > (colloid.self);
		assign << <Calc_blocks(colloid.GeShu * ENSEMBLE), threadsPerBlock >> > (colloid.self, force_cell.self);
		assign2 << <Calc_blocks(colloid.GeShu * ENSEMBLE), threadsPerBlock >> > (colloid.self, force_cell.self);
	}
}

