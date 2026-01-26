#include "Integration.h"
#include "cuda_runtime.h"
#include "Structure.h"
#include "Tools.h"
#include "PushAndRemove.h"
#include "RandomNumber.h"
#include <curand_kernel.h>
#include <stdio.h>

__device__ int Divergency_Judge(float tested_value)
{
	return (abs(tested_value) > 10E3);
}

__global__ void Integration_Orientation(Particle* colloid, SP* system_para)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	float ex_sto_rot   = sqrtf(6.0f * system_para->temperature * system_para->timeStep/GAMMA);
	int divergency_flag = 0;

	if (tid < colloid->GeShu*ENSEMBLE)
	{
		colloid->dtheta[tid] = ex_sto_rot * curand_normal(colloid->device_seed + tid);
		colloid->device_ori[tid].phi += colloid->dtheta[tid];
	}
}

__global__ void Position_reduction(Particle* colloid, Cell* cell_system)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < colloid->GeShu*ENSEMBLE)
	{
		for (int j=0;j<2;j++)
		{
			colloid->device_pos[tid].p[j] = my_fmodf(colloid->device_pos[tid].p[j], cell_system->box_l[j]);
			if (colloid->device_pos[tid].p[j] < 0)
			{
				colloid->device_pos[tid].p[j] += cell_system->box_l[j];
			}
		}
	}
}

__global__ void Get_displacement(Particle* colloid, Cell* force_cell, SP* system_para)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int ensemble_id = int(tid/colloid->GeShu);
	int i;
	int cell_pos[2];
	bool repair_cell_list = false;
	int cell_dist_old;
	int cell_dist_new;
	int divergency_flag = 0;
	float delta_r;

	if (tid < colloid->GeShu*ENSEMBLE)
	{
		for (i=0;i<2;i++)
		{
			delta_r =  colloid->device_force[tid].f[i] * system_para->timeStep /GAMMA;

			colloid->device_pos[tid].dx[i] = delta_r;

			//printf("%f\n", colloid->device_force[tid].dx[i]);

			divergency_flag = Divergency_Judge(delta_r);
			if (divergency_flag == 1) colloid->divergency[0] = tid;

			colloid->device_pos[tid].p[i] += delta_r;

			cell_pos[i] = get_grid_pos(colloid->device_pos[tid].p[i], force_cell->box_l[i], force_cell->cellsPerDim[i]);
			if (cell_pos[i] != colloid->device_force_cell_info[tid].cell_pos[i]) repair_cell_list = true;
		}


		if (repair_cell_list == true)
		{
			cell_dist_old = get_total_index(colloid->device_force_cell_info[tid].cell_pos[0],
				                            colloid->device_force_cell_info[tid].cell_pos[1],
				                            force_cell->cellsPerDim[1]);
			cell_dist_old += force_cell->cells * ensemble_id;

			cell_dist_new =  get_total_index(cell_pos[0], cell_pos[1], force_cell->cellsPerDim[1]);
			cell_dist_new += force_cell->cells * ensemble_id;

			Remove_From_Cell_Safe(colloid->device_force_cell_node, *force_cell, tid, cell_dist_old);
			Push_Into_Cell_Safe(colloid->device_force_cell_node, *force_cell, tid, cell_dist_new);
			for (i=0;i<2;i++) colloid->device_force_cell_info[tid].cell_pos[i] = cell_pos[i];
		}

	}
}

__global__ void update_pos(Particle* colloid, Particle* colloid_1, Cell* force_cell) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int ensemble_id = int(tid / colloid->GeShu);
	int i;
	int cell_pos[2];
	bool repair_cell_list = false;
	int cell_dist_old;
	int cell_dist_new;
	int divergency_flag = 0;

	if (tid < colloid->GeShu) {
		for (int i = 0; i < 2; i++) {
			colloid_1->device_pos[tid].p[i] = colloid->device_pos[tid].p[i] - 0.5*colloid->device_pos[tid].dx[i];
			//printf("%f\n", colloid->device_force[tid].dx[i]);
			colloid_1->device_pos[tid].dx[i] = colloid->device_pos[tid].dx[i];
			//printf("%f\n", colloid_1->device_force[tid].dx[i]);

			cell_pos[i] = get_grid_pos(colloid_1->device_pos[tid].p[i], force_cell->box_l[i], force_cell->cellsPerDim[i]);
			if (cell_pos[i] != colloid_1->device_force_cell_info[tid].cell_pos[i]) repair_cell_list = true;
		}
		if (repair_cell_list == true)
		{
			cell_dist_old = get_total_index(colloid_1->device_force_cell_info[tid].cell_pos[0],
				colloid_1->device_force_cell_info[tid].cell_pos[1],
				force_cell->cellsPerDim[1]);
			cell_dist_old += force_cell->cells * ensemble_id;

			cell_dist_new = get_total_index(cell_pos[0], cell_pos[1], force_cell->cellsPerDim[1]);
			cell_dist_new += force_cell->cells * ensemble_id;

			Remove_From_Cell_Safe(colloid_1->device_force_cell_node, *force_cell, tid, cell_dist_old);
			Push_Into_Cell_Safe(colloid_1->device_force_cell_node, *force_cell, tid, cell_dist_new);
			for (i = 0; i < 2; i++) colloid_1->device_force_cell_info[tid].cell_pos[i] = cell_pos[i];
		}
	}
}

__global__ void update_ori(Particle* colloid, Particle* colloid_1)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int divergency_flag = 0;

	if (tid < colloid->GeShu * ENSEMBLE)
	{
		colloid_1->device_ori[tid].phi = colloid->device_ori[tid].phi - 0.5 * colloid->dtheta[tid];
	}
}

void Integration(Particle& colloid, Cell& force_cell, Particle& colloid_1, Cell& force_cell_1, SP& system_para)
{
	Get_displacement <<<Calc_blocks(colloid.GeShu)*ENSEMBLE, threadsPerBlock>>> (colloid.self, force_cell.self, system_para.self);
	update_pos << <Calc_blocks(colloid.GeShu) * ENSEMBLE, threadsPerBlock >> > (colloid.self, colloid_1.self, force_cell_1.self);
	Integration_Orientation <<<Calc_blocks(colloid.GeShu)*ENSEMBLE, threadsPerBlock>>> (colloid.self, system_para.self);
	update_ori << <Calc_blocks(colloid.GeShu) * ENSEMBLE, threadsPerBlock >> > (colloid.self, colloid_1.self);
	Position_reduction <<<Calc_blocks(colloid.GeShu)*ENSEMBLE, threadsPerBlock>>> (colloid.self, force_cell.self);
}

void Integration(Particle& colloid, Cell& force_cell, SP& system_para)
{
	Get_displacement << <Calc_blocks(colloid.GeShu) * ENSEMBLE, threadsPerBlock >> > (colloid.self, force_cell.self, system_para.self);
	Integration_Orientation << <Calc_blocks(colloid.GeShu) * ENSEMBLE, threadsPerBlock >> > (colloid.self, system_para.self);
	Position_reduction << <Calc_blocks(colloid.GeShu) * ENSEMBLE, threadsPerBlock >> > (colloid.self, force_cell.self);
}
