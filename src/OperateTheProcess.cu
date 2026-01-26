#include "OperateTheProcess.h"
#include "MacroDefinition.h"
#include "Structure.h"
#include "ForceComputing.h"
#include "Integration.h"
#include "CellList.h"
#include "Tools.h"
#include "PropertyCheck.h"
#include "RandomNumber.h"
#include "LocalDensity.h"
#include <stdio.h>

__global__ void Random_Generated_System(Particle colloid, Cell force_cell)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	int index ;

	if (tid <colloid.GeShu*ENSEMBLE)
	{
		index = tid * 3;
		colloid.device_pos[tid].p[0] = force_cell.box_l[0] * colloid.device_rand[index];
		colloid.device_pos[tid].p[1] = force_cell.box_l[1] * colloid.device_rand[index+1];
		colloid.device_ori[tid].phi = 2.0f * PI * colloid.device_rand[index+2];
		colloid.device_ori[tid].phi0 = colloid.device_ori[tid].phi;

		//int qq1 = index / 64;
		//int qq2 = index - qq1 * 64;
		//colloid.device_pos[tid].p[0] = force_cell.box_l[0] * qq1 / 64.0;
		//colloid.device_pos[tid].p[1] = force_cell.box_l[1] * qq2 / 64.0;

		colloid.device_pos[tid].identity = tid;
	}
}

void Random_Generated_System_Host(Particle colloid)
{
	SetSeed(colloid.host_time_seed);

	for (int i=0;i<colloid.GeShu*ENSEMBLE;i++)
	{
		for (int j=0;j<2;j++)
		{
			colloid.host_pos[i].p[j] = Uniform() * BOX_L;
		}
	}

	for (int i=0;i<colloid.GeShu*ENSEMBLE;i++)
	{
		colloid.host_ori[i].phi = Uniform() * 2.0f * PI;
		colloid.host_pos[i].identity = i;
	}

}

void Initialize_system(Particle& colloid, Cell& force_cell, IP& interact_para, SP& system_para, char *flag)
{
	if (colloid.if_input_file == 1)
	{
		char FileName[100];
        strcpy(FileName, flag);
		colloid.InputFile2(FileName);
		colloid.DataTransmit_HostToDevice();
		Device_set_seed <<<Calc_blocks(colloid.GeShu)*ENSEMBLE, threadsPerBlock>>> (colloid);
	}
	else if (colloid.if_input_file == 2) {
		char FileName[100];
		strcpy(FileName, flag);
		colloid.InputFile3(FileName);
		colloid.DataTransmit_HostToDevice();
		Device_set_seed << <Calc_blocks(colloid.GeShu) * ENSEMBLE, threadsPerBlock >> > (colloid);
	}
	else {
		Device_set_seed << <Calc_blocks(colloid.GeShu) * ENSEMBLE, threadsPerBlock >> > (colloid);
		Generate_Random_Number(colloid);
		Random_Generated_System << <Calc_blocks(colloid.GeShu) * ENSEMBLE, threadsPerBlock >> > (colloid, force_cell);
		//Random_Generated_System_Host(colloid);
		//colloid.DataTransmit_HostToDevice();
	}

	colloid.Initialize_node();
	colloid.Initialize_check();

	force_cell.Initialize_simulation_parameters();
#ifdef CHECK_MIN_DIST
	Initialize_check_min_dist <<<Calc_blocks(colloid.GeShu)*ENSEMBLE, threadsPerBlock>>> (colloid);
#endif

	Get_Cell_Coordination <<<Calc_blocks(colloid.GeShu)*ENSEMBLE, threadsPerBlock>>> (colloid.device_pos, 
		                                                                              colloid.device_force_cell_info, 
																			          force_cell,
																			          colloid.GeShu);

	Build_Cell_List <<<Calc_blocks(colloid.GeShu)*ENSEMBLE, threadsPerBlock>>> (colloid.device_force_cell_info,
		                                                                        colloid.device_force_cell_node,
																	            force_cell,
																	            colloid.GeShu);
	colloid.Toself();
	force_cell.Toself();
	interact_para.Toself();
	system_para.Toself();
}

void Finish_system(Particle& colloid, Cell& force_cell, Cell& density_cell)
{
	colloid.MemoryDestroy();
	force_cell.MemoryDestroy();
	density_cell.MemoryDestroy();
}

/*
void Finish_system(Particle& colloid, Cell& force_cell, Cell& density_cell)
{
	colloid.MemoryDestroy();
	force_cell.MemoryDestroy();
	density_cell.MemoryDestroy();
}
*/

int Warm_System(Particle& colloid, Cell& force_cell, IP& interact_para, SP& system_para)
{
	int i,j;
	int initial_flag = 1;
	float min_dist = 0.0f;
	float safe_dist = 1.45f;
    int warm_flag = 1;

	for(i=0; i< system_para.n_cycle_warm; i++)
	{
		for (j=0; j<system_para.n_per_cycle_warm; j++)
		{
			Computing_Forces(colloid, force_cell, interact_para, system_para, initial_flag);
			Integration (colloid, force_cell, system_para);
			Check_divergency(colloid);
		}
		min_dist = Check_min_dist(colloid);

        if(min_dist == 0.0f)
        {
            warm_flag = 0;
            printf("At least two particle overlap\n");
            break;
        }
		if (min_dist < safe_dist)
		{
			initial_flag = 1;
			printf("cycle %d is initializing and min dist is %f\n",i, min_dist);
		}else{
			printf("\n");
			printf("the warm is over\n");
			break;
		}
	}
	if (i == system_para.n_cycle_warm)
	{
		printf("the warm is failed\n");
		exit(0);
	}
	
    return warm_flag;
}


void Execute_System(Particle& colloid, Cell& force_cell, Cell& density_cell, IP& interact_para, SP& system_para, char *flag)
{
	int i,j;
	int initial_flag = 0;
	int index_for_density_sampling ;
	char FileIndex[100];

	
	for(i=0; i< system_para.n_cycle; i++)
	{
		int i1;
		
		char FileName[100];
        strcpy(FileName, flag);
		
		if (i % 100 == 0)  printf("cycle %d the execution is in\n",i);
		
		if (i % system_para.n_cycle_density_sampling == 0)
		{
			Computing_Forces(colloid, force_cell, interact_para, system_para, 2);
			cudaMemcpy(colloid.host_pos, colloid.device_pos, colloid.GeShu * ENSEMBLE * sizeof(molP), cudaMemcpyDeviceToHost);
			colloid.OutputFile2(flag);
		}

		if (i%5000 == 0) {
			index_for_density_sampling = i / system_para.n_cycle_density_sampling;
			sprintf(FileIndex,"%d",index_for_density_sampling);
            strcat(FileName, "_T_");
            strcat(FileName, FileIndex);
			
			//Computing_Forces(colloid, force_cell, interact_para, system_para, 2);
			colloid.DataTransmit_DeviceToHost();
			colloid.OutputFile(FileName);
		}


		for (j = 0; j < system_para.n_per_cycle; j++)
		{
			Computing_Forces(colloid, force_cell, interact_para, system_para, initial_flag);
			Integration(colloid, force_cell, system_para);
			//Check_divergency(colloid);
		}
	}
}

__global__ void Calculate_Sproduction(Particle* colloid, Particle* colloid_1) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < colloid->GeShu) {
		//for (int i = 0; i < 2; i++) {
		//	colloid.S_gpu[tid].s[i] = 0;
		//}
		for (int i = 0; i < 2; i++) {
			//colloid.S_gpu[tid].s[0] = 0;
			colloid->S_gpu[tid].s[0] += colloid_1->device_conven_force[tid].f[i] * colloid->device_pos[tid].dx[i];
			//colloid.S_gpu[tid].s[1] = 0;
			colloid->S_gpu[tid].s[1] += colloid_1->device_active_force[tid].f[i] * colloid->device_pos[tid].dx[i];
		}
	}
}

void Execute_System(Particle& colloid, Cell& force_cell, Particle& colloid_1, Cell& force_cell_1, Cell& density_cell, IP& interact_para, SP& system_para, char* flag)
{
	int i, j;
	int initial_flag = 0;
	int index_for_density_sampling;
	char FileIndex[100];


	cudaMemset(colloid.S_gpu, 0, sizeof(S) * colloid.GeShu);
	for (i = 0; i < system_para.n_cycle; i++)
	{
		int i1;

		char FileName[100];
		strcpy(FileName, flag);

		//if (i % 100 == 0)  printf("cycle %d the execution is in\n", i);

		
		if (i % system_para.n_cycle_density_sampling == 0)
		{
			Computing_Forces(colloid, force_cell, interact_para, system_para, 2);
			//cudaMemcpy(colloid.host_pos, colloid.device_pos, colloid.GeShu * ENSEMBLE * sizeof(molP), cudaMemcpyDeviceToHost);
			//colloid.OutputFile2(flag);
		}
		
		/*
		if (i % 99 == 0) {
			index_for_density_sampling = i / system_para.n_cycle_density_sampling;
			sprintf(FileIndex, "%d", index_for_density_sampling);
			strcat(FileName, "_T_");
			strcat(FileName, FileIndex);

			//Computing_Forces(colloid, force_cell, interact_para, system_para, 2);
			//cudaMemcpy(colloid.S_cpu, colloid.S_gpu, colloid.GeShu * ENSEMBLE * sizeof(float), cudaMemcpyDeviceToHost);
			//std::cout << colloid.S_cpu[0] << std::endl;
			//colloid.DataTransmit_DeviceToHost();
			//colloid.OutputFile(FileName);
		}
		*/

		/*
		if (i % system_para.n_cycle == 0) {
			cudaMemset(colloid.S_gpu, 0, sizeof(S) * colloid.GeShu);
		}
		*/

		for (j = 0; j < system_para.n_per_cycle; j++)
		{
			Computing_Forces(colloid, force_cell, interact_para, system_para, initial_flag);
			Integration(colloid, force_cell, colloid_1, force_cell_1, system_para);
			Computing_Forces(colloid_1, force_cell_1, interact_para, system_para, initial_flag);
			//Check_divergency(colloid);
			Calculate_Sproduction << <Calc_blocks(colloid.GeShu) * ENSEMBLE, threadsPerBlock >> > (colloid.self, colloid_1.self);
		}
	}
}
