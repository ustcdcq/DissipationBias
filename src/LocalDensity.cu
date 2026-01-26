#include "LocalDensity.h"
#include "MacroDefinition.h"
#include "Structure.h"
#include "CellList.h"
#include "Tools.h"
#include <thrust/sort.h>

__global__ void Particle_Count_In_Cell(Particle colloid, Cell density_cell)
{
	int tid = threadIdx.x+blockIdx.x*blockDim.x;
	int count = 0;
	float q1, q2;
	float cell_l[2];
	cell_l[0] = density_cell.box_l[0] / density_cell.cellsPerDim[0];
	cell_l[1] = density_cell.box_l[1] / density_cell.cellsPerDim[1];
	float cell_area = cell_l[0] * cell_l[1];
	int flag ;
	if (tid < density_cell.cells*ENSEMBLE)
	{
		flag = density_cell.cell_head[tid];
		while (flag != -1)
		{
			flag = colloid.device_density_cell_node[flag].next;
			count++;
		}
		density_cell.device_local_density[tid] = count/cell_area;
	}
}


void Calc_frequency_profile(Cell density_cell)
{
	int index;
	float bin = (density_cell.To_max - density_cell.From_min)/density_cell.Bins;

	for (int i=0;i<density_cell.cells*ENSEMBLE;i++)
	{
		index = int(density_cell.host_local_density[i]/bin);
		if (index >= density_cell.Bins)
		{
			printf("density overflow\n");
			continue;
		}
		density_cell.host_frequency_count[index]++;
	}
}

void Calc_density_profile(Particle colloid, Cell density_cell, char *flag, int index)
{
	density_cell.Initialize_simulation_parameters();
	cudaMemset(colloid.device_density_cell_node, -1, colloid.GeShu * ENSEMBLE * sizeof(Node));
	Get_Cell_Coordination <<<Calc_blocks(colloid.GeShu*ENSEMBLE), threadsPerBlock>>> (colloid.device_pos, 
		                                                                              colloid.device_density_cell_info, 
		                                                                              density_cell,
		                                                                              colloid.GeShu);

	Build_Cell_List <<<Calc_blocks(colloid.GeShu*ENSEMBLE), threadsPerBlock>>> (colloid.device_density_cell_info,
		                                                                        colloid.device_density_cell_node,
		                                                                        density_cell,
		                                                                        colloid.GeShu);

	Particle_Count_In_Cell <<<Calc_blocks(density_cell.cells*ENSEMBLE), threadsPerBlock>>> (colloid, density_cell);

	density_cell.DataTransmit_DeviceToHost();
	Calc_frequency_profile(density_cell);	        
	density_cell.Output(flag);
	
}
