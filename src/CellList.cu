#include "CellList.h"
#include "Structure.h"
#include "MacroDefinition.h"
#include "cuda_runtime.h"
#include "PushAndRemove.h"
#include "Tools.h"


__global__ void Get_Cell_Coordination(molP *device_pos, Information *device_info, Cell cell_system, int GeShu)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid < GeShu * ENSEMBLE)
	{
		//get the coordinate in cell matrix of each particle
		for (int i=0;i<2;i++) 
		{
			device_info[tid].cell_pos[i] = get_grid_pos(device_pos[tid].p[i], cell_system.box_l[i], cell_system.cellsPerDim[i]);
		}
	}
}

__global__ void Get_Cell_Coordination2(molP *device_pos, Information *device_info, Cell cell_system, int GeShu)
{
        int tid = threadIdx.x + blockIdx.x*blockDim.x;

        if (tid < GeShu * ENSEMBLE)
        {
                //get the coordinate in cell matrix of each particle
                for (int i=0;i<2;i++)
                {
                        device_info[tid].cell_pos[i] = get_grid_pos(device_pos[tid].p0[i], cell_system.box_l[i], cell_system.cellsPerDim[i]);
                }
        }
}

__global__ void Build_Cell_List (Information *device_info, Node *device_node, Cell cell_system, int GeShu)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int cell_dist;
	int ensemble_id = int(tid/GeShu);
	
	if (tid < GeShu*ENSEMBLE)
	{
		cell_dist = get_total_index(device_info[tid].cell_pos[0],
									device_info[tid].cell_pos[1],
									cell_system.cellsPerDim[1]);
		cell_dist = cell_dist + cell_system.cells * ensemble_id;
		Push_Into_Cell_Safe(device_node, cell_system, tid, cell_dist);
	}
}
