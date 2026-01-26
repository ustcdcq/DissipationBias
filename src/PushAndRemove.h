#ifndef PUSH_REMOVE
#define PUSH_REMOVE

#include "MacroDefinition.h"
#include "Structure.h"
#include "cuda_runtime.h"


__device__ inline int LOCK(Cell& cell_system, int cell_dist)
{
	return atomicExch(&cell_system.lock_flag[cell_dist],1);

}


__device__ inline void UNLOCK(Cell& cell_system, int cell_dist)
{
	atomicExch(&cell_system.lock_flag[cell_dist],0);
}


__device__ inline void Push_Into_Cell (Node *device_node, Cell& cell_system, int tid, int cell_dist)
{
	int head_in_cell = cell_system.cell_head[cell_dist];

	if (head_in_cell == -1)
	{
		cell_system.cell_head[cell_dist] = tid;
	}else{
		cell_system.cell_head[cell_dist] = tid;
		device_node[tid].next = head_in_cell;
		device_node[head_in_cell].front = tid;
	}
}


__device__ inline void Push_Into_Cell_Safe(Node *device_node, Cell& cell_system, int tid, int cell_dist)
{
	bool succeded=false;
	int lockState = LOCK(cell_system, cell_dist);
	while (succeded!=true)
	{
		if (lockState==0)
		{
			Push_Into_Cell(device_node, cell_system, tid, cell_dist);
			__threadfence();
			UNLOCK(cell_system, cell_dist);
			succeded = true;
		}
		else
		{
			lockState = LOCK(cell_system, cell_dist);
		}
	}
}

__device__ inline void Remove_From_Cell(Node *device_node, Cell& cell_system, int tid, int cell_dist)
{
	int front ;
	int next  ;

	if (device_node[tid].front == -1)
	{
		next = device_node[tid].next;
		if (next != -1)
		{
			device_node[next].front = -1;
		}
		
		cell_system.cell_head[cell_dist] = next;
	}else if (device_node[tid].next == -1)
	{
		front = device_node[tid].front;
	    device_node[front].next = -1;
	}else{
		front = device_node[tid].front;
		next  = device_node[tid].next;
		device_node[front].next = next;
		device_node[next].front = front;
	}

	//back the particle to a origin state;
	device_node[tid].front = -1;
    device_node[tid].next  = -1;
}


__device__ inline void Remove_From_Cell_Safe(Node *device_node, Cell& cell_system, int tid, int cell_dist)
{
	bool succeded = false;
	int lockState = LOCK(cell_system, cell_dist);

	while (succeded != true)
	{
		if (lockState == 0)
		{
			Remove_From_Cell(device_node, cell_system, tid, cell_dist);
			__threadfence();
			UNLOCK(cell_system, cell_dist);
			succeded = true;
		}else {
			lockState = LOCK(cell_system, cell_dist);
		}
	}
}

#endif
