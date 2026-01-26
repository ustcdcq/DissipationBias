#ifndef CELL_LIST
#define CELL_LIST

#include "cuda_runtime.h"
#include "Structure.h"

__global__ void Get_Cell_Coordination(molP *device_pos, Information *device_info, Cell cell_system, int GeShu);

__global__ void Get_Cell_Coordination2(molP *device_pos, Information *device_info, Cell cell_system, int GeShu);

__global__ void Build_Cell_List (Information *device_info, Node *device_node, Cell cell_system, int GeShu);


#endif
