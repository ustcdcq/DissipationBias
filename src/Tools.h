#ifndef TOOLS_H
#define TOOLS_H

#include "cuda_runtime.h"
#include "MacroDefinition.h"
#include <math.h>

__device__ inline float get_length(float dist[2])
{
	return sqrtf(dist[0]*dist[0]+dist[1]*dist[1]);
}

__device__ inline float get_length_square(float dist[2])
{
	return (dist[0]*dist[0]+dist[1]*dist[1]);
}

__device__ inline int get_total_index(int x, int y, int dimension)
{
	return  (x*dimension + y);
}
__device__ inline float my_fmodf(float data_input, float data_reduction)
{
	int mid1 = data_input/data_reduction;
	return (data_input - mid1 * data_reduction);
}

__device__ inline int get_grid_pos(float position, float box_l, float grid_number)
{
	float locateInBox;
	int grid_position;
	float grid_length;

	grid_length = box_l / grid_number;
	locateInBox = my_fmodf(position, box_l);
	if (locateInBox < 0) locateInBox += box_l;
	grid_position = int(locateInBox/grid_length);
	//avoid the "float point error";
	if (grid_position == grid_number) grid_position--;

	return grid_position;
}

__device__ inline void check_cell_pos_overflow(int *cell_pos, int cellsPerDim)
{
	if (*cell_pos<0)
	{
		*cell_pos+=cellsPerDim;
	} 
	else if(*cell_pos>=cellsPerDim)
	{
		*cell_pos-=cellsPerDim;
	}
}

__device__ inline void get_reduced_distance(Particle& colloid, Cell& cell_system, int& p1, int& p2, float dist[2])
{
#pragma  unroll
	for (int i=0;i<2;i++) 
	{
		dist[i] = colloid.device_pos[p1].p[i] - colloid.device_pos[p2].p[i];
		dist[i] = my_fmodf(dist[i],cell_system.box_l[i]);
		if (dist[i] >= cell_system.box_l[i]/2.0f) dist[i] -= cell_system.box_l[i];
		else if(dist[i]< - cell_system.box_l[i]/2.0f) dist[i] += cell_system.box_l[i];
	}
}

__device__ inline void get_angle(float& q1, float& q2, float& q3, float& q4)
{
	q4 = (q1 * q1 + q2 * q2 - q3 * q3) / 2.0 / q1 / q2;
}


inline int Calc_blocks(int input)
{
	int output;

	if (input%threadsPerBlock == 0)
	{
		output = int(input/threadsPerBlock);
	}else{
		output = int(input/threadsPerBlock)+1;
	}

	return output;
}



template<typename T>
void swapObjectValue(T &a,T &b)
{
	T temp=a;
	a=b;
	b=temp;
}

#endif