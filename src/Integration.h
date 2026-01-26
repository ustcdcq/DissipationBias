#ifndef INTEGRATION
#define INTEGRATION

#include "Structure.h"
#include <curand_kernel.h>

void Integration(Particle& colloid, Cell& force_cell_1, SP& system_para);

void Integration(Particle& colloid, Cell& force_cell, Particle& colloid_1, Cell& force_cell_1, SP& system_para);

__global__ void Position_reduction(Particle* colloid, Cell* cell_system);

#endif