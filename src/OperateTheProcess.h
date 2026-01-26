#ifndef  OPERATE_PROCESS
#define  OPERATE_PROCESS

#include "Structure.h"

void Initialize_system(Particle& colloid, Cell& force_cell, IP& interact_para, SP& system_para, char *flag);

void Finish_system(Particle& colloid, Cell& force_cell, Cell& density_cell);

void Execute_System(Particle& colloid, Cell& force_cell, Particle& colloid_1, Cell& force_cell_1, Cell& density_cell, IP& interact_para, SP& system_para, char* flag);

int Warm_System(Particle& colloid, Cell& force_cell, IP& interact_para, SP& system_para);

#endif
