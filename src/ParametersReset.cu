#include "Structure.h"
#include "ParametersReset.h"
#include <math.h>

void Parameters_Reset(Particle *colloid, Cell *force_cell, Cell *density_cell, IP *interact_para, SP *system_para)
{
	/************************************************************************/
	/* the relevant parameters for IP                                       */
	/************************************************************************/
	interact_para->epsilon        = 1.0f;
	interact_para->sigma          = 1.0f;
	interact_para->r_cut          = 1.1225f;
	interact_para->r_offset       = 0.0f;
	
	interact_para->Harmonic_cut   = 1.5f;
	interact_para->Harmonic_offset= 1.5f;
	interact_para->Harmonic_K     = 500.0f;

	interact_para->Harmonic_cut2 = 3.0f;
	interact_para->Harmonic_offset2 = 3.0f;

	interact_para->Harmonic_K2 = 220.0f;
	colloid->Activity = 40.0f;

	/************************************************************************/
	/* the relevant parameters for Cell                                     */
	/************************************************************************/
	force_cell->box_l[0]          = 100.0f;
	force_cell->box_l[1]          = 100.0f;
	force_cell->cellsPerDim[0]    = int(force_cell->box_l[0] / interact_para->Harmonic_cut2);
	force_cell->cellsPerDim[1]    = int(force_cell->box_l[1] / interact_para->Harmonic_cut2);
	force_cell->cells             = force_cell->cellsPerDim[0] * force_cell->cellsPerDim[1];

	density_cell->box_l[0] = 100.0f;
	density_cell->box_l[1] = 100.0f;
	density_cell->cellsPerDim[0] = int(density_cell->box_l[0] / 5.0f);
	density_cell->cellsPerDim[1] = int(density_cell->box_l[1] / 5.0f);
	density_cell->cells = density_cell->cellsPerDim[0] * density_cell->cellsPerDim[1];

	density_cell->From_min        = 0.0f;
	density_cell->To_max          = 3.0f;
	density_cell->Bins            = 75;
	/************************************************************************/
	/* the relevant parameters for SP                                       */
	/************************************************************************/
	system_para->temperature      = 0.1f;
	system_para->gamma            = 1.0f;
	system_para->timeStep         = 1.0e-5;
	system_para->n_cycle          = 100;
	system_para->n_per_cycle      = 100;
	system_para->n_cycle_warm     = 100000;
	system_para->n_per_cycle_warm = 10;

	system_para->n_cycle_density_sampling = 1;
	/************************************************************************/
	/* the relevant parameters for Particle                                 */
	/************************************************************************/
	colloid->if_input_file        = 0;
}

