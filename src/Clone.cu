#include "Clone.h"

Clone::Clone()
{
	amount_number = 10;
	clone_num = nullptr; //clone number of every gene
	alpha = 20.0; //bias
	iter_id = 0; //...
	iter_number = 5000;
	numClonesPredicted = 0;
	pre_predict = 0;
#ifdef PIC
	pic.initial(4096);
#endif // PIC
}

Clone::~Clone()
{
}

void Clone::Memory_allocation()
{
	clone_num = new int[amount_number];
	memset(clone_num, 0, sizeof(int) * amount_number);
	father_id = new int[amount_number];
	memset(father_id, -1, sizeof(int) * amount_number);
	EPR = new float[amount_number];
	memset(EPR, 0, sizeof(float) * amount_number);
	ratio = new float[3 * amount_number];
	memset(ratio, 0, sizeof(float) * amount_number * 3);
}

void Clone::calc_n()
{
	long double s_a = 0.0;
	numClonesPredicted = 0;
	if (iter_id > 2) {
		for (int i = 0; i < amount_number; i++) {
			s_a += expl(alpha * EPR[i]);
		}
	}
	else {
		for (int i = 0; i < amount_number; i++) {
			clone_num[i] = 1;
		}
		return;
	}

	for (int i = 0; i < amount_number; i++) {
		clone_num[i] = int(expl(alpha * EPR[i]) / s_a * amount_number + 0.5);
		numClonesPredicted += clone_num[i];
	}
	pre_predict = numClonesPredicted;
	adjust_population();
	
}

void Clone::adjust_population()
{
	while (numClonesPredicted != amount_number) {
		if (numClonesPredicted > amount_number) {
			int id = int(Uniform() * amount_number);
			while (clone_num[id] == 0) {
				id = int(Uniform() * amount_number);
			}
			clone_num[id] --;
			numClonesPredicted--;
		}
		else if (numClonesPredicted < amount_number) {
			int id = int(Uniform() * amount_number);
			clone_num[id] ++;
			numClonesPredicted++;
		}
	}
}

void Clone::update_fatherid()
{
	int left_id = 0;

	for (int i = 0; i < amount_number; i++) {
		
		if (clone_num[i] > 0) {
			for (int j = left_id; j < left_id + clone_num[i]; j++) {
				father_id[j] = i;
			}
			left_id += clone_num[i];
		}
	}
}

void Clone::outputfile()
{
	FILE* fpp;
	static int cnt = 0;
	char name[100] = "";
	sprintf(name, "Gen/clone_info/info_clone%d.txt", cnt);
	fpp = fopen(name, "w");
	for (int i = 0; i < amount_number; i++) {
		fprintf(fpp, "%f %d %d %d %0.2f %0.2f %0.2f\n", EPR[i], clone_num[i], father_id[i], pre_predict, ratio[3*i], ratio[3*i+1], ratio[3*i+2]);
	}
	cnt++;
	fclose(fpp);
}

void Clone::run(int clone_id)
{
	Particle colloid;
	Particle colloid_1;
	Cell force_cell;
	Cell force_cell_1;

	Cell density_cell;
	IP interact_para;
	SP system_para;

	//the Global parameters
	colloid.Initialize_particle_parameters();
	force_cell.Initializ_cell_parameters();
	density_cell.Initializ_cell_parameters();
	interact_para.Initialize_interaction_parameters();
	system_para.Initialize_system_parameters();

	//if any change of the global parameters needed;
	Parameters_Reset(&colloid, &force_cell, &density_cell, &interact_para, &system_para);

	//Input info
	colloid.GeShu = 4096;

	//Allocation of memory for variety uses;
	colloid.MemoryAllocation();
	force_cell.MemoryAllocation();
	density_cell.MemoryAllocation();
	interact_para.memory_allocation();
	system_para.memory_allocation();

	if (iter_id == 0) {
		colloid.if_input_file = 0;
		system_para.n_cycle = 300;
	}
	else {
		colloid.if_input_file = 1;
		system_para.n_cycle = 500;
	}

	/************************************************************************/
	/* the program's main-body;                                             */
	/************************************************************************/
	/*
	colloid.Activity = 40.0f;
	interact_para.Harmonic_K2 = 300.0f;
	*/

	char flag[100] = "";
	//sprintf(flag, "%s/A_%2.1f_Ks_%2.1f_iterID_%d_", out_dir, colloid.Activity, interact_para.Harmonic_K2, iter_id);

	colloid.device_time_seed = rand();
	int warm_flag;
	do {
		colloid.host_time_seed = rand();

		sprintf(flag, "%d", father_id[clone_id]);
		Initialize_system(colloid, force_cell, interact_para, system_para, flag);

		if (colloid.if_input_file == 0) {
			warm_flag = Warm_System(colloid, force_cell, interact_para, system_para);
			//warm_flag = 1;
		}
		else {
			warm_flag = 1;
		}

	} while (!warm_flag);

	colloid_1 = colloid;
	force_cell_1 = force_cell;
	colloid_1.Toself();
	force_cell_1.Toself();

	sprintf(flag, "%d", clone_id);
	Execute_System(colloid, force_cell, colloid_1, force_cell_1, density_cell, interact_para, system_para, flag);
#ifdef PIC
	if (clone_id == 0) {
		p = colloid;
		//pic.draw(p);
	}
#endif // PIC

	colloid.DataTransmit_DeviceToHost();
	colloid.OutputFile3(flag);

	EPR[clone_id] = colloid.calc_sumepr(system_para.n_cycle * system_para.n_per_cycle);
	colloid.calc_ratio();
	ratio[3 * clone_id] = colloid.ratio[0];
	ratio[3 * clone_id + 1] = colloid.ratio[1];
	ratio[3 * clone_id + 2] = colloid.ratio[2];
	

	sprintf(flag, "Gen/Output/A_%2.1f_Ks_%2.1f_clone_%d", colloid.Activity, interact_para.Harmonic_K2, clone_id);
	//colloid.OutputFile(flag);
	colloid.OutputFileLammpsTraj(flag, iter_id, 0, 100, 0, 100, 0, 0);
	

	//printf("parameter group %d is done, current activity is %f\n", i, colloid.Activity);
	/************************************************************************/
	/*                                                                      */
	/************************************************************************/
	//free the memory both device and host;
	interact_para.memory_destroy();
	system_para.memory_destroy();
	Finish_system(colloid, force_cell, density_cell);
	colloid_1.MemoryDestroyforcopy();
	force_cell_1.MemoryDestroyforcopy();
}


void Clone::draw()
{
#ifdef PIC
	pic.draw(p);
#endif // PIC
}



