#include <stdio.h>
#include <stdlib.h>
#include "Structure.h"
#include "MacroDefinition.h"
#include "cuda_runtime.h"
#include "RandomNumber.h"

#include "Tools.h"
#include <curand_kernel.h>
#include <math.h>

/************************************************************************/
/*The realization of Particle                                           */
/************************************************************************/
void Particle::Initialize_particle_parameters()
{
	GeShu = 4096;
	Activity = 0.0f;
	device_time_seed = 231;
	host_time_seed = 321;

	if_input_file = 0;
}

void Particle::MemoryAllocation()
{
	cudaMalloc((void**)&device_pos, GeShu * ENSEMBLE * sizeof(molP));
	cudaMalloc((void**)&device_force, GeShu * ENSEMBLE * sizeof(molF));
	cudaMalloc((void**)&device_force_cell_info, GeShu * ENSEMBLE * sizeof(Information));
	cudaMalloc((void**)&device_density_cell_info, GeShu * ENSEMBLE * sizeof(Information));

	cudaMalloc((void**)&device_ori, GeShu * ENSEMBLE * sizeof(molO));

	cudaMalloc((void**)&device_force_cell_node, GeShu * ENSEMBLE * sizeof(Node));
	cudaMalloc((void**)&device_density_cell_node, GeShu * ENSEMBLE * sizeof(Node));

	cudaMalloc((void**)&device_seed, GeShu * ENSEMBLE * sizeof(curandState));
	cudaMalloc((void**)&device_rand, GeShu * 3 * ENSEMBLE * sizeof(float));

	cudaMalloc((void**)&device_active_force, GeShu * ENSEMBLE * sizeof(molF));
	cudaMalloc((void**)&device_conven_force, GeShu * ENSEMBLE * sizeof(molF));

	cudaMalloc((void**)&S_gpu, GeShu * ENSEMBLE * sizeof(S));
	cudaMemset(S_gpu, 0, GeShu * ENSEMBLE * sizeof(S));

	cudaMalloc((void**)&dtheta, GeShu * ENSEMBLE * sizeof(float));

	S_cpu = (S*)malloc(GeShu * ENSEMBLE * sizeof(S));
	memset(S_cpu, 0, GeShu * ENSEMBLE * sizeof(S));

	host_pos = (molP*)malloc(GeShu * ENSEMBLE * sizeof(molP));
	host_force = (molF*)malloc(GeShu * ENSEMBLE * sizeof(molF));
	host_ori = (molO*)malloc(GeShu * ENSEMBLE * sizeof(molO));
	ratio = (float*)malloc(3 * sizeof(float));
	//Property check;
#ifdef CHECK_MIN_DIST
	cudaMalloc((void**)&device_min_dist_check, GeShu * ENSEMBLE * sizeof(int));
	cudaMalloc((void**)&device_min_dist_check_flag, GeShu * ENSEMBLE * sizeof(int));
	host_min_dist_check = (float*)malloc(GeShu * ENSEMBLE * sizeof(float));
#endif
	//cudaMallocManaged((void**)&divergency, ENSEMBLE * sizeof(int));
	cudaMalloc((int**)&divergency, ENSEMBLE * sizeof(int));
	cudaMemset(divergency, -1, ENSEMBLE*sizeof(int));

	host_div = (int*)malloc(ENSEMBLE * sizeof(int));
	memset(host_div, -1, ENSEMBLE * sizeof(int));
	//cudaMemset()

	cudaMalloc((void**)&self, sizeof(Particle));

	return;
}

void Particle::MemoryDestroy()
{
	cudaFree(device_pos);
	cudaFree(device_force);
	cudaFree(device_force_cell_info);
	cudaFree(device_density_cell_info);
	cudaFree(device_ori);
	cudaFree(device_force_cell_node);
	cudaFree(device_density_cell_node);
	cudaFree(device_seed);
	cudaFree(device_rand);

	cudaFree(device_active_force);
	cudaFree(device_conven_force);
	cudaFree(S_gpu);

	free(S_cpu);
	free(host_pos);
	free(host_force);
	free(host_ori);
	free(ratio);

	//Property check;
#ifdef CHECK_MIN_DIST
	cudaFree(device_min_dist_check);
	cudaFree(device_min_dist_check_flag);
	free(host_min_dist_check);
#endif

	cudaFree(divergency);
	cudaFree(self);

	return;
}

void Particle::DataTransmit_HostToDevice()
{
	cudaMemcpy(device_pos, host_pos, GeShu * ENSEMBLE * sizeof(molP), cudaMemcpyHostToDevice);
	cudaMemcpy(device_ori, host_ori, GeShu * ENSEMBLE * sizeof(molO), cudaMemcpyHostToDevice);
}

void Particle::DataTransmit_DeviceToHost()
{
	cudaMemcpy(host_pos, device_pos, GeShu * ENSEMBLE * sizeof(molP), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_force, device_force, GeShu * ENSEMBLE * sizeof(molF), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_ori, device_ori, GeShu * ENSEMBLE * sizeof(molO), cudaMemcpyDeviceToHost);
	cudaMemcpy(S_cpu, S_gpu, GeShu * ENSEMBLE * sizeof(S), cudaMemcpyDeviceToHost);
}

Particle::Particle()
{
}

Particle::~Particle()
{
}

float Particle::calc_sumepr(int n_step)
{
	float sum = 0.0;
	for (int i = 0; i < GeShu; i++) {
		sum += (S_cpu[i].s[1] + S_cpu[i].s[0]);
	}
	//return sum/n_step;
	return sum/GeShu;
}

void Particle::operator=(Particle& p)
{
	memcpy(this, &p, sizeof(Particle));

	cudaMalloc((void**)&device_force, GeShu * ENSEMBLE * sizeof(molF));
	cudaMemcpy(device_force, p.device_force, GeShu * ENSEMBLE * sizeof(molF), cudaMemcpyDeviceToDevice);
	cudaMalloc((void**)&device_pos, GeShu * ENSEMBLE * sizeof(molP));
	cudaMemcpy(device_pos, p.device_pos, GeShu * ENSEMBLE * sizeof(molP), cudaMemcpyDeviceToDevice);
	cudaMalloc((void**)&device_force_cell_info, GeShu * ENSEMBLE * sizeof(Information));
	cudaMemcpy(device_force_cell_info, p.device_force_cell_info, GeShu * ENSEMBLE * sizeof(Information), cudaMemcpyDeviceToDevice);
	cudaMalloc((void**)&device_force_cell_node, GeShu * ENSEMBLE * sizeof(Node));
	cudaMemcpy(device_force_cell_node, p.device_force_cell_node, GeShu * ENSEMBLE * sizeof(Node), cudaMemcpyDeviceToDevice);
	cudaMalloc((void**)&device_ori, GeShu * ENSEMBLE * sizeof(molO));
	cudaMemcpy(device_ori, p.device_ori, GeShu * ENSEMBLE * sizeof(molO), cudaMemcpyDeviceToDevice);
	cudaMalloc((void**)&S_gpu, GeShu * ENSEMBLE * sizeof(S));
	cudaMemset(S_gpu, 0, GeShu * ENSEMBLE * sizeof(S));

	cudaMalloc((void**)&device_conven_force, GeShu * ENSEMBLE * sizeof(molF));
	cudaMalloc((void**)&device_active_force, GeShu * ENSEMBLE * sizeof(molF));

	cudaMalloc((void**)&self, sizeof(Particle));
}

void Particle::Toself()
{
	cudaMemcpy(self, this, sizeof(Particle), cudaMemcpyHostToDevice);
}

void Particle::calc_ratio()
{
	int dis = 0;
	int stripe = 0;
	int trimer = 0;
	if (host_pos != nullptr) {
		for (int i = 0; i < GeShu; i++) {

			if (host_pos[i].num0[2] == 1)
			{
				dis++;
			}
			else if (host_pos[i].num0[2] == 2)
			{
				trimer++;
			}
			else {
				stripe++;
			}
		}
		ratio[0] = dis*1.0 / GeShu;
		ratio[1] = stripe*1.0 / GeShu;
		ratio[2] = trimer*1.0 / GeShu;
	}
}

void Particle::MemoryDestroyforcopy()
{
	cudaFree(device_force);
	cudaFree(device_pos);
	cudaFree(device_force_cell_info);
	cudaFree(device_force_cell_node);
	cudaFree(device_ori);
	cudaFree(S_gpu);
	cudaFree(device_conven_force);
	cudaFree(device_active_force);
	cudaFree(self);
}

void Particle::InputFile()
{
	FILE* fp_pos[ENSEMBLE];
	FILE* fp_ori[ENSEMBLE];
	int i, j;
	int mid1, mid2;
	char Pre_Name_Pos[100] = "Input/Position/Position";
	char FileName_Pos[100];
	char Pre_Name_Ori[100] = "Input/Orientation/Orientation";
	char FileName_Ori[100];
	char FileIndex[100];

	for (i = 0; i < ENSEMBLE; i++)
	{
		strcpy(FileName_Pos, Pre_Name_Pos);
		strcpy(FileName_Ori, Pre_Name_Ori);
		//sprintf(FileIndex,"%d",i);

		//strcat(FileName_Pos, FileIndex);
		strcat(FileName_Pos, ".txt");
		//strcat(FileName_Ori, FileIndex);
		strcat(FileName_Ori, ".txt");

		mid1 = i * GeShu;
		mid2 = mid1 + GeShu;
		fp_pos[i] = fopen(FileName_Pos, "r");
		fp_ori[i] = fopen(FileName_Ori, "r");
		for (j = mid1; j < mid2; j++)
		{
			fscanf(fp_pos[i], "%f%f", &host_pos[j].p[0], &host_pos[j].p[1]);
			fscanf(fp_ori[i], "%f", &host_ori[j].phi);
			host_pos[j].p0[0] = host_pos[j].p[0];
			host_pos[j].p0[1] = host_pos[j].p[1];
		}
		fclose(fp_pos[i]);
		fclose(fp_ori[i]);
	}
}

void Particle::InputFile2(char* flag)
{
	FILE* fp_pos;
	FILE* fp_ori;

	char FileName_Pos[100] = "";
	sprintf(FileName_Pos, "Gen/Input/Position%s.txt", flag);
	char FileName_Ori[100] = "";
	sprintf(FileName_Ori, "Gen/Input/Orientation%s.txt", flag);

	//printf("%s\n", FileName_Pos);
	fp_pos = fopen(FileName_Pos, "r");
	fp_ori = fopen(FileName_Ori, "r");

	for (int j = 0; j < GeShu; j++)
	{
		fscanf(fp_pos, "%f %f", &host_pos[j].p[0], &host_pos[j].p[1]);
		fscanf(fp_ori, "%f", &host_ori[j].phi);
		host_pos[j].p0[0] = host_pos[j].p[0];
		host_pos[j].p0[1] = host_pos[j].p[1];
		host_ori[j].phi0 = host_ori[j].phi;
	}
	fclose(fp_pos);
	fclose(fp_ori);
}

void Particle::InputFile3(char* flag)
{
	FILE* fp_pos;
	FILE* fp_ori;

	char FileName_Pos[100] = "";
	sprintf(FileName_Pos, "Input/Position.txt");
	char FileName_Ori[100] = "";
	sprintf(FileName_Ori, "Input/Orientation.txt");

	//printf("%s\n", FileName_Pos);
	fp_pos = fopen(FileName_Pos, "r");
	fp_ori = fopen(FileName_Ori, "r");

	for (int j = 0; j < GeShu; j++)
	{
		fscanf(fp_pos, "%f %f", &host_pos[j].p[0], &host_pos[j].p[1]);
		fscanf(fp_ori, "%f", &host_ori[j].phi);
		host_pos[j].p0[0] = host_pos[j].p[0];
		host_pos[j].p0[1] = host_pos[j].p[1];
		host_ori[j].phi0 = host_ori[j].phi;
	}
	fclose(fp_pos);
	fclose(fp_ori);
}

void Particle::OutputFile(char* flag)
{
	FILE* fp_pos[ENSEMBLE];
	FILE* fp_ori[ENSEMBLE];
	int mid1, mid2;

	char FileName_Pos[100];
	sprintf(FileName_Pos, "%s_Pos", flag);
	char FileName_Ori[100];
	sprintf(FileName_Ori, "%s_Ori", flag);
	char FileIndex[100];

	for (int i = 0; i < ENSEMBLE; i++)
	{
		sprintf(FileIndex, "%d", i);
		strcat(FileName_Pos, "_ems_");
		strcat(FileName_Ori, "_ems_");
		strcat(FileName_Pos, FileIndex);
		strcat(FileName_Ori, FileIndex);
		strcat(FileName_Pos, ".txt");
		strcat(FileName_Ori, ".txt");
		mid1 = i * GeShu;
		mid2 = mid1 + GeShu;
		fp_pos[i] = fopen(FileName_Pos, "w");
		fp_ori[i] = fopen(FileName_Ori, "w");

		for (int j = mid1; j < mid2; j++)
		{
			fprintf(fp_pos[i], "%f %f %f %d %f %f\n", host_pos[j].p[0], host_pos[j].p[1], host_ori[j].phi, host_pos[j].num0[2], S_cpu[j].s[0], S_cpu[j].s[1]);
			fprintf(fp_ori[i], "%f\n", host_ori[j].phi);
		}
		fclose(fp_pos[i]);
		fclose(fp_ori[i]);
	}
}

void Particle::OutputFileLammpsTraj(const char* flag, long long timestep,
	double xlo, double xhi,
	double ylo, double yhi,
	double zlo, double zhi)
{
	// 单个轨迹文件：xxx.dump
	char fileName[256];
	std::snprintf(fileName, sizeof(fileName), "%s.dump", flag);

	// 轨迹文件：每次调用追加一帧
	FILE* fp = std::fopen(fileName, "a");
	if (!fp) return;

	const int N = GeShu;   // 原来每个 ensemble 的粒子数；现在就是总粒子数
	// 如果你的总粒子数不是 GeShu，而是别的（比如 N_tot），改这里即可

	// ---- LAMMPS dump header ----
	std::fprintf(fp, "ITEM: TIMESTEP\n%lld\n", timestep);
	std::fprintf(fp, "ITEM: NUMBER OF ATOMS\n%d\n", N);
	std::fprintf(fp, "ITEM: BOX BOUNDS pp pp pp\n");
	std::fprintf(fp, "%.16g %.16g\n", xlo, xhi);
	std::fprintf(fp, "%.16g %.16g\n", ylo, yhi);
	std::fprintf(fp, "%.16g %.16g\n", zlo, zhi);

	// ---- ATOMS columns ----
	// 你原来输出：x y phi num0[2] s0 s1
	// 这里我做成：id type x y z phi s0 s1
	std::fprintf(fp, "ITEM: ATOMS id type x y z phi s0 s1\n");

	// ---- atom lines ----
	for (int j = 0; j < N; j++)
	{
		int id = j + 1;
		int type = host_pos[j].num0[2];   // 你原来的 %d

		double x = host_pos[j].p[0];
		double y = host_pos[j].p[1];
		double z = 0.0;                  // 2D 体系就写 0；若有 p[2] 就替换

		double phi = host_ori[j].phi;
		double s0 = S_cpu[j].s[0];
		double s1 = S_cpu[j].s[1];

		std::fprintf(fp, "%d %d %.16g %.16g %.16g %.16g %.16g %.16g\n",
			id, type, x, y, z, phi, s0, s1);
	}

	std::fclose(fp);
}

void Particle::OutputFile2(char* flag)
{
	FILE* fpp;
	char FileName_total[100] = "";
	strcat(FileName_total, flag);
	strcat(FileName_total, ".txt");
	fpp = fopen(FileName_total, "a");
	int P[3];
	float P2[3];

	P[0] = 0;
	P[1] = 0;
	P[2] = 0;
	for (int i = 0; i < GeShu; i++)
	{
		P[host_pos[i].num0[2] - 1]++;
	}
	for (int j = 0; j < 3; j++)
	{
		P2[j] = 1.0 * P[j] / GeShu;
	}
	fprintf(fpp, "%f\t%f\t%f\n", P2[0], P2[1], P2[2]);
	fclose(fpp);
}

void Particle::OutputFile3(char* flag)
{
	FILE* fp_pos;
	FILE* fp_ori;

	char FileNamePos[100] = "";
	char FileNameOri[100] = "";
	sprintf(FileNamePos, "Gen/Input/Position%s.txt", flag);
	sprintf(FileNameOri, "Gen/Input/Orientation%s.txt", flag);

	fp_pos = fopen(FileNamePos, "w");
	fp_ori = fopen(FileNameOri, "w");

	for (int i = 0; i < this->GeShu; i++) {
		fprintf(fp_pos, "%f %f\n", host_pos[i].p[0], host_pos[i].p[1]);
		fprintf(fp_ori, "%f\n", host_ori[i].phi, host_ori[i].phi);
	}

	fclose(fp_pos);
	fclose(fp_ori);
}

void Particle::Initialize_node()
{
	cudaMemset(device_force_cell_node, -1, GeShu * ENSEMBLE * sizeof(Node));
	cudaMemset(device_density_cell_node, -1, GeShu * ENSEMBLE * sizeof(Node));
}

void Particle::Initialize_check()
{
	cudaMemset(divergency, -1, ENSEMBLE * sizeof(int));
	cudaDeviceSynchronize();
}

/************************************************************************/
/*The realization of Cell;                                              */
/************************************************************************/
void Cell::MemoryAllocation()
{
	cudaMalloc((void**)&cell_head, cells * ENSEMBLE * sizeof(int));
	cudaMalloc((void**)&lock_flag, cells * ENSEMBLE * sizeof(int));
	cudaMalloc((void**)&device_local_density, cells * ENSEMBLE * sizeof(float));
	cudaMalloc((void**)&device_local_vx, cells * ENSEMBLE * sizeof(float));
	cudaMalloc((void**)&device_local_vy, cells * ENSEMBLE * sizeof(float));
	host_frequency_count = (int*)malloc(Bins * sizeof(int));
	host_local_density = (float*)malloc(cells * ENSEMBLE * sizeof(float));
	host_local_vx = (float*)malloc(cells * ENSEMBLE * sizeof(float));
	host_local_vy = (float*)malloc(cells * ENSEMBLE * sizeof(float));
	cudaMalloc((void**)&self, sizeof(Cell));
}

void Cell::Initialize_simulation_parameters()
{
	cudaMemset(cell_head, -1, cells * ENSEMBLE * sizeof(int));
	cudaMemset(lock_flag, 0, cells * ENSEMBLE * sizeof(int));
	cudaMemset(device_local_density, 0, cells * ENSEMBLE * sizeof(float));
	cudaMemset(device_local_vx, 0, cells * ENSEMBLE * sizeof(float));
	cudaMemset(device_local_vy, 0, cells * ENSEMBLE * sizeof(float));
	memset(host_local_density, 0, cells * ENSEMBLE * sizeof(float));
	memset(host_local_vx, 0, cells * ENSEMBLE * sizeof(float));
	memset(host_local_vy, 0, cells * ENSEMBLE * sizeof(float));
	memset(host_frequency_count, 0, Bins * sizeof(int));
}

void Cell::MemoryDestroy()
{
	cudaFree(cell_head);
	cudaFree(lock_flag);
	cudaFree(device_local_density);
	cudaFree(device_local_vx);
	cudaFree(device_local_vy);
	cudaFree(self);
	free(host_local_density);
	free(host_local_vx);
	free(host_local_vy);
	free(host_frequency_count);
	return;
}



void Cell::Initializ_cell_parameters()
{
	cells = 6400;

	From_min = 0.0f;
	To_max = 1.2f;
	Bins = 24;
}

void Cell::DataTransmit_DeviceToHost()
{
	cudaMemcpy(host_local_density, device_local_density, cells * ENSEMBLE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_local_vx, device_local_vx, cells * ENSEMBLE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_local_vy, device_local_vy, cells * ENSEMBLE * sizeof(float), cudaMemcpyDeviceToHost);
}

void Cell::Output(char* flag)
{
	FILE* fpp;
	char Pre_Name_total[100] = "Output/Density/P_";
	char FileName_total[100];
	strcpy(FileName_total, Pre_Name_total);
	strcat(FileName_total, flag);
	strcat(FileName_total, ".txt");
	fpp = fopen(FileName_total, "a");
	int P1, P2, P3;
	for (int i = 0; i < Bins; i++)
	{
		fprintf(fpp, "%f\t", host_frequency_count[i] * 1.0f / (ENSEMBLE * cells));
	}
	fprintf(fpp, "\n");
	fclose(fpp);
}

void Cell::operator=(Cell& c)
{
	memcpy(this, &c, sizeof(Cell));
	cudaMalloc((void**)&cell_head, cells * ENSEMBLE * sizeof(int));
	cudaMemcpy(cell_head, c.cell_head, cells * ENSEMBLE * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMalloc((void**)&lock_flag, cells * ENSEMBLE * sizeof(int));
	cudaMemcpy(lock_flag, c.lock_flag, cells * ENSEMBLE * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMalloc((void**)&self, sizeof(Cell));
}

void Cell::MemoryDestroyforcopy()
{
	cudaFree(cell_head);
	cudaFree(lock_flag);
	cudaFree(self);
}

void Cell::Toself()
{
	cudaMemcpy(self, this, sizeof(Cell), cudaMemcpyHostToDevice);
}

void IP::Toself()
{
	cudaMemcpy(self, this, sizeof(IP), cudaMemcpyHostToDevice);
}

/************************************************************************/
/* The realization of interaction parameters                            */
/************************************************************************/
void IP::Initialize_interaction_parameters()
{
	//LJ interaction parameters between bead and bead;
#ifdef LJ
	sigma = 1.0f;
	epsilon = 1.0f;
	r_offset = 0.0f;
	r_cut = 1.1225f;
#endif

#ifdef HARMONIC
	Harmonic_K = 500.0f;
	Harmonic_offset = 1.2f;
	Harmonic_cut = 1.2f;
#endif

}

void IP::memory_destroy()
{
	cudaFree(self);
}

void IP::memory_allocation()
{
	cudaMalloc((void**)&self, sizeof(IP));
}


void SP::Toself()
{
	cudaMemcpy(self, this, sizeof(SP), cudaMemcpyHostToDevice);
}

void SP::memory_destroy()
{
	cudaFree(self);
}

/************************************************************************/
/* The realization of system parameters                                 */
/************************************************************************/
void SP::Initialize_system_parameters()
{
	temperature = 1.0;
	timeStep = 0.0001;
	gamma = 1.0f;

	n_cycle = 1;
	n_per_cycle = 1;
	n_cycle_warm = 1;
	n_per_cycle_warm = 1;

	n_cycle_density_sampling = 1;
}

void SP::memory_allocation()
{
	cudaMalloc((void**)&self, sizeof(SP));
}


