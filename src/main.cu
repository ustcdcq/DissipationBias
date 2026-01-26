#include <stdio.h>
#include <string>
#include "Clone.h"
#include <thread>

//#define WEB
/*
#ifndef WEB
#include <direct.h>
#endif // DEBUG
*/
#ifdef PIC
#include "Picture.h"
#endif // 



void func(Clone& clone, int j, int cuda_id);
//void func(Clone& clone, Assemble& ass, int j);

int main(int argc, char* argv[])
{
	int N_GPU;
	cudaGetDeviceCount(&N_GPU);

	Clone clone;
	clone.alpha = 0.0;
	clone.iter_number = 5000;
	const int amount = 1000;

	clone.amount_number = amount;
	clone.Memory_allocation();

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	std::thread th[amount];
	for (clone.iter_id = 0; clone.iter_id < clone.iter_number; clone.iter_id++) {

		printf("in %d generation\n", clone.iter_id);

		for (int i = 0; i < clone.amount_number; i++) {
			int cuda_id = i % N_GPU;
			th[i] = std::thread(func, std::ref(clone), i, cuda_id);
		}

		for (int i = 0; i < clone.amount_number; i++) {
			th[i].join();
		}
		
		clone.calc_n();
		clone.update_fatherid();
		clone.outputfile();
		printf("the %dth generation is done\n", clone.iter_id);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float timeCost;
	cudaEventElapsedTime(&timeCost, start, stop);
	printf("time to generate:%f ms\n", timeCost);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}


void func(Clone& clone, int j, int cuda_id) {
	cudaSetDevice(cuda_id);
	std::random_device rd; // Non-determinstic seed source
	//std::default_random_engine rng3{ rd() };
	srand(rd());
	clone.run(j);
}
