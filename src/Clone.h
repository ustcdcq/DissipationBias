#pragma once
#include "Structure.h"
#include "ParametersReset.h"
#include "OperateTheProcess.h"
#include "RandomNumber.h"

#ifdef PIC
#include "Picture.h"
#endif // 

class Clone
{
public:
#ifdef PIC
	Picture pic;
	Particle p;
#endif // PIC
	int amount_number; //amount number 
	int* clone_num ; //clone number of every gene
	int* father_id ;
	float* EPR ;
	float* ratio ;
	float epr = 0.0;
	float alpha; //bias
	int iter_id; //...
	int iter_number;
	int numClonesPredicted;
	int pre_predict;

public:
	Clone();
	~Clone();
	void Memory_allocation();
	void calc_n();
	void run(int);
	void adjust_population();
	void update_fatherid();
	void draw();
	void outputfile();
};

