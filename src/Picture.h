#pragma once
#ifdef PIC
#include "cuda_runtime.h"
#define  GLEW_STATIC
#include "./GLEW/include/GL/glew.h"
#include "./GLFW/include/GLFW/glfw3.h"
#include "shader.h"
#include "structure.h"
#include "cuda_gl_interop.h"
#include "MacroDefinition.h"

class Picture
{
	char* vertexPath = "./Shader/geo/passive_particle_2D.vs";
	char* fragmentPath = "./Shader/geo/passive_particle_2D.frag";
	char* geoPath = "./Shader/geo/passive_particle_2D.gs";
public:
	cudaGraphicsResource* resource;
	size_t size;
	GLfloat* devptr;

	GLFWwindow* window;
	float width = 800;
	float height = 800;

public:
	unsigned int VBO;
	unsigned int VAO;

public:
	void initial(int N);
	void draw(Particle& s);
	void grabPicture(int i);
};

template <typename Particle>
__global__ void static float_to_color(GLfloat* ptr, Particle* p, int N);
#endif // PIC

