#ifdef PIC
#include "Picture.h"

void Picture::initial(int N)
{
	/* Initialize the library */
	if (!glfwInit()) return;

	window = glfwCreateWindow(width, height, "DCQ", NULL, NULL);

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	if (glewInit() != GLEW_OK) return;


	glCreateVertexArrays(1, &VAO);
	glCreateBuffers(1, &VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, 5 * sizeof(float) * N, NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));

	/*
	glCreateVertexArrays(2, VAO);
	glCreateBuffers(2, VBO);

	glBindVertexArray(VAO[0]);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, 5 * sizeof(float) * N, NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
	*/

	/*
	glBindVertexArray(VAO[1]);
	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, 5 * sizeof(float) * Nrat, NULL, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
	*/

	Shader shader = Shader(vertexPath, fragmentPath, geoPath);
	shader.use();
	shader.setFloat("cycle_r", 2 * RADIUS / BOX_L);
}

void Picture::draw(Particle& s)
{
	int N = s.GeShu;
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//painting active particle
	glBindVertexArray(VAO);
	cudaGraphicsGLRegisterBuffer(&resource, VBO, cudaGraphicsMapFlagsNone);
	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&devptr, &size, resource);
	float_to_color << <(N + 511) / 512, 512 >> > (devptr, s.device_pos, N);
	cudaGraphicsUnmapResources(1, &resource, NULL);
	glDrawArrays(GL_POINTS, 0, N);
	cudaGraphicsUnregisterResource(resource);

	/* Swap front and back buffers */
	glfwSwapBuffers(window);
	/* Poll for and process events */
	glfwPollEvents();
}


template <typename Particle>
__global__ void static float_to_color(GLfloat* ptr, Particle* p, int N)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = 5;
	if (id >= N) return;

	for (int i = 0; i < 2; i++) {
		ptr[stride * id + i] = 2 * p[id].p[i] / BOX_L - 1;
	}
	if (p[id].num0[2] == 1) {
		ptr[stride * id + 2] = 1.0f;
		ptr[stride * id + 3] = 0.0f;
		ptr[stride * id + 4] = 0.0f;
	}
	else if (p[id].num0[2] == 3) {
		ptr[stride * id + 2] = 0.0f;
		ptr[stride * id + 3] = 1.0f;
		ptr[stride * id + 4] = 0.0f;
	}
	else {
		ptr[stride * id + 2] = 0.0f;
		ptr[stride * id + 3] = 0.0f;
		ptr[stride * id + 4] = 1.0f;
	}
	/*
	if (stat->fa[id] == stat->max_cluster_info[0]) {
		ptr[stride * id + 2] = 1.0f;
		ptr[stride * id + 3] = 0.0f;
		ptr[stride * id + 4] = 0.0f;
	}
	else {
		ptr[stride * id + 2] = 1.0f;
		ptr[stride * id + 3] = 0.0f;
		ptr[stride * id + 4] = 0.0f;
	}
	*/
}

void Picture::grabPicture(int i)
{
}
#endif

