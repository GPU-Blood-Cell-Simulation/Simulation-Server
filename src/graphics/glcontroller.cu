#include <glad/glad.h>

#include "glcontroller.cuh"

#include "../meta_factory/blood_cell_factory.hpp"
#include "../objects/vein_triangles.cuh"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/cuda_vec3.cuh"
#include "../utilities/cuda_threads.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaGL.h"
#include "cuda_gl_interop.h"
#include <functional>

namespace graphics
{

	template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int bloodCellTypeStart>
	__global__ void calculatePositionsKernel(float* devCudaPositionsBuffer, cudaVec3 positions)
	{
		int relativeId = blockIdx.x * blockDim.x + threadIdx.x;
		if (relativeId >= particlesInBloodCell * bloodCellCount)
			return;
		int id = particlesStart + relativeId;

		devCudaPositionsBuffer[6 * relativeId] = positions.x[id];
		devCudaPositionsBuffer[6 * relativeId + 1] = positions.y[id];
		devCudaPositionsBuffer[6 * relativeId + 2] = positions.z[id];

		// normals are not modified at the moment
		//devCudaPositionsBuffer[6 * id + 3] = 0;
		//devCudaPositionsBuffer[6 * id + 4] = 0;
		//devCudaPositionsBuffer[6 * id + 5] = 0;
	}

	__global__ void calculateTriangleVerticesKernel(float* devVeinVBOBuffer, cudaVec3 positions, int vertexCount)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= vertexCount)
			return;

		// Insert any debug position changes here
		float3 v = positions.get(id);
		devVeinVBOBuffer[6 * id] = v.x;
		devVeinVBOBuffer[6 * id + 1] = v.y;
		devVeinVBOBuffer[6 * id + 2] = v.z;
		devVeinVBOBuffer[6 * id + 3] = 0;
		devVeinVBOBuffer[6 * id + 4] = 0;
		devVeinVBOBuffer[6 * id + 5] = 0;
	}

	void* mapResourceAndGetPointer(cudaGraphicsResource_t resource)
	{
		// get CUDA a pointer to openGL buffer
		void* resourceBuffer = 0;
		size_t numBytes;

		HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, 0));
		HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&resourceBuffer, &numBytes, resource));
		return resourceBuffer;
	}


	graphics::GLController::GLController( Mesh& veinMesh, std::vector<glm::vec3>& initialPositions)
	{
		veinModel.addMesh(veinMesh);
		// Register OpenGL buffer in CUDA for vein
		HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaVeinVBOResource, veinModel.getVboBuffer(0), cudaGraphicsRegisterFlagsNone));
		//HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaVeinEBOResource, veinModel.getEboBuffer(0), cudaGraphicsRegisterFlagsNone));

		using TypeList = mp_iota_c<bloodCellTypeCount>;
		std::array<unsigned int, bloodCellTypeCount> VBOs;
		std::cout << "Before mp11 loop\n";
		mp_for_each<TypeList>([&](auto typeIndex) 
		{
			std::vector<Vertex> vertices;
			std::vector<unsigned int> indices;
			using BloodCellDefinition = mp_at_c<BloodCellList, typeIndex>;
			constexpr int verticesCount = BloodCellDefinition::particlesInCell;

			using verticeIndexList = mp_iota_c<verticesCount>;

			using VerticeList = typename BloodCellDefinition::Vertices;
			using NormalList = typename BloodCellDefinition::Normals;
			using IndiceList = typename BloodCellDefinition::Indices;

			std::cout << "Before vertice index list\n";
			mp_for_each<verticeIndexList>([&](auto i)
				{
					Vertex v = Vertex();
					v.position = glm::vec3(
						mp_at_c<VerticeList, i>::x,
						mp_at_c<VerticeList, i>::y,
						mp_at_c<VerticeList, i>::z
					);
					v.normal = glm::vec3(
						mp_at_c<NormalList, i>::x,
						mp_at_c<NormalList, i>::y,
						mp_at_c<NormalList, i>::z
					);
					vertices.push_back(v);
				});
			std::cout << "After vertice index list\n";

			using indiceIndexList = mp_iota_c<mp_size<IndiceList>::value>;
			mp_for_each<indiceIndexList>([&](auto i)
				{
					indices.push_back(mp_at_c<IndiceList, i>::value);
				});
			std::cout << "After index list loop\n";

			std::vector<glm::vec3> bloodCellInitials(BloodCellDefinition::count);
			auto initialsIterStart = initialPositions.begin() + bloodCellTypesStarts[typeIndex];
			std::copy(initialsIterStart, initialsIterStart + BloodCellDefinition::count, bloodCellInitials.begin());

			std::cout << "Middle\n";

			bloodCellmodel[typeIndex] = MultipleObjectModel(std::move(vertices), std::move(indices), bloodCellInitials, BloodCellDefinition::count);
			std::cout << "Multiple object made\n";
			VBOs[typeIndex] = bloodCellmodel[typeIndex].getVboBuffer(0);
			std::cout << "Get VBO\n";
			// Register OpenGL buffer in CUDA for blood cell
			HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&(cudaPositionsResource[typeIndex]), bloodCellmodel[typeIndex].getVboBuffer(0), cudaGraphicsRegisterFlagsNone));
			std::cout << "Buffer registered\n";
			HANDLE_ERROR(cudaPeekAtLastError());

			std::cout << "End\n";
		});
		std::cout << "After mp11 loop\n";
		springLines.constructSprings(VBOs);

		// Create a directional light
		directionalLight = DirLight
		{
			{
				vec3(0.4f, 0.4f, 0.4f), vec3(1, 1, 1), vec3(1, 1, 1)
			},
			vec3(0, 0, -1.0f)
		};

		// tell OpenGL which color attachments we'll use (of this framebuffer) for rendering 
		unsigned int attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
		glDrawBuffers(2, attachments);

		// create and attach depth buffer (renderbuffer)
		unsigned int rboDepth;
		glGenRenderbuffers(1, &rboDepth);
		glBindRenderbuffer(GL_RENDERBUFFER, rboDepth);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, windowWidth, windowHeight);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboDepth);

		// finally check if framebuffer is complete
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			std::cout << "Framebuffer not complete!" << std::endl;
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		// Create the shaders
		solidColorShader = std::make_unique<Shader>(SolidColorShader());
		phongForwardShader = std::make_unique<Shader>(PhongForwardShader());
		cylinderSolidColorShader = std::make_unique<Shader>(CylinderSolidColorShader());
		springShader = std::make_unique<Shader>(SpringShader());


		// Create streams
		for (int i = 0; i < bloodCellTypeCount; i++)
		{
			streams[i] = cudaStream_t();
			HANDLE_ERROR(cudaStreamCreate(&streams[i]));
		}
	}

	GLController::~GLController()
	{
		for (int i = 0; i < bloodCellTypeCount; i++)
		{
			HANDLE_ERROR(cudaStreamDestroy(streams[i]));
		}
	}

	void graphics::GLController::calculatePositions(cudaVec3 positions)
	{
		using TypeList = mp_iota_c<bloodCellTypeCount>;
		mp_for_each<TypeList>([&](auto typeIndex) 
		{
			// get CUDA a pointer to openGL buffer
			// jak cos to to do odkomentowania
			float* devCudaPositionBuffer = (float*)mapResourceAndGetPointer(cudaPositionsResource[typeIndex]);
			using BloodCellDefinition = mp_at_c<BloodCellList, typeIndex>;

			constexpr int particlesStart = particlesStarts[typeIndex];
			constexpr int bloodCellTypeStart = bloodCellTypesStarts[typeIndex];

			CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
			// translate our CUDA positions into Vertex offsets
			calculatePositionsKernel<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellTypeStart>
				<< <threads.blocks, threads.threadsPerBlock, 0, streams[typeIndex] >> > (devCudaPositionBuffer, positions);
			HANDLE_ERROR(cudaPeekAtLastError());
			HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaPositionsResource[typeIndex], 0));
			HANDLE_ERROR(cudaPeekAtLastError());
		});
	}

	void graphics::GLController::calculateTriangles(VeinTriangles triangles)
	{
		// map vertices
		float* vboPtr = (float*)mapResourceAndGetPointer(cudaVeinVBOResource);
		int threadsPerBlock = triangles.vertexCount > 1024 ? 1024 : triangles.vertexCount;
		int blocks = (triangles.vertexCount + threadsPerBlock - 1) / threadsPerBlock;
		calculateTriangleVerticesKernel << <blocks, threadsPerBlock >> > (vboPtr, triangles.positions, triangles.vertexCount);
		HANDLE_ERROR(cudaPeekAtLastError());
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaVeinVBOResource, 0));
		HANDLE_ERROR(cudaPeekAtLastError());
	}

	void graphics::GLController::draw(Camera& camera)
	{

		// Draw particles
		if constexpr (!useLighting) // solidcolor
		{
			solidColorShader->use();
			solidColorShader->setMatrix("model", model);
			solidColorShader->setMatrix("view", camera.getView());
			solidColorShader->setMatrix("projection", projection);

			using TypeList = mp_iota_c<bloodCellTypeCount>;
			mp_for_each<TypeList>([&](auto typeIndex)
				{
					bloodCellmodel[typeIndex].draw(solidColorShader.get());
				});
		}
		else
		{
			phongForwardShader->use();
			phongForwardShader->setMatrix("model", model);
			phongForwardShader->setMatrix("view", camera.getView());
			phongForwardShader->setMatrix("projection", projection);

			phongForwardShader->setVector("viewPos", camera.getPosition());
			phongForwardShader->setVector("Diffuse", particleDiffuse);
			phongForwardShader->setFloat("Specular", particleSpecular);
			phongForwardShader->setFloat("Shininess", 32);

			phongForwardShader->setLighting(directionalLight);
			using TypeList = mp_iota_c<bloodCellTypeCount>;
			mp_for_each<TypeList>([&](auto typeIndex)
				{
					bloodCellmodel[typeIndex].draw(phongForwardShader.get());
				});
		}

		if (BLOOD_CELL_SPRINGS_RENDER)
		{
			// Draw lines
			springShader->use();
			springShader->setMatrix("projection_view_model", projection * camera.getView());
			springLines.draw(springShader.get());
		}

		// Draw vein
		cylinderSolidColorShader->use();
		cylinderSolidColorShader->setMatrix("view", camera.getView());
		cylinderSolidColorShader->setMatrix("projection", projection);

		glCullFace(GL_FRONT);
		veinModel.draw(cylinderSolidColorShader.get());
		glCullFace(GL_BACK);
	}
}