#include <glad/glad.h>

#include "glcontroller.cuh"

#include "../meta_factory/blood_cell_factory.hpp"
#include "../objects/vein_triangles.cuh"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/cuda_vec3.cuh"
#include "../utilities/cuda_threads.hpp"

#include <functional>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaGL.h"
#include "cuda_gl_interop.h"



namespace graphics
{
	// __global__ void calculateOffsetsKernel(float* devCudaOffsetBuffer, cudaVec3 positions)
	// {
	// 	int id = blockIdx.x * blockDim.x + threadIdx.x;
	// 	if (id >= particleCount)
	// 		return;
		
	// 	//printf("[%d] x=%.5f, y=%.5f, z=%.5f\n", id, positions.x[id], positions.y[id], positions.z[id]);
	// 	devCudaOffsetBuffer[3 * id] = positions.x[id];
	// 	devCudaOffsetBuffer[3 * id + 1] = positions.y[id];
	// 	devCudaOffsetBuffer[3 * id + 2] = positions.z[id];

	// }

	template<int bloodCellCount, int particlesInBloodCell, int particlesStart, int bloodCellTypeStart>
	__global__ void calculatePositionsKernel(float* devCudaPositionsVertices, float* devCudaPositionsOffsets, cudaVec3 positions)
	{
		int relativeId = blockIdx.x * blockDim.x + threadIdx.x;
		if (relativeId >= particlesInBloodCell * bloodCellCount)
			return;
		int id = particlesStart + relativeId;

		devCudaPositionsVertices[6 * relativeId] = positions.x[id];
		devCudaPositionsVertices[6 * relativeId + 1] = positions.y[id];
		devCudaPositionsVertices[6 * relativeId + 2] = positions.z[id];
		devCudaPositionsOffsets[3 * id] = positions.x[id];
		devCudaPositionsOffsets[3 * id + 1] = positions.y[id];
		devCudaPositionsOffsets[3 * id + 2] = positions.z[id];
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


	GLController::GLController(SingleObjectMesh& veinMesh, InstancedObjectMesh& sphereMesh, sim::SimulationController& simulationController): veinModel(veinMesh), cellSphereModel(sphereMesh, particleCount)
	{
		// Register OpenGL buffer in CUDA for vein
		HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaVeinVBOResource, veinModel.getVboBuffer(0), cudaGraphicsRegisterFlagsNone));
		HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaOffsetResource, cellSphereModel.getCudaOffsetBuffer(), cudaGraphicsRegisterFlagsNone));

		using TypeList = mp_iota_c<bloodCellTypeCount>;
		std::array<unsigned int, bloodCellTypeCount> VBOs;
		std::copy(simulationController.smallestRadiusInType.begin(), simulationController.smallestRadiusInType.end(), cellSphereRadius.begin());
		minimalRadius = *std::min_element(cellSphereRadius.begin(), cellSphereRadius.end());
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

				mp_for_each<verticeIndexList>([&](auto i)
					{
						Vertex v;
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

				using indiceIndexList = mp_iota_c<mp_size<IndiceList>::value>;
				mp_for_each<indiceIndexList>([&](auto i)
					{
						indices.push_back(mp_at_c<IndiceList, i>::value);
					});

				std::vector<glm::vec3> bloodCellInitials(BloodCellDefinition::count);
				auto initialsIterStart = simulationController.initialCellPositions.begin() + bloodCellTypesStarts[typeIndex];
				std::copy(initialsIterStart, initialsIterStart + BloodCellDefinition::count, bloodCellInitials.begin());

				bloodCellmodel[typeIndex] = MultipleObjectModel(std::move(vertices), std::move(indices), bloodCellInitials, BloodCellDefinition::count);
				VBOs[typeIndex] = bloodCellmodel[typeIndex].getVboBuffer(0);
				// Register OpenGL buffer in CUDA for blood cell
				HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&(cudaPositionsResource[typeIndex]), bloodCellmodel[typeIndex].getVboBuffer(0), cudaGraphicsRegisterFlagsNone));
				HANDLE_ERROR(cudaPeekAtLastError());

				// create diffuse color for blood cell type
				vec3 color;
				color.b = float(BloodCellDefinition::color & 0xFF) / 255.0f;
				color.g = float((BloodCellDefinition::color >> 8) & 0xFF) / 255.0f;
				color.r = float((BloodCellDefinition::color >> 16) & 0xFF) / 255.0f;
				bloodCellTypeDiffuse[typeIndex] = color;
			});
		springLines.constructSprings(VBOs);

		// Create a directional light
		directionalLight = DirLight
		{
			{
				vec3(0.4f, 0.4f, 0.4f), vec3(1, 1, 1), vec3(1, 1, 1)
			},
			vec3(0, 0, -1.0f)
		};

		// Create the shaders
		if constexpr (!useLighting)
		{
			solidColorShader = std::make_unique<SolidColorShader>();
			solidColorSphereShader = std::make_unique<SpheresSolidColorShader>();
		}
		else
		{
			phongForwardShader = std::make_unique<PhongForwardShader>();
			phongForwardSphereShader = std::make_unique<SpheresPhongForwardShader>();
		}

		veinSolidColorShader = std::make_unique<VeinSolidColorShader>();
		springShader = std::make_unique<SpringShader>();

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

	void GLController::calculatePositions(cudaVec3 positions)
	{			
		float* devCudaPositionOffsets = (float*)mapResourceAndGetPointer(cudaOffsetResource);
		using TypeList = mp_iota_c<bloodCellTypeCount>;
		mp_for_each<TypeList>([&](auto typeIndex)
			{
				// get CUDA a pointer to openGL buffer
				// jak cos to to do odkomentowania
				float* devCudaPositionVertices = (float*)mapResourceAndGetPointer(cudaPositionsResource[typeIndex]);
				using BloodCellDefinition = mp_at_c<BloodCellList, typeIndex>;
					constexpr int particlesStart = particleStarts[typeIndex];
				constexpr int bloodCellTypeStart = bloodCellTypesStarts[typeIndex];
					CudaThreads threads(BloodCellDefinition::count * BloodCellDefinition::particlesInCell);
				// translate our CUDA positions into Vertex offsets
				calculatePositionsKernel<BloodCellDefinition::count, BloodCellDefinition::particlesInCell, particlesStart, bloodCellTypeStart>
					<< <threads.blocks, threads.threadsPerBlock, 0, streams[typeIndex] >> > (devCudaPositionVertices, devCudaPositionOffsets, positions);
				HANDLE_ERROR(cudaPeekAtLastError());
				HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaPositionsResource[typeIndex], 0));
				HANDLE_ERROR(cudaPeekAtLastError());
			});
			HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaOffsetResource, 0));
	}

	void GLController::calculateTriangles(VeinTriangles triangles)
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

	void GLController::draw(Camera& camera)
	{
		// Draw particles
		if constexpr (!useLighting) // solidcolor
		{
			if (BLOOD_CELL_SPHERE_RENDER)
			{
				solidColorSphereShader->use();
				solidColorSphereShader->setMatrix("model", model);
				solidColorSphereShader->setMatrix("view", camera.getView());
				solidColorSphereShader->setMatrix("projection", projection);
				solidColorSphereShader->setFloat("sphereRadius", minimalRadius*float(boundingSpheresCoeff)/2);
				cellSphereModel.draw(solidColorSphereShader.get());
			}
			else
			{
				solidColorShader->use();
				solidColorShader->setMatrix("model", model);
				solidColorShader->setMatrix("view", camera.getView());
				solidColorShader->setMatrix("projection", projection);
				using TypeList = mp_iota_c<bloodCellTypeCount>;
				mp_for_each<TypeList>([&](auto typeIndex)
					{
						solidColorShader->setVector("Diffuse", bloodCellTypeDiffuse[typeIndex]);
						bloodCellmodel[typeIndex].draw(solidColorShader.get());
					});
			}
		}
		else
		{
			if (BLOOD_CELL_SPHERE_RENDER)
			{
				phongForwardSphereShader->use();
				phongForwardSphereShader->setMatrix("model", model);
				phongForwardSphereShader->setMatrix("view", camera.getView());
				phongForwardSphereShader->setMatrix("projection", projection);
				phongForwardSphereShader->setVector("viewPos", camera.getPosition());
				phongForwardSphereShader->setFloat("Specular", particleSpecular);
				phongForwardSphereShader->setFloat("Shininess", 32);
				phongForwardSphereShader->setLighting(directionalLight);
				phongForwardSphereShader->setFloat("sphereRadius", minimalRadius*float(boundingSpheresCoeff)/2);
				phongForwardSphereShader->setVector("Diffuse", bloodCellTypeDiffuse[0]); /// TODO
				cellSphereModel.draw(phongForwardSphereShader.get());
			}
			else
			{
				phongForwardShader->use();
				phongForwardShader->setMatrix("model", model);
				phongForwardShader->setMatrix("view", camera.getView());
				phongForwardShader->setMatrix("projection", projection);
				phongForwardShader->setVector("viewPos", camera.getPosition());
				phongForwardShader->setFloat("Specular", particleSpecular);
				phongForwardShader->setFloat("Shininess", 32);
				phongForwardShader->setLighting(directionalLight);
				using TypeList = mp_iota_c<bloodCellTypeCount>;
				mp_for_each<TypeList>([&](auto typeIndex)
					{
						phongForwardShader->setVector("Diffuse", bloodCellTypeDiffuse[typeIndex]);
						bloodCellmodel[typeIndex].draw(phongForwardShader.get());
					});
			}
		}

		if (BLOOD_CELL_SPRINGS_RENDER)
		{
			// Draw lines
			springShader->use();
			springShader->setMatrix("projection_view_model", projection * camera.getView());
			springLines.draw(springShader.get());
		}


		// Draw vein
		veinSolidColorShader->use();
		veinSolidColorShader->setMatrix("view", camera.getView());
		veinSolidColorShader->setMatrix("projection", projection);

		glDisable(GL_CULL_FACE);
		//glCullFace(GL_FRONT);
		veinModel.draw(veinSolidColorShader.get());
		//glCullFace(GL_BACK);
		glEnable(GL_CULL_FACE);
	}
}