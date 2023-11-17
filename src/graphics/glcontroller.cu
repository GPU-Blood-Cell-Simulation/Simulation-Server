#include <glad/glad.h>

#include "glcontroller.cuh"

#include "../meta_factory/blood_cell_factory.hpp"
#include "../objects/vein_triangles.cuh"
#include "../utilities/cuda_handle_error.cuh"
#include "../utilities/cuda_vec3.cuh"

#ifdef _WIN32
#include <windows.h>
#endif

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaGL.h"
#include "cuda_gl_interop.h"


namespace graphics
{
	__global__ void calculateOffsetsKernel(float* devCudaOffsetBuffer, cudaVec3 positions)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= particleCount)
			return;

		// Insert any debug position changes here

		devCudaOffsetBuffer[3 *id] = positions.x[id];
		devCudaOffsetBuffer[3 * id + 1] = positions.y[id];
		devCudaOffsetBuffer[3 * id + 2] = positions.z[id];

	}

	__global__ void calculateVerticesKernel(float* devVeinVBOBuffer, cudaVec3 positions, int vertexCount)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= vertexCount)
			return;

		// Insert any debug position changes here

		float3 v = positions.get(id);
		devVeinVBOBuffer[8 * id] = v.x;
		devVeinVBOBuffer[8 * id + 1] = v.y;
		devVeinVBOBuffer[8 * id + 2] = v.z;
		devVeinVBOBuffer[8 * id + 3] = 0;
		devVeinVBOBuffer[8 * id + 4] = 0;
		devVeinVBOBuffer[8 * id + 5] = 0;
		devVeinVBOBuffer[8 * id + 6] = 0;
		devVeinVBOBuffer[8 * id + 7] = 0;
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


	graphics::GLController::GLController(Mesh veinMesh) :
		veinModel(veinMesh), springLines(particleModel.getCudaOffsetBuffer())
	{
		// Register OpenGL buffer in CUDA
		HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaOffsetResource, particleModel.getCudaOffsetBuffer(), cudaGraphicsRegisterFlagsNone));
		HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaVeinVBOResource, veinModel.getTopVboBuffer(), cudaGraphicsRegisterFlagsNone));
		HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaVeinEBOResource, veinModel.getTopEboBuffer(), cudaGraphicsRegisterFlagsNone));
		
		// Create a directional light
		directionalLight = DirLight
		{
			{
				vec3(0.4f, 0.4f, 0.4f), vec3(1, 1, 1), vec3(1, 1, 1)
			},
			vec3(0, 0, -1.0f)
		};

		//// Set up deferred shading
		//// Set up OpenGL frame buffers
		//glGenFramebuffers(1, &gBuffer);
		//glBindFramebuffer(GL_FRAMEBUFFER, gBuffer);

		//unsigned int gPosition, gNormal;
		//// position color buffer
		//glGenTextures(1, &gPosition);
		//glBindTexture(GL_TEXTURE_2D, gPosition);
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gPosition, 0);
		//// normal color buffer
		//glGenTextures(1, &gNormal);
		//glBindTexture(GL_TEXTURE_2D, gNormal);
		//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gNormal, 0);
		
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
		//geometryPassShader = std::make_shared<Shader>(GeometryPassShader(gBuffer));
		//phongDeferredShader = std::make_shared<Shader>(PhongDeferredShader(gPosition, gNormal));
		cylinderSolidColorShader = std::make_unique<Shader>(CylinderSolidColorShader());
		springShader = std::make_unique<Shader>(SpringShader());
	}

	void graphics::GLController::calculateOffsets(cudaVec3 positions)
	{
		// get CUDA a pointer to openGL buffer
		/*float* devCudaOffsetBuffer = 0;
		size_t numBytes;

		HANDLE_ERROR(cudaGraphicsMapResources(1, &cudaOffsetResource, 0));
		HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devCudaOffsetBuffer, &numBytes, cudaOffsetResource));*/


		// get CUDA a pointer to openGL buffer
		float* devCudaOffsetBuffer = (float*)mapResourceAndGetPointer(cudaOffsetResource);
		
		// translate our CUDA positions into Vertex offsets
		int threadsPerBlock = particleCount > 1024 ? 1024 : particleCount;
		int blocks = (particleCount + threadsPerBlock - 1) / threadsPerBlock;
		calculateOffsetsKernel << <blocks, threadsPerBlock >> > (devCudaOffsetBuffer, positions);

		HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaOffsetResource, 0));
	}

	void graphics::GLController::calculateTriangles(VeinTriangles triangles)
	{
		// map vertices
		float* vboPtr = (float*)mapResourceAndGetPointer(cudaVeinVBOResource);
		int threadsPerBlock = triangles.vertexCount > 1024 ? 1024 : triangles.vertexCount;
		int blocks = (triangles.vertexCount + threadsPerBlock - 1) / threadsPerBlock;
		calculateVerticesKernel << <blocks, threadsPerBlock >> > (vboPtr, triangles.positions, triangles.vertexCount);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaVeinVBOResource, 0));

		// map indices
		/*unsigned int* eboPtr = (unsigned int*)mapResourceAndGetPointer(cudaVeinEBOResource);
		cudaMemcpy(eboPtr, triangles.indices, triangles.triangleCount * 3 * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
		HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaVeinEBOResource, 0));*/

	}

	void graphics::GLController::draw()
	{

		// Draw particles

		if constexpr (!useLighting) // solidcolor
		{
			solidColorShader->use();
			solidColorShader->setMatrix("model", model);
			solidColorShader->setMatrix("view", camera.getView());
			solidColorShader->setMatrix("projection", projection);
			particleModel.draw(solidColorShader.get());
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

			particleModel.draw(phongForwardShader.get());
		}

		// Draw lines

		springShader->use();
		springShader->setMatrix("projection_view_model", projection * camera.getView());
		springLines.draw(springShader.get());

		// Draw vein

		cylinderSolidColorShader->use();
		cylinderSolidColorShader->setMatrix("view", camera.getView());
		cylinderSolidColorShader->setMatrix("projection", projection);

		glCullFace(GL_FRONT);
		veinModel.draw(cylinderSolidColorShader.get(), false);
		glCullFace(GL_BACK);

		/////////////////////////////////////////////////////////////////////////////////////////////
		//// Deferred shading - not working yet

		//// Geometry pass

		//geometryPassShader->use();
		//geometryPassShader->setMatrix("model", model);
		//geometryPassShader->setMatrix("view", camera.getView());
		//geometryPassShader->setMatrix("projection", projection);

		//glEnable(GL_DEPTH_TEST);

		//particleModel.draw(geometryPassShader);

		//// Phong Lighting Pass

		//glDisable(GL_DEPTH_TEST);

		//phongDeferredShader->use();
		//phongDeferredShader->setVector("viewPos", camera.getPosition());
		//phongDeferredShader->setVector("Diffuse", particleDiffuse);
		//phongDeferredShader->setFloat("Specular", particleSpecular);
		//phongDeferredShader->setFloat("Shininess", 32);

		//phongDeferredShader->setLighting(directionalLight);

		//// copy content of geometry's depth buffer to default framebuffer's depth buffer
		//// ----------------------------------------------------------------------------------
		//glBindFramebuffer(GL_READ_FRAMEBUFFER, gBuffer);
		//glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // write to default framebuffer
		//// blit to default framebuffer. Note that this may or may not work as the internal formats of both the FBO and default framebuffer have to match.
		//// the internal formats are implementation defined. This works on all of my systems, but if it doesn't on yours you'll likely have to write to the 		
		//// depth buffer in another shader stage (or somehow see to match the default framebuffer's internal format with the FBO's internal format).
		//glBlitFramebuffer(0, 0, windowWidth, windowHeight, 0, 0, windowWidth, windowHeight, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
		//glBindFramebuffer(GL_FRAMEBUFFER, 0);

		//// Render quad
		//// TODO: optimization - don't generate buffers each frame
		//unsigned int quadVAO = 0;
		//unsigned int quadVBO = 0;

		//float quadVertices[] = {
		//	// positions        // texture Coords
		//	-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
		//	-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		//		1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
		//		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		//};
		//// setup plane VAO
		//glGenVertexArrays(1, &quadVAO);
		//glGenBuffers(1, &quadVBO);
		//glBindVertexArray(quadVAO);
		//glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		//glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		//glEnableVertexAttribArray(0);
		//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		//glEnableVertexAttribArray(1);
		//glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
		//glBindVertexArray(quadVAO);
		//glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		//glBindVertexArray(0);
		//glEnable(GL_DEPTH_TEST);
		/////////////////////////////////////////////////////////////////////////////////////////////
	}
}