#pragma once

#include "camera.hpp"
#include "../config/graphics.hpp"
#include "../config/simulation.hpp"
#include "inputcontroller.hpp"
#include "light.hpp"
#include "model.hpp"
#include "../objects/vein_triangles.cuh"
#include "../simulation/simulation_controller.cuh"
#include "spring_lines.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <memory>

namespace graphics
{
	/// <summary>
	/// Controls rendering of the particles
	/// </summary>
	class GLController {
	public:

		GLController(Mesh& veinMesh, InstancedObjectMesh& sphereMesh, sim::SimulationController& simulationController);
		~GLController();

		/// <summary>
		/// Translate device particle positions to OpenGL buffer
		/// </summary>
		/// <param name="positions">cuda buffer of particle positions</param>
		void calculatePositions(cudaVec3 positions);

		/// <summary>
		/// Translate device triangles positions to OpenGL buffer
		/// </summary>
		/// <param name="triangles">device data for vein triangles</param>
		void calculateTriangles(VeinTriangles triangles);

		/// <summary>
		/// Calls OpenGL rendering pipeline for current camera view
		/// </summary>
		/// <param name="camera">visualization camera</param>
		void draw(Camera& camera);

	private:
		// Particle color
		std::array<glm::vec3, bloodCellTypeCount> bloodCellTypeDiffuse; // = glm::vec3(0.8f, 0.2f, 0.2f);
		float particleSpecular = 0.6f;

		// Uniform matrices
		glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f));
		glm::mat4 projection = glm::perspective(glm::radians<float>(45.0f), static_cast<float>(windowWidth) / windowHeight, 0.1f, depth * 100);

		MultipleObjectModel bloodCellmodel[bloodCellTypeCount];
		Model veinModel;
		InstancedModel cellSphereModel;

		std::array<float, bloodCellCount> cellSphereRadius;

		DirLight directionalLight;

		SpringLines springLines;

		std::unique_ptr<Shader> solidColorShader;
		std::unique_ptr<Shader> geometryPassShader;
		std::unique_ptr<Shader> phongDeferredShader;
		std::unique_ptr<Shader> phongForwardShader;
		std::unique_ptr<Shader> veinSolidColorShader;
		std::unique_ptr<Shader> springShader;
		std::unique_ptr<Shader> solidColorSphereShader;
		std::unique_ptr<Shader> phongForwardSphereShader;
		
		/// <summary>
		/// Deferred shading gBuffer
		/// </summary>
		unsigned int gBuffer;
		/// <summary>
		/// Deferred shading textures
		/// </summary>
		unsigned int gPosition, gNormal, gAlbedoSpec;

		cudaGraphicsResource_t cudaPositionsResource[bloodCellTypeCount];
		cudaGraphicsResource_t cudaVeinVBOResource;
		cudaGraphicsResource_t cudaVeinEBOResource;
		cudaGraphicsResource_t cudaOffsetResource;

		cudaStream_t streams[bloodCellTypeCount];
	};
}