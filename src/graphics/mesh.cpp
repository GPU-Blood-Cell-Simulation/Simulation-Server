#include "mesh.hpp"

#include "../meta_factory/blood_cell_factory.hpp"

#include <glad/glad.h>
#include <memory>
#include <algorithm>


Mesh::Mesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices) :
	vertices(vertices), indices(indices) {}

void Mesh::setupMesh()
{
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(int),
		&indices[0], GL_STATIC_DRAW);

	// vertex positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
	// vertex normals
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));

	glBindVertexArray(0);
}

unsigned int Mesh::getVBO()
{
	return VBO;
}

unsigned int Mesh::getEBO()
{
	return EBO;
}

void Mesh::setVertexOffsetAttribute()
{
	glBindVertexArray(VAO);
	// instance offset vectors
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	glVertexAttribDivisor(2, 1);
	glBindVertexArray(0);
}

SingleObjectMesh::SingleObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices)
	: Mesh(std::move(vertices), std::move(indices))
{
	setupMesh();
}

void Mesh::draw(const Shader* shader) const
{
	// draw mesh
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);
}

MultiObjectMesh::MultiObjectMesh(std::vector<Vertex>&& vertices, std::vector<unsigned int>&& indices, std::vector<glm::vec3>& initialPositions, unsigned int objectCount)
	: Mesh(std::move(vertices), std::move(indices))
{
	this->objectCount = objectCount;
	prepareMultipleObjects(initialPositions);
	setupMesh();
}

void MultiObjectMesh::prepareMultipleObjects(std::vector<glm::vec3>& initialPositions)
{
	unsigned int indicesCount = indices.size() * initialPositions.size();
	unsigned int initialVerticesCount = vertices.size();

	std::vector<unsigned int> newIndices(indicesCount);
	for (unsigned int i = 0; i < initialPositions.size(); ++i) {
		std::transform(indices.cbegin(), indices.cend(), newIndices.begin() + i * indices.size(),
			[&](unsigned int indice) {
				return indice + i * initialVerticesCount;
			});
	}
	indices = std::vector<unsigned int>(indicesCount);
	std::move(newIndices.begin(), newIndices.end(), indices.begin());

	unsigned int verticesCount = vertices.size() * initialPositions.size();
	std::vector<Vertex> newVertices(verticesCount);

	for (unsigned int i = 0; i < initialPositions.size(); ++i) {
		

		std::transform(vertices.cbegin(), vertices.cend(),
			(newVertices.begin() + i * vertices.size()),
			[&](Vertex v) {
				Vertex v2;
				v2.normal = v.normal;
				// v2.position = v.position + initialPositions[i];
				v2.position.x = v.position.x + initialPositions[i].x;
				v2.position.y = v.position.y + initialPositions[i].y;
				v2.position.z = v.position.z + initialPositions[i].z;
				return v2;

			});
	}
	vertices = std::vector<Vertex>(verticesCount);
	std::move(newVertices.begin(), newVertices.end(), vertices.begin());

}

void MultiObjectMesh::DuplicateObjects(std::vector<glm::vec3>& initialPositions)
{
	for (unsigned int i = 0; i < initialPositions.size(); ++i) {
		std::transform(vertices.cbegin(), vertices.cend(),
			(!i ? vertices.begin() : vertices.end()),
			[&](Vertex v) {
				Vertex v2;
				v2.normal = v.normal;
				v2.position = v.position + initialPositions[i];
				return v2;
			});
	}

	for (unsigned int i = 1; i < initialPositions.size(); ++i) {
		std::transform(indices.cbegin(), indices.cend(), indices.end(),
			[&](unsigned int indice) {
				return indice + i * objectCount;
			});
	}
}

void InstancedObjectMesh::draw(const Shader* shader) const
{
	// draw mesh
	glBindVertexArray(VAO);
	glDrawElementsInstanced(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0, instancesCount);
	glBindVertexArray(0);
}
