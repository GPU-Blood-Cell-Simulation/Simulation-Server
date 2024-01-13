#pragma once
#include "../graphics/mesh.hpp"
#include "../meta_factory/blood_cell_factory.hpp"
#include <vector>

namespace SphereGenerator
{
    InstancedObjectMesh createMesh(int parallels, int meridians, float radius, int instancesCount)
    {
        const float PI = 3.1415926f;
	    std::vector<Vertex> vertices;
	    std::vector<unsigned int> indices;

	    vertices.push_back(Vertex{ {0,radius,0},{0,1,0} });

	    for (int i = 0; i < parallels - 1; ++i)
	    {
	    	float phi = PI * float(i + 1) / float(parallels);
	    	for (int j = 0; j < meridians; ++j)
		    {
		    	float theta = 2 * PI * float(j) / float(meridians);

	    		float x = sin(phi) * cos(theta);
    			float y = cos(phi);
    			float z = sin(phi) * sin(theta);

			    glm::vec3 normal{ x,y,z };
			    vertices.push_back(Vertex{ radius * normal, normal });
		    }
	    }

	    vertices.push_back(Vertex{ {0,0,-radius},{0,-1,0} });


		for (int i = 0; i < meridians; ++i)
  		{
   			int i0 = i + 1;
    		int i1 = (i + 1) % meridians + 1;
			indices.push_back(0);
			indices.push_back(i1);
			indices.push_back(i0);
    		i0 = i + meridians * (parallels - 2) + 1;
    		i1 = (i + 1) % meridians + meridians * (parallels - 2) + 1;
			indices.push_back((parallels - 1)*meridians + 1);
			indices.push_back(i0);
			indices.push_back(i1);
  		}

		for (int j = 0; j < parallels - 2; j++)
  		{
    		int j0 = j * meridians + 1;
    		int j1 = (j + 1) * meridians + 1;
    		for (int i = 0; i < meridians; i++)
	    	{
    	  	int i0 = j0 + i;
      		int i1 = j0 + (i + 1) % meridians;
      		int i2 = j1 + (i + 1) % meridians;
      		int i3 = j1 + i;
			indices.push_back(i0);
			indices.push_back(i2);
			indices.push_back(i3);
			indices.push_back(i1);
			indices.push_back(i2);
			indices.push_back(i0);
    	}
  	}

	    return InstancedObjectMesh(std::move(vertices), std::move(indices), instancesCount);
    }
}