#pragma once

#include <glm/vec3.hpp>

struct cvec
{
	float x;
	float y;
	float z;

	inline constexpr float operator[](int i)
	{
		switch (i)
		{
		case 0:
			return x;
		case 1:
			return y;
		case 2:
			return z;
		}
		return 0;
	}

	inline glm::vec3 toGLM()
	{
		return { x, y, z };
	}
};