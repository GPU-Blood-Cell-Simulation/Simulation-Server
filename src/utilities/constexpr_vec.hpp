#pragma once

#include <glm/vec3.hpp>

/// <summary>
/// A vec3 alternative with a constexpr constructor
/// </summary>
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

inline cvec operator-(const cvec& v1, const cvec& v2)
{
	return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
}