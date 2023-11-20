#pragma once

/// constexpr power helper
template <int A, int B>
struct get_power
{
	static const int value = A * get_power<A, B - 1>::value;
};
template <int A>
struct get_power<A, 0>
{
	static const int value = 1;
};

inline constexpr int springLength = 30;
inline constexpr int springLengthDiagonal = 1.7 * 30;
inline constexpr int springLengthDiagonalSmall = 1.41 * 30;

template<int X, int Y, int Z>
struct mpInt3
{
	inline static constexpr int x = X;
	inline static constexpr int y = Y;
	inline static constexpr int z = Z;
};

template <int X, int Y, int Z, int decimalPrecision = 1>
struct mpFloat3
{
	inline static constexpr float x = float(X) / get_power<10, decimalPrecision - 1>::value;
	inline static constexpr float y = float(Y) / get_power<10, decimalPrecision - 1>::value;
	inline static constexpr float z = float(Z) / get_power<10, decimalPrecision - 1>::value;
};

template<int I>
struct mpIndex
{
	inline static constexpr int index = I;
};

template<int Count, int ParticlesInCell, int IndicesInCell, typename L, typename V, typename I, typename N>
struct BloodCellDef
{
	using List = L;
	using Vertices = V;
	using Indices = I;
	using Normals = N;
	inline static constexpr int count = Count;
	inline static constexpr int particlesInCell = ParticlesInCell;
	inline static constexpr int indicesInCell = IndicesInCell;
};

template<int Start, int End, int Length, int decimalPrecision>
struct Spring
{
	inline static constexpr int start = Start;
	inline static constexpr int end = End;
	inline static constexpr float length = float(Length) / get_power<10, decimalPrecision - 1>::value;
};