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

inline constexpr int springLength = 10;
inline constexpr int springLengthDiagonal = 1.7 * springLength;
inline constexpr int springLengthDiagonalSmall = 1.41 * springLength;

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

/// <summary>
/// Represents blood cell type definition
/// </summary>
/// <typeparam name="L">List of springs</typeparam>
/// <typeparam name="V">List of vertices</typeparam>
/// <typeparam name="I">List of indices</typeparam>
/// <typeparam name="N">List of normals</typeparam>
/// <typeparam name="Count">Amount of blood cells in given type</typeparam>
/// <typeparam name="ParticlesInCell">Size of each blood cell in given type</typeparam>
/// <typeparam name="IndicesInCell">Number of indices for given type</typeparam>
/// <typeparam name="Color">Color of blood cell in given type</typeparam>
template<int Count, int ParticlesInCell, int IndicesInCell, int Color, typename L, typename V, typename I, typename N>
struct BloodCellDef
{
	using List = L;
	using Vertices = V;
	using Indices = I;
	using Normals = N;
	inline static constexpr int color = Color;
	inline static constexpr int count = Count;
	inline static constexpr int particlesInCell = ParticlesInCell;
	inline static constexpr int indicesInCell = IndicesInCell;
};

/// <summary>
/// Represents spring definition
/// </summary>
/// <typeparam name="Start">Primary vertex for spring</typeparam>
/// <typeparam name="End">Terminal vertex for spring</typeparam>
/// <typeparam name="Length">Spring initial lenght</typeparam>
/// <typeparam name="decimalPrecision">Decimal float template precision</typeparam>
template<int Start, int End, int Length, int decimalPrecision>
struct Spring
{
	inline static constexpr int start = Start;
	inline static constexpr int end = End;
	inline static constexpr float length = float(Length) / get_power<10, decimalPrecision - 1>::value;
};