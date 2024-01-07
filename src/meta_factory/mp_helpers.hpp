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