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

template <int X, int Y, int Z>
struct mp_int3
{
	inline static constexpr int x = X;
	inline static constexpr int y = Y;
	inline static constexpr int z = Z;
};

template <int X, int Y, int Z, int decimalPrecision = 7>
struct mp_float3
{
	inline static constexpr float x = float(X) / get_power<10, decimalPrecision - 1>::value;
	inline static constexpr float y = float(Y) / get_power<10, decimalPrecision - 1>::value;
	inline static constexpr float z = float(Z) / get_power<10, decimalPrecision - 1>::value;
};

template <int Value, int decimalPrecision = 7>
struct mp_float
{
	inline static constexpr float value = float(Value) / get_power<10, decimalPrecision - 1>::value;
};