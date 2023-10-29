#pragma once

template<int Count, int ParticlesInCell, typename L>
struct BloodCellDef
{
	using List = L;
	inline static constexpr int count = Count;
	inline static constexpr int particlesInCell = ParticlesInCell;
};

template<int Start, int End, int Length>
struct Spring
{
	inline static constexpr int start = Start;
	inline static constexpr int end = End;
	inline static constexpr float length = Length;
};