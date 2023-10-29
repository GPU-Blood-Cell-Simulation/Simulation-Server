#pragma once

#include "../config/blood_cells_definition.hpp"
#include "../config/simulation.hpp"

#include <array>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/utility.hpp>


namespace
{
	using namespace boost::mp11;

	// Helper meta-function for adding particles
	template<class State, class Def>
	using Add = mp_int<State::value + Def::count * Def::particlesInCell>;

	// Helper meta-function for calculating the graph size
	template<class State, class Def>
	using AddSquared = mp_int<State::value + Def::particlesInCell * Def::particlesInCell>;

	// Helper meta-functor for filtering the user-provided list
	template<class Def1, class Def2>
	struct IsDuplicate
	{
		static constexpr bool value = (Def1::particlesInCell == Def2::particlesInCell) && std::is_same_v<typename Def1::List, typename Def2::List>;
	};

	// Helper meta-function for folding the blood cell list for every distinct type
	template<class State, class Def>
	using AddDistinctTypes = BloodCellDef
		<
		IsDuplicate<State, Def>::value ? State::count + Def::count : State::count,
		State::particlesInCell,
		typename State::List
		>;

	// In order to merge all definitions of the same type, we first fold the list for every definitions using
	// a custom meta-function and then we remove all the duplicates
	template<int i>
	struct FoldedBloodCellList
	{
		using type = mp_push_back
			<
			typename FoldedBloodCellList<i - 1>::type,
			mp_fold
				<
				UserDefinedBloodCellList,
				BloodCellDef
					<
					0,
					mp_at_c<UserDefinedBloodCellList, i>::particlesInCell,
					typename mp_at_c<UserDefinedBloodCellList, i>::List
					>,
				AddDistinctTypes
				>
			>;
	};

	// Recursion end
	template<>
	struct FoldedBloodCellList<-1>
	{
		using type = mp_list<>;
	};

	// Filter out duplicated folded types
	using UniqueBloodCellList = mp_unique_if<FoldedBloodCellList<mp_size<UserDefinedBloodCellList>::value - 1>::type, IsDuplicate>;

	inline constexpr bool isPowerOfTwo(int n)
	{
		return n & (n - 1);
	}
	
	// Heuristics: powers of 2 should be at the beginning
	inline constexpr bool orderBloodCells(int particlesInCell1, int particlesInCell2)
	{
		// Power of two?
		if (isPowerOfTwo(particlesInCell1) && !isPowerOfTwo(particlesInCell2))
			return true;
		if (!isPowerOfTwo(particlesInCell1) && isPowerOfTwo(particlesInCell2))
			return false;

		// Even?
		if (particlesInCell1 & 0 && particlesInCell2 & 1)
			return true;
		if (particlesInCell1 & 1 && particlesInCell2 & 0)
			return false;

		// Default: we don't change the order
		return true;
	}

	// Helper meta-functor for ordering the blood cell definitions
	template<class Def1, class Def2>
	struct BloodCellComparator
	{
		static constexpr bool value = orderBloodCells(Def1::particlesInCell, Def2::particlesInCell);
	};

	// Sort the blood cell definitions
	using BloodCellList = mp_sort<UniqueBloodCellList, BloodCellComparator>;

	// Check the final list
	static_assert(mp_size<BloodCellList>::value <= maxCudaStreams, "Max number of streams exceeded");

	// Particle count
	inline constexpr int particleCount = mp_fold<BloodCellList, mp_int<0>, Add>::value;

	// Total particle graph size
	inline constexpr int totalGraphSize = mp_fold<BloodCellList, mp_int<0>, AddSquared>::value;

	// Distinct blood cell types
	inline constexpr int bloodCellTypeCount = mp_size<BloodCellList>::value;

	// Fill the particles start array
	inline constexpr auto particlesStartsGenerator = []()
	{
		std::array<int, bloodCellTypeCount> arr{};

		// Iterate over user-provided definition (particle type)
		using IndexList = mp_iota_c<mp_size<BloodCellList>::value>;
		int state = 0;
		mp_for_each<IndexList>([&](auto i)
		{
			using BloodCellDefinition = mp_at_c<BloodCellList, i>;
			arr[i] = state;
			state += BloodCellDefinition::count * BloodCellDefinition::particlesInCell;
		});
		return arr;
	};


	// Determine where in the device array a particular stream should start its job (calculate accumulated particlesInCell sums)
	inline constexpr auto particlesStarts = particlesStartsGenerator();

	// Fill the accumulated graph sizes array
	inline constexpr auto graphSizesGenerator = []()
	{
		std::array<int, bloodCellTypeCount> arr{};

		// Iterate over user-provided definition (particle type)
		using IndexList = mp_iota_c<mp_size<BloodCellList>::value>;
		int state = 0;
		mp_for_each<IndexList>([&](auto i)
		{
			using BloodCellDefinition = mp_at_c<BloodCellList, i>;
			arr[i] = state;
			state += BloodCellDefinition::particlesInCell * BloodCellDefinition::particlesInCell;
		});
		return arr;
	};

	// Graph sizes for each type
	inline constexpr auto accumulatedGraphSizes = graphSizesGenerator();

	// Fill the particle neighborhood graph
	inline constexpr auto springGraphGenerator = [&]()
	{
		std::array<float, totalGraphSize> arr{};

		// Iterate over user-provided definition (particle type)
		using IndexList = mp_iota<mp_size<BloodCellList>>;
		mp_for_each<IndexList>([&](auto i)
		{
			using BloodCellDefinition = mp_at_c<BloodCellList, i>;
			constexpr int particlesInThisCell = BloodCellDefinition::particlesInCell;

			using SpringList = typename BloodCellDefinition::List;
			constexpr int springCount = mp_size<SpringList>::value;
			constexpr int graphStart = accumulatedGraphSizes[i];

			// For every definition iterate over its particle count
			using IndexListPerBloodCell = mp_iota_c<springCount>;
			mp_for_each<IndexListPerBloodCell>([&](auto j)
			{
				using SpringDefinition = mp_at_c<SpringList, j>;

				// Check if the spring definition is well formed
				static_assert(SpringDefinition::start >= 0, "Ill-formed spring definition");
				static_assert(SpringDefinition::end < BloodCellDefinition::particlesInCell, "Ill-formed spring definition");

				// Fill the graph
				arr[graphStart + SpringDefinition::start * particlesInThisCell + SpringDefinition::end] = SpringDefinition::length * springLengthCoefficient;
				arr[graphStart + SpringDefinition::end * particlesInThisCell + SpringDefinition::start] = SpringDefinition::length * springLengthCoefficient;
			});
		});
		return arr;
	};

	inline constexpr auto springGraph = springGraphGenerator();
}