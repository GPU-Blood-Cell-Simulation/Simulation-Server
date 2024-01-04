#pragma once

#include "../config/blood_cells_definition.hpp"
#include "../config/simulation.hpp"

#include <array>
#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/utility.hpp>
#include <glm/vec3.hpp>

namespace
{
	using namespace boost::mp11;

	/// <summary>
	/// Helper meta-function for adding particles
	/// </summary>
	/// <typeparam name="State">Actual state value</typeparam>
	/// <typeparam name="Def">Consecutive definition value</typeparam>
	template<class State, class Def>
	using Add = mp_int<State::value + Def::count * Def::particlesInCell>;

	/// <summary>
	/// Helper meta-function for adding particles of distinct cellTypes
	/// </summary>
	/// <typeparam name="State">Actual state value</typeparam>
	/// <typeparam name="Def">Consecutive definition value</typeparam>
	template<class State, class Def>
	using AddDistinct = mp_int<State::value + Def::particlesInCell>;

	/// <summary>
	/// Helper meta-function for adding blood Cells
	/// </summary>
	/// <typeparam name="State">Actual state value</typeparam>
	/// <typeparam name="Def">Consecutive definition value</typeparam>
	template<class State, class Def>
	using AddCells = mp_int<State::value + Def::count>;

	/// <summary>
	/// Helper meta-function for calculating the graph size
	/// </summary>
	/// <typeparam name="State">Actual state value</typeparam>
	/// <typeparam name="Def">Consecutive definition value</typeparam>
	template<class State, class Def>
	using AddSquared = mp_int<State::value + Def::particlesInCell * Def::particlesInCell>;

	/// <summary>
	/// Helper meta-functor for filtering the user-provided list
	/// </summary>
	/// <typeparam name="Def1">First blood cell definition</typeparam>
	/// <typeparam name="Def2">Second blood cell definition</typeparam>
	template<class Def1, class Def2>
	struct IsDuplicate
	{
		static constexpr bool value = (Def1::particlesInCell == Def2::particlesInCell) && std::is_same_v<typename Def1::List, typename Def2::List>;
	};

	/// <summary>
	/// Helper meta-function for folding the blood cell list for every distinct type
	/// </summary>
	/// <typeparam name="State">Actual state value</typeparam>
	/// <typeparam name="Def">Consecutive definition value</typeparam>
	template<class State, class Def>
	using AddDistinctTypes = BloodCellDef
		<
		IsDuplicate<State, Def>::value ? State::count + Def::count : State::count,
		State::particlesInCell,
		State::indicesInCell,
		State::color,
		typename State::List,
		typename State::Vertices,
		typename State::Indices,
		typename State::Normals
		>;

	/// <summary>
	/// In order to merge all definitions of the same type, we first fold the list for every definitions using
	/// a custom meta-function and then we remove all the duplicates
	/// </summary>
	/// <typeparam name="i">index</typeparam>
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
					mp_at_c<UserDefinedBloodCellList, i>::indicesInCell,
					mp_at_c<UserDefinedBloodCellList, i>::color,
					typename mp_at_c<UserDefinedBloodCellList, i>::List,
					typename mp_at_c<UserDefinedBloodCellList, i>::Vertices,
					typename mp_at_c<UserDefinedBloodCellList, i>::Indices,
					typename mp_at_c<UserDefinedBloodCellList, i>::Normals
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

	/// <summary>
	/// Filter out duplicated folded types
	/// </summary>
	using UniqueBloodCellList = mp_unique_if<FoldedBloodCellList<mp_size<UserDefinedBloodCellList>::value - 1>::type, IsDuplicate>;

	inline constexpr bool isPowerOfTwo(int n)
	{
		return n & (n - 1);
	}
	
	/// <summary>
	/// Heuristics: powers of 2 should be at the beginning
	/// </summary>
	/// <param name="particlesInCell1">number of particles in first cell</param>
	/// <param name="particlesInCell2">number of particles in second cell</param>
	/// <returns></returns>
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

	/// <summary>
	/// Helper meta-functor for ordering the blood cell definitions
	/// </summary>
	/// <typeparam name="Def1">First blood cell definition</typeparam>
	/// <typeparam name="Def2">Second blood cell definition</typeparam>
	template<class Def1, class Def2>
	struct BloodCellComparator
	{
		static constexpr bool value = orderBloodCells(Def1::particlesInCell, Def2::particlesInCell);
	};

	/// <summary>
	/// Sort the blood cell definitions
	/// </summary>
	using BloodCellList = mp_sort<UniqueBloodCellList, BloodCellComparator>;

	/// <summary>
	/// Check the final list
	/// </summary>
	static_assert(mp_size<BloodCellList>::value <= maxCudaStreams, "Max number of streams exceeded");

	/// <summary>
	/// Particle count
	/// </summary>
	inline constexpr int particleCount = mp_fold<BloodCellList, mp_int<0>, Add>::value;

	/// <summary>
	/// Particle count of distinct blood cell types
	/// </summary>
	inline constexpr int particleDistinctCellsCount = mp_fold<BloodCellList, mp_int<0>, AddDistinct>::value;

	/// <summary>
	/// Blood cell count
	/// </summary>
	inline constexpr int bloodCellCount = mp_fold<BloodCellList, mp_int<0>, AddCells>::value;

	/// <summary>
	/// Total particle graph size
	/// </summary>
	inline constexpr int totalGraphSize = mp_fold<BloodCellList, mp_int<0>, AddSquared>::value;

	/// <summary>
	/// Distinct blood cell types
	/// </summary>
	inline constexpr int bloodCellTypeCount = mp_size<BloodCellList>::value;

	/// <summary>
	/// Fill the particles start array
	/// </summary>
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

	/// <summary>
	/// Determine where in the device array a particular stream should start its job (calculate accumulated particlesInCell sums)
	/// </summary>
	inline constexpr auto particlesStarts = particlesStartsGenerator();

	/// <summary>
	/// Fill the blood cells start array
	/// </summary>
	inline constexpr auto bloodCellTypesStartsGenerator = []()
		{
			std::array<int, bloodCellTypeCount> arr{};
			using IndexList = mp_iota_c<bloodCellTypeCount>;
			int state = 0;
			mp_for_each<IndexList>([&](auto i) {
				using BloodCellDefinition = mp_at_c<BloodCellList, i>;
				arr[i] = state;
				state += BloodCellDefinition::count;
				});
			return arr;
		};

	/// <summary>
	/// Array of accumulated sums of blood cells in particular type
	/// </summary>
	inline constexpr auto bloodCellTypesStarts = bloodCellTypesStartsGenerator();

	/// <summary>
	/// Fill the array of blood cell models sizes in consecutive types;
	/// </summary>
	inline constexpr auto bloodCellModelStartsGenerator = []()
		{
			std::array<int, bloodCellTypeCount> arr{};
			using IndexList = mp_iota_c<bloodCellTypeCount>;
			int state = 0;
			mp_for_each<IndexList>([&](auto i) {
				using BloodCellDefinition = mp_at_c<BloodCellList, i>;
				arr[i] = state;
				state += BloodCellDefinition::particlesInCell;
				});
			return arr;
		};

	/// <summary>
	/// Array of accumulated sums of blood cell models sizes in consecutive types;
	/// </summary>
	inline constexpr auto bloodCellModelStarts = bloodCellModelStartsGenerator();

	/// <summary>
	/// Fill the accumulated graph sizes array
	/// </summary>
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

	/// <summary>
	/// Graph sizes for each type
	/// </summary>
	inline constexpr auto accumulatedGraphSizes = graphSizesGenerator();

	/// <summary>
	/// for initial spring length calculations
	/// </summary>
	inline constexpr float springLengthCoefficient = 1.0f;

	/// <summary>
	/// Fill the particle neighborhood graph
	/// </summary>
	inline constexpr auto springGraphGenerator = []()
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

	/// <summary>
	/// Particle neighbour graph
	/// </summary>
	inline constexpr auto springGraph = springGraphGenerator();

}