#pragma once

#include "blood_cell_presets.hpp"
#include "../objects/blood_cells_def_type.hpp"

#include <boost/mp11/list.hpp>


namespace
{
	using namespace boost::mp11;
	// BLOOD CELL PARAMETERS

	inline constexpr float particleRadius = 5;
	inline constexpr float springLengthCoefficient = 1.0f;

	// Please always double check your commas!
	using UserDefinedBloodCellList = mp_list <

		/*BloodCellDef<1, 2,
		preset::Dipole
		>,

		BloodCellDef<1, 4,
		preset::Quadrupole
		>,

		BloodCellDef<3, 2,
		preset::Dipole
		>,

		BloodCellDef<1, 3,
		mp_list<
		Spring<0, 1, 3>,
		Spring<1, 2, 3>,
		Spring<2, 0, 3>
		> >,*/

		BloodCellDef<3, 8, 48,
			preset::Cube,
			preset::CubeVertices,
			preset::CubeIndices,
			preset::CubeNormals
		>,

		BloodCellDef<5, 92, 540,
		preset::BloodCellSprings,
		preset::BloodCellVertices,
		preset::BloodCellIndices,
		preset::BloodCellNormals
		>
	> ;
}