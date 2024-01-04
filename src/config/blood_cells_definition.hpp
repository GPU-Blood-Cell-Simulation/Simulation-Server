#pragma once

#include "blood_cell_presets.hpp"
#include "../objects/blood_cells_def_type.hpp"

#include <boost/mp11/list.hpp>


namespace
{
	using namespace boost::mp11;

	using UserDefinedBloodCellList = mp_list<
	BloodCellDef<500, 20, 108, 15259093,
		preset::White_blood_cell_One_Springs,
		preset::White_blood_cell_One_Vertices,
		preset::White_blood_cell_One_Indices,
		preset::White_blood_cell_One_Normals>,

	BloodCellDef<1000, 20, 108, 14886695,
		preset::Blood_dust_One_Springs,
		preset::Blood_dust_One_Vertices,
		preset::Blood_dust_One_Indices,
		preset::Blood_dust_One_Normals>
	> ;
}