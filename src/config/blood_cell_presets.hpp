#pragma once

#include "../objects/blood_cells_def_type.hpp"

#include <boost/mp11/list.hpp>


namespace preset
{
	using namespace boost::mp11;

	inline constexpr int springLength = 10;

	// TODO: add a full-graph preset

	using NoSprings =
		mp_list<>;

	using Dipole =
		mp_list<
		Spring<0, 1, springLength>
		>;

	using Quadrupole =
		mp_list<
		Spring<0, 1, springLength>,
		Spring<1, 2, springLength>,
		Spring<2, 3, springLength>,
		Spring<3, 0, springLength>
		>;

	using Octupole =
		mp_list<
		Spring<0, 1, springLength>,
		Spring<1, 2, springLength>,
		Spring<2, 3, springLength>,
		Spring<3, 0, springLength>,

		Spring<4, 5, springLength>,
		Spring<5, 6, springLength>,
		Spring<6, 7, springLength>,
		Spring<7, 4, springLength>,

		Spring<0, 4, springLength>,
		Spring<1, 5, springLength>,
		Spring<2, 6, springLength>,
		Spring<3, 7, springLength>
		>;
}