#pragma once

#include "../objects/blood_cells_def_type.hpp"

#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl_tuple.hpp>
#include <glm/vec3.hpp>

//#define BLOOD_CELL_FULL_GRAPH

namespace preset
{
	using namespace boost::mp11;

	// TODO: add a full-graph preset

	using NoSprings =
		mp_list<>;

	using Dipole =
		mp_list<
		Spring<0, 1, springLength, 1>
		>;

	using Quadrupole =
		mp_list<
		Spring<0, 1, springLength, 1>,
		Spring<1, 2, springLength, 1>,
		Spring<2, 3, springLength, 1>,
		Spring<3, 0, springLength, 1>
		>;

	using Octupole =
		mp_list<
		Spring<0, 1, springLength, 1>,
		Spring<1, 2, springLength, 1>,
		Spring<2, 3, springLength, 1>,
		Spring<3, 0, springLength, 1>,

		Spring<4, 5, springLength, 1>,
		Spring<5, 6, springLength, 1>,
		Spring<6, 7, springLength, 1>,
		Spring<7, 4, springLength, 1>,

		Spring<0, 4, springLength, 1>,
		Spring<1, 5, springLength, 1>,
		Spring<2, 6, springLength, 1>,
		Spring<3, 7, springLength, 1>
		>;

#pragma region Tetrahedron
	using Tetrahedron =
		mp_list <
		Spring<0, 1, springLength, 1>,
		Spring<1, 2, springLength, 1>,
		Spring<2, 0, springLength, 1>,
		Spring<0, 3, springLength, 1>,
		Spring<1, 3, springLength, 1>,
		Spring<2, 3, springLength, 1>
		>;

	using TetrahedronVertices =
		mp_list <
		mpFloat3<0, 0, 0, 1>,
		mpFloat3<10, 0, 0, 1>,
		mpFloat3<50000, 86602, 0, 5>,
		mpFloat3<50000, 28867, 81649, 5>
		>;

	using TetrahedronIndices =
		mp_list<
		mp_int<0>, mp_int<2>, mp_int<1>,
		mp_int<0>, mp_int<1>, mp_int<3>,
		mp_int<2>, mp_int<0>, mp_int<3>,
		mp_int<1>, mp_int<2>, mp_int<3>
		>;

	using TetrahedronNormals =
		mp_list<
		mpFloat3<-7071, -7071, 0, 5>,
		mpFloat3<7071, -7071, 0, 5>,
		mpFloat3<0, 1, 0, 1>,
		mpFloat3<0, 0, 1, 1>
		>;
#pragma endregion

#pragma region Cube
	using Cube =
		mp_list <
		Spring<0, 1, springLength, 1>,
		Spring<1, 2, springLength, 1>,
		Spring<2, 3, springLength, 1>,
		Spring<3, 0, springLength, 1>,
		Spring<0, 2, springLengthDiagonalSmall, 1>,
		Spring<3, 1, springLengthDiagonalSmall, 1>,
		Spring<0, 4, springLength, 1>,
		Spring<1, 5, springLength, 1>,
		Spring<0, 5, springLengthDiagonalSmall, 1>,
		Spring<1, 4, springLengthDiagonalSmall, 1>,
		Spring<1, 6, springLengthDiagonalSmall, 1>,
		Spring<2, 5, springLengthDiagonalSmall, 1>,
		Spring<2, 6, springLength, 1>,
		Spring<3, 7, springLength, 1>,
		Spring<2, 7, springLengthDiagonalSmall, 1>,
		Spring<3, 6, springLengthDiagonalSmall, 1>,
		Spring<3, 4, springLengthDiagonalSmall, 1>,
		Spring<0, 7, springLengthDiagonalSmall, 1>,
		Spring<4, 5, springLength, 1>,
		Spring<5, 6, springLength, 1>,
		Spring<6, 7, springLength, 1>,
		Spring<7, 4, springLength, 1>,
		Spring<4, 6, springLengthDiagonalSmall, 1>,
		Spring<7, 5, springLengthDiagonalSmall, 1>,
		Spring<0, 6, springLengthDiagonal, 1>,
		Spring<1, 7, springLengthDiagonal, 1>,
		Spring<2, 4, springLengthDiagonal, 1>,
		Spring<3, 5, springLengthDiagonal, 1>
		> ;

	using CubeVertices =
		mp_list<
		mpFloat3<0, 0, 0, 1>,
		mpFloat3<10, 0, 0, 1>,
		mpFloat3<10, 10, 0, 1>,
		mpFloat3<0, 10, 0, 1>,
		mpFloat3<0, 0, 10, 1>,
		mpFloat3<10, 0, 10, 1>,
		mpFloat3<10, 10, 10, 1>,
		mpFloat3<0, 10, 10, 1>
		>;

	using CubeIndices =
		mp_list<

		mp_int<0>, mp_int<3>, mp_int<2>,
		mp_int<1>, mp_int<0>, mp_int<2>,

		mp_int<4>, mp_int<3>, mp_int<0>,
		mp_int<7>, mp_int<3>, mp_int<4>,

		mp_int<7>, mp_int<2>, mp_int<3>,
		mp_int<6>, mp_int<2>, mp_int<7>,

		mp_int<6>, mp_int<1>, mp_int<2>,
		mp_int<5>, mp_int<1>, mp_int<6>,

		mp_int<4>, mp_int<0>, mp_int<1>,
		mp_int<5>, mp_int<4>, mp_int<1>,

		mp_int<5>, mp_int<7>, mp_int<4>,
		mp_int<5>, mp_int<6>, mp_int<7>
		>;

	using CubeNormals =
		mp_list<
		mpFloat3<-57735, -57735, -57735, 5>, // 0
		mpFloat3<57735, -57735, -57735, 5>, // 1
		mpFloat3<57735, 57735, -57735, 5>, // 2
		mpFloat3<-57735, 57735, -57735, 5>, // 3
		mpFloat3<-57735, -57735, 57735, 5>, // 4
		mpFloat3<57735, -57735, 57735, 5>, // 5
		mpFloat3<57735, 57735, 57735, 5>, // 6
		mpFloat3<-57735, 57735, 57735, 5> // 7
		>;

#pragma endregion 


	/// <summary>
	/// Blood cell model vertices
	/// </summary>
#pragma region BloodCell20Vertices


	using BloodCellVertices =
		mp_list <
		mpFloat3<184, -43671, 198, 5>,
		mpFloat3<0, 0, 0, 5>,
		mpFloat3<-107862, -520, 2, 5>,
		mpFloat3<-63795, 18894, -85, 5>,
		mpFloat3<-64675, -43527, 197, 5>,
		mpFloat3<-53931, -717, -93410, 5>,
		mpFloat3<-31938, 18777, -55264, 5>,
		mpFloat3<-32246, -43646, -55971, 5>,
		mpFloat3<53931, -261, -93412, 5>,
		mpFloat3<31776, 19046, -55266, 5>,
		mpFloat3<32611, -43372, -55972, 5>,
		mpFloat3<107863, 390, -1, 5>,
		mpFloat3<63633, 19432, -88, 5>,
		mpFloat3<65040, -42980, 195, 5>,
		mpFloat3<53931, 587, 93411, 5>,
		mpFloat3<31776, 19548, 55090, 5>,
		mpFloat3<32611, -42862, 56364, 5>,
		mpFloat3<-53931, 132, 93413, 5>,
		mpFloat3<-31938, 19279, 55091, 5>,
		mpFloat3<-32246, -43135, 56365, 5>
		> ;

	using BloodCellIndices =
		mp_list <
		mp_int<3>, mp_int<1>, mp_int<6>,
		mp_int<6>, mp_int<1>, mp_int<9>,
		mp_int<9>, mp_int<1>, mp_int<12>,
		mp_int<12>, mp_int<1>, mp_int<15>,
		mp_int<15>, mp_int<1>, mp_int<18>,
		mp_int<18>, mp_int<1>, mp_int<3>,
		mp_int<2>, mp_int<7>, mp_int<4>,
		mp_int<7>, mp_int<8>, mp_int<10>,
		mp_int<8>, mp_int<13>, mp_int<10>,
		mp_int<13>, mp_int<14>, mp_int<16>,
		mp_int<14>, mp_int<19>, mp_int<16>,
		mp_int<17>, mp_int<4>, mp_int<19>,
		mp_int<3>, mp_int<5>, mp_int<2>,
		mp_int<5>, mp_int<9>, mp_int<8>,
		mp_int<8>, mp_int<12>, mp_int<11>,
		mp_int<12>, mp_int<14>, mp_int<11>,
		mp_int<15>, mp_int<17>, mp_int<14>,
		mp_int<17>, mp_int<3>, mp_int<2>,
		mp_int<0>, mp_int<4>, mp_int<7>,
		mp_int<0>, mp_int<7>, mp_int<10>,
		mp_int<0>, mp_int<10>, mp_int<13>,
		mp_int<0>, mp_int<13>, mp_int<16>,
		mp_int<0>, mp_int<16>, mp_int<19>,
		mp_int<0>, mp_int<19>, mp_int<4>,
		mp_int<2>, mp_int<5>, mp_int<7>,
		mp_int<7>, mp_int<5>, mp_int<8>,
		mp_int<8>, mp_int<11>, mp_int<13>,
		mp_int<13>, mp_int<11>, mp_int<14>,
		mp_int<14>, mp_int<17>, mp_int<19>,
		mp_int<17>, mp_int<2>, mp_int<4>,
		mp_int<3>, mp_int<6>, mp_int<5>,
		mp_int<5>, mp_int<6>, mp_int<9>,
		mp_int<8>, mp_int<9>, mp_int<12>,
		mp_int<12>, mp_int<15>, mp_int<14>,
		mp_int<15>, mp_int<18>, mp_int<17>,
		mp_int<17>, mp_int<18>, mp_int<3>
		> ;

	using BloodCellNormals =
		mp_list <
		mpFloat3<-17479, 98460, -449, 5>,
		mpFloat3<-419, 100000, -449, 5>,
		mpFloat3<-8950, 98430, -15230, 5>,
		mpFloat3<8120, 98500, -15230, 5>,
		mpFloat3<16650, 98600, -449, 5>,
		mpFloat3<8120, 98629, 14329, 5>,
		mpFloat3<-8950, 98559, 14329, 5>,
		mpFloat3<-98900, 14779, -69, 5>,
		mpFloat3<-23780, -87849, -41429, 5>,
		mpFloat3<-47929, -87760, 400, 5>,
		mpFloat3<49349, 15019, -85670, 5>,
		mpFloat3<24519, -87650, -41429, 5>,
		mpFloat3<48669, -87360, 400, 5>,
		mpFloat3<49349, 15800, 85530, 5>,
		mpFloat3<24519, -87269, 42230, 5>,
		mpFloat3<-23780, -87470, 42230, 5>,
		mpFloat3<-49480, 15379, 85530, 5>,
		mpFloat3<-49480, 14599, -85659, 5>,
		mpFloat3<98769, 15620, -69, 5>,
		mpFloat3<419, -100000, 449, 5>
		> ;
#ifdef BLOOD_CELL_FULL_GRAPH
	using BloodCellSprings =
		mp_list <
		Spring<0, 1, 43673, 5>,
		Spring<0, 2, 116345, 5>,
		Spring<0, 3, 89487, 5>,
		Spring<0, 4, 64859, 5>,
		Spring<0, 5, 116345, 5>,
		Spring<0, 6, 89487, 5>,
		Spring<0, 7, 64859, 5>,
		Spring<0, 8, 116345, 5>,
		Spring<0, 9, 89487, 5>,
		Spring<0, 10, 64859, 5>,
		Spring<0, 11, 116345, 5>,
		Spring<0, 12, 89487, 5>,
		Spring<0, 13, 64859, 5>,
		Spring<0, 14, 116345, 5>,
		Spring<0, 15, 89487, 5>,
		Spring<0, 16, 64859, 5>,
		Spring<0, 17, 116345, 5>,
		Spring<0, 18, 89487, 5>,
		Spring<0, 19, 64859, 5>,
		Spring<1, 2, 107863, 5>,
		Spring<1, 3, 66534, 5>,
		Spring<1, 4, 77959, 5>,
		Spring<1, 5, 107863, 5>,
		Spring<1, 6, 66534, 5>,
		Spring<1, 7, 77959, 5>,
		Spring<1, 8, 107863, 5>,
		Spring<1, 9, 66534, 5>,
		Spring<1, 10, 77959, 5>,
		Spring<1, 11, 107863, 5>,
		Spring<1, 12, 66534, 5>,
		Spring<1, 13, 77959, 5>,
		Spring<1, 14, 107863, 5>,
		Spring<1, 15, 66534, 5>,
		Spring<1, 16, 77959, 5>,
		Spring<1, 17, 107863, 5>,
		Spring<1, 18, 66534, 5>,
		Spring<1, 19, 77959, 5>,
		Spring<2, 3, 48154, 5>,
		Spring<2, 4, 60949, 5>,
		Spring<2, 5, 107863, 5>,
		Spring<2, 6, 95871, 5>,
		Spring<2, 7, 103492, 5>,
		Spring<2, 8, 186825, 5>,
		Spring<2, 9, 151448, 5>,
		Spring<2, 10, 157170, 5>,
		Spring<2, 11, 215727, 5>,
		Spring<2, 12, 172653, 5>,
		Spring<2, 13, 178040, 5>,
		Spring<2, 14, 186825, 5>,
		Spring<2, 15, 151448, 5>,
		Spring<2, 16, 157170, 5>,
		Spring<2, 17, 107863, 5>,
		Spring<2, 18, 95871, 5>,
		Spring<2, 19, 103492, 5>,
		Spring<3, 4, 62429, 5>,
		Spring<3, 5, 95871, 5>,
		Spring<3, 6, 63715, 5>,
		Spring<3, 7, 89609, 5>,
		Spring<3, 8, 151448, 5>,
		Spring<3, 9, 110357, 5>,
		Spring<3, 10, 127651, 5>,
		Spring<3, 11, 172653, 5>,
		Spring<3, 12, 127430, 5>,
		Spring<3, 13, 142923, 5>,
		Spring<3, 14, 151448, 5>,
		Spring<3, 15, 110357, 5>,
		Spring<3, 16, 127651, 5>,
		Spring<3, 17, 95871, 5>,
		Spring<3, 18, 63715, 5>,
		Spring<3, 19, 89609, 5>,
		Spring<4, 5, 103492, 5>,
		Spring<4, 6, 89609, 5>,
		Spring<4, 7, 64858, 5>,
		Spring<4, 8, 157170, 5>,
		Spring<4, 9, 127651, 5>,
		Spring<4, 10, 112338, 5>,
		Spring<4, 11, 178040, 5>,
		Spring<4, 12, 142923, 5>,
		Spring<4, 13, 129716, 5>,
		Spring<4, 14, 157170, 5>,
		Spring<4, 15, 127651, 5>,
		Spring<4, 16, 112338, 5>,
		Spring<4, 17, 103492, 5>,
		Spring<4, 18, 89609, 5>,
		Spring<4, 19, 64858, 5>,
		Spring<5, 6, 48154, 5>,
		Spring<5, 7, 60949, 5>,
		Spring<5, 8, 107863, 5>,
		Spring<5, 9, 95871, 5>,
		Spring<5, 10, 103492, 5>,
		Spring<5, 11, 186825, 5>,
		Spring<5, 12, 151448, 5>,
		Spring<5, 13, 157170, 5>,
		Spring<5, 14, 215727, 5>,
		Spring<5, 15, 172653, 5>,
		Spring<5, 16, 178040, 5>,
		Spring<5, 17, 186825, 5>,
		Spring<5, 18, 151448, 5>,
		Spring<5, 19, 157170, 5>,
		Spring<6, 7, 62429, 5>,
		Spring<6, 8, 95871, 5>,
		Spring<6, 9, 63715, 5>,
		Spring<6, 10, 89609, 5>,
		Spring<6, 11, 151448, 5>,
		Spring<6, 12, 110357, 5>,
		Spring<6, 13, 127651, 5>,
		Spring<6, 14, 172653, 5>,
		Spring<6, 15, 127430, 5>,
		Spring<6, 16, 142924, 5>,
		Spring<6, 17, 151448, 5>,
		Spring<6, 18, 110357, 5>,
		Spring<6, 19, 127651, 5>,
		Spring<7, 8, 103492, 5>,
		Spring<7, 9, 89609, 5>,
		Spring<7, 10, 64858, 5>,
		Spring<7, 11, 157170, 5>,
		Spring<7, 12, 127651, 5>,
		Spring<7, 13, 112338, 5>,
		Spring<7, 14, 178040, 5>,
		Spring<7, 15, 142923, 5>,
		Spring<7, 16, 129716, 5>,
		Spring<7, 17, 157170, 5>,
		Spring<7, 18, 127651, 5>,
		Spring<7, 19, 112338, 5>,
		Spring<8, 9, 48154, 5>,
		Spring<8, 10, 60949, 5>,
		Spring<8, 11, 107863, 5>,
		Spring<8, 12, 95871, 5>,
		Spring<8, 13, 103492, 5>,
		Spring<8, 14, 186825, 5>,
		Spring<8, 15, 151448, 5>,
		Spring<8, 16, 157170, 5>,
		Spring<8, 17, 215727, 5>,
		Spring<8, 18, 172653, 5>,
		Spring<8, 19, 178040, 5>,
		Spring<9, 10, 62429, 5>,
		Spring<9, 11, 95871, 5>,
		Spring<9, 12, 63715, 5>,
		Spring<9, 13, 89609, 5>,
		Spring<9, 14, 151448, 5>,
		Spring<9, 15, 110357, 5>,
		Spring<9, 16, 127651, 5>,
		Spring<9, 17, 172652, 5>,
		Spring<9, 18, 127430, 5>,
		Spring<9, 19, 142923, 5>,
		Spring<10, 11, 103492, 5>,
		Spring<10, 12, 89609, 5>,
		Spring<10, 13, 64858, 5>,
		Spring<10, 14, 157170, 5>,
		Spring<10, 15, 127651, 5>,
		Spring<10, 16, 112338, 5>,
		Spring<10, 17, 178040, 5>,
		Spring<10, 18, 142924, 5>,
		Spring<10, 19, 129716, 5>,
		Spring<11, 12, 48154, 5>,
		Spring<11, 13, 60949, 5>,
		Spring<11, 14, 107863, 5>,
		Spring<11, 15, 95871, 5>,
		Spring<11, 16, 103492, 5>,
		Spring<11, 17, 186825, 5>,
		Spring<11, 18, 151448, 5>,
		Spring<11, 19, 157170, 5>,
		Spring<12, 13, 62429, 5>,
		Spring<12, 14, 95871, 5>,
		Spring<12, 15, 63715, 5>,
		Spring<12, 16, 89609, 5>,
		Spring<12, 17, 151447, 5>,
		Spring<12, 18, 110357, 5>,
		Spring<12, 19, 127650, 5>,
		Spring<13, 14, 103492, 5>,
		Spring<13, 15, 89609, 5>,
		Spring<13, 16, 64858, 5>,
		Spring<13, 17, 157170, 5>,
		Spring<13, 18, 127651, 5>,
		Spring<13, 19, 112338, 5>,
		Spring<14, 15, 48154, 5>,
		Spring<14, 16, 60949, 5>,
		Spring<14, 17, 107863, 5>,
		Spring<14, 18, 95871, 5>,
		Spring<14, 19, 103492, 5>,
		Spring<15, 16, 62429, 5>,
		Spring<15, 17, 95871, 5>,
		Spring<15, 18, 63715, 5>,
		Spring<15, 19, 89609, 5>,
		Spring<16, 17, 103492, 5>,
		Spring<16, 18, 89609, 5>,
		Spring<16, 19, 64858, 5>,
		Spring<17, 18, 48154, 5>,
		Spring<17, 19, 60949, 5>,
		Spring<18, 19, 62429, 5>
		> ;
#else
	using BloodCellSprings =
		mp_list <
		Spring<1, 3, 66534, 5>,
		Spring<1, 9, 66534, 5>,
		Spring<1, 6, 66534, 5>,
		Spring<1, 12, 66534, 5>,
		Spring<1, 15, 66534, 5>,
		Spring<1, 18, 66534, 5>,
		Spring<3, 6, 63715, 5>,
		Spring<6, 9, 63715, 5>,
		Spring<9, 12, 63715, 5>,
		Spring<12, 15, 63715, 5>,
		Spring<15, 18, 63715, 5>,
		Spring<18, 3, 63715, 5>,
		Spring<3, 2, 48154, 5>,
		Spring<6, 5, 48154, 5>,
		Spring<9, 8, 48154, 5>,
		Spring<12, 11, 48154, 5>,
		Spring<15, 14, 48154, 5>,
		Spring<18, 17, 48154, 5>,
		Spring<2, 5, 107863, 5>,
		Spring<5, 8, 107863, 5>,
		Spring<8, 11, 107863, 5>,
		Spring<11, 14, 107863, 5>,
		Spring<14, 17, 107863, 5>,
		Spring<17, 2, 107863, 5>,
		Spring<2, 4, 60949, 5>,
		Spring<5, 7, 60949, 5>,
		Spring<8, 10, 60949, 5>,
		Spring<11, 13, 60949, 5>,
		Spring<14, 16, 60949, 5>,
		Spring<17, 19, 60949, 5>,
		Spring<4, 7, 64858, 5>,
		Spring<7, 10, 64858, 5>,
		Spring<10, 13, 64858, 5>,
		Spring<13, 16, 64858, 5>,
		Spring<16, 19, 64858, 5>,
		Spring<19, 4, 64858, 5>,
		Spring<4, 0, 64859, 5>,
		Spring<7, 0, 64859, 5>,
		Spring<10, 0, 64859, 5>,
		Spring<13, 0, 64859, 5>,
		Spring<16, 0, 64859, 5>,
		Spring<19, 0, 64859, 5>,
		Spring<2, 11, 215727, 5>,
		Spring<5, 14, 215727, 5>,
		Spring<8, 17, 215727, 5>,
		Spring<3, 4, 62429, 5>,
		Spring<6, 7, 62429, 5>,
		Spring<9, 10, 62429, 5>,
		Spring<12, 13, 62429, 5>,
		Spring<15, 16, 62429, 5>,
		Spring<18, 19, 62429, 5>,
		Spring<1, 0, 43673, 5>,
		Spring<1, 4, 77959, 5>,
		Spring<1, 7, 77959, 5>,
		Spring<1, 10, 77959, 5>,
		Spring<1, 13, 77959, 5>,
		Spring<1, 16, 77959, 5>,
		Spring<1, 19, 77959, 5>
		> ;

#endif
#pragma endregion

}