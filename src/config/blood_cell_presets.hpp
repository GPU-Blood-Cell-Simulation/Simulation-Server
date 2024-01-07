#pragma once

#include "../meta_factory/blood_cells_def_type.hpp"

#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl_tuple.hpp>
#include <glm/vec3.hpp>

namespace preset
{
	using namespace boost::mp11;

using White_blood_cell_One_Springs = mp_list<
	Spring<8, 16, 7697985, 7>,
	Spring<0, 8, 4757623, 7>,
	Spring<16, 0, 4757623, 7>,
	Spring<8, 9, 4757600, 7>,
	Spring<4, 8, 7697985, 7>,
	Spring<9, 4, 4757623, 7>,
	Spring<12, 8, 7697985, 7>,
	Spring<0, 12, 4757623, 7>,
	Spring<12, 13, 4757600, 7>,
	Spring<1, 12, 7697985, 7>,
	Spring<13, 1, 4757623, 7>,
	Spring<16, 12, 7697985, 7>,
	Spring<16, 17, 4757600, 7>,
	Spring<2, 16, 7697985, 7>,
	Spring<17, 2, 4757623, 7>,
	Spring<1, 9, 7697985, 7>,
	Spring<8, 1, 4757623, 7>,
	Spring<1, 18, 4757623, 7>,
	Spring<5, 1, 7698000, 7>,
	Spring<18, 5, 4757623, 7>,
	Spring<2, 13, 7697985, 7>,
	Spring<12, 2, 4757623, 7>,
	Spring<2, 10, 4757623, 7>,
	Spring<3, 2, 7698000, 7>,
	Spring<10, 3, 4757623, 7>,
	Spring<4, 17, 7697985, 7>,
	Spring<16, 4, 4757623, 7>,
	Spring<4, 14, 4757623, 7>,
	Spring<6, 4, 7698000, 7>,
	Spring<14, 6, 4757623, 7>,
	Spring<5, 4, 7698000, 7>,
	Spring<9, 5, 4757623, 7>,
	Spring<5, 15, 4757623, 7>,
	Spring<14, 5, 7697985, 7>,
	Spring<15, 14, 4757600, 7>,
	Spring<11, 17, 7697985, 7>,
	Spring<6, 11, 4757623, 7>,
	Spring<17, 6, 4757623, 7>,
	Spring<11, 10, 4757600, 7>,
	Spring<2, 11, 7697985, 7>,
	Spring<19, 13, 7697985, 7>,
	Spring<3, 19, 4757623, 7>,
	Spring<13, 3, 4757623, 7>,
	Spring<19, 18, 4757600, 7>,
	Spring<1, 19, 7697985, 7>,
	Spring<7, 18, 7697985, 7>,
	Spring<19, 7, 4757623, 7>,
	Spring<18, 15, 7697985, 7>,
	Spring<7, 14, 7697985, 7>,
	Spring<15, 7, 4757623, 7>,
	Spring<14, 11, 7697985, 7>,
	Spring<7, 10, 7697985, 7>,
	Spring<11, 7, 4757623, 7>,
	Spring<10, 19, 7697985, 7>,
	Spring<8, 10, 12455600, 7>,
	Spring<9, 11, 12455600, 7>,
	Spring<12, 14, 12455600, 7>,
	Spring<13, 15, 12455600, 7>,
	Spring<16, 18, 12455600, 7>,
	Spring<17, 19, 12455600, 7>,
	Spring<0, 2, 7698000, 7>,
	Spring<2, 3, 7698000, 7>,
	Spring<3, 1, 7698000, 7>,
	Spring<1, 0, 7698000, 7>,
	Spring<6, 4, 7698000, 7>,
	Spring<4, 5, 7698000, 7>,
	Spring<5, 7, 7698000, 7>,
	Spring<7, 6, 7698000, 7>,
	Spring<0, 4, 7698000, 7>,
	Spring<1, 5, 7698000, 7>,
	Spring<2, 6, 7698000, 7>,
	Spring<3, 7, 7698000, 7>
>;

using White_blood_cell_One_Vertices = mp_list<
	mpFloat3<3849000, 3849000, -3849000, 7>,
	mpFloat3<3849000, -3849000, -3849000, 7>,
	mpFloat3<3849000, 3849000, 3849000, 7>,
	mpFloat3<3849000, -3849000, 3849000, 7>,
	mpFloat3<-3849000, 3849000, -3849000, 7>,
	mpFloat3<-3849000, -3849000, -3849000, 7>,
	mpFloat3<-3849000, 3849000, 3849000, 7>,
	mpFloat3<-3849000, -3849000, 3849000, 7>,
	mpFloat3<2378800, 0, -6227800, 7>,
	mpFloat3<-2378800, 0, -6227800, 7>,
	mpFloat3<2378800, 0, 6227800, 7>,
	mpFloat3<-2378800, 0, 6227800, 7>,
	mpFloat3<6227800, 2378800, 0, 7>,
	mpFloat3<6227800, -2378800, 0, 7>,
	mpFloat3<-6227800, 2378800, 0, 7>,
	mpFloat3<-6227800, -2378800, 0, 7>,
	mpFloat3<0, 6227800, -2378800, 7>,
	mpFloat3<0, 6227800, 2378800, 7>,
	mpFloat3<0, -6227800, -2378800, 7>,
	mpFloat3<0, -6227800, 2378800, 7>
>;

using White_blood_cell_One_Indices = mp_list<
	mp_int<8>, mp_int<16>, mp_int<0>,
	mp_int<8>, mp_int<9>, mp_int<4>,
	mp_int<12>, mp_int<8>, mp_int<0>,
	mp_int<12>, mp_int<13>, mp_int<1>,
	mp_int<16>, mp_int<12>, mp_int<0>,
	mp_int<16>, mp_int<17>, mp_int<2>,
	mp_int<1>, mp_int<9>, mp_int<8>,
	mp_int<1>, mp_int<18>, mp_int<5>,
	mp_int<2>, mp_int<13>, mp_int<12>,
	mp_int<2>, mp_int<10>, mp_int<3>,
	mp_int<4>, mp_int<17>, mp_int<16>,
	mp_int<4>, mp_int<14>, mp_int<6>,
	mp_int<5>, mp_int<4>, mp_int<9>,
	mp_int<5>, mp_int<15>, mp_int<14>,
	mp_int<11>, mp_int<17>, mp_int<6>,
	mp_int<11>, mp_int<10>, mp_int<2>,
	mp_int<19>, mp_int<13>, mp_int<3>,
	mp_int<19>, mp_int<18>, mp_int<1>,
	mp_int<7>, mp_int<18>, mp_int<19>,
	mp_int<15>, mp_int<5>, mp_int<18>,
	mp_int<7>, mp_int<14>, mp_int<15>,
	mp_int<11>, mp_int<6>, mp_int<14>,
	mp_int<7>, mp_int<10>, mp_int<11>,
	mp_int<19>, mp_int<3>, mp_int<10>,
	mp_int<8>, mp_int<4>, mp_int<16>,
	mp_int<12>, mp_int<1>, mp_int<8>,
	mp_int<16>, mp_int<2>, mp_int<12>,
	mp_int<1>, mp_int<5>, mp_int<9>,
	mp_int<2>, mp_int<3>, mp_int<13>,
	mp_int<4>, mp_int<6>, mp_int<17>,
	mp_int<5>, mp_int<14>, mp_int<4>,
	mp_int<11>, mp_int<2>, mp_int<17>,
	mp_int<19>, mp_int<1>, mp_int<13>,
	mp_int<7>, mp_int<15>, mp_int<18>,
	mp_int<7>, mp_int<11>, mp_int<14>,
	mp_int<7>, mp_int<19>, mp_int<10>
>;

using White_blood_cell_One_Normals = mp_list<
	mpFloat3<5773500, 5773500, -5773500, 7>,
	mpFloat3<4919500, -5859000, -6439800, 7>,
	mpFloat3<6439800, 4919500, 5859000, 7>,
	mpFloat3<6881999, -4253199, 5877699, 7>,
	mpFloat3<-5859000, 6439800, -4919500, 7>,
	mpFloat3<-6303700, -3895699, -6714699, 7>,
	mpFloat3<-5877699, 6881999, 4253199, 7>,
	mpFloat3<-5773500, -5773500, 5773500, 7>,
	mpFloat3<3484599, 2153300, -9122499, 7>,
	mpFloat3<-2628700, -1624400, -9510599, 7>,
	mpFloat3<2047500, -2530600, 9455400, 7>,
	mpFloat3<-3484599, 2153300, 9122499, 7>,
	mpFloat3<9122499, 3484599, -2153300, 7>,
	mpFloat3<8965100, -4232900, 1307899, 7>,
	mpFloat3<-9787999, 1742299, 1076700, 7>,
	mpFloat3<-8506399, -5257499, 0, 7>,
	mpFloat3<2153300, 9122499, -3484599, 7>,
	mpFloat3<-1307899, 8965100, 4232900, 7>,
	mpFloat3<-2530600, -9455400, -2047500, 7>,
	mpFloat3<2153300, -9122499, 3484599, 7>
>;


using Blood_dust_One_Springs = mp_list<
	Spring<1, 3, 4435609, 7>,
	Spring<1, 9, 4435677, 7>,
	Spring<1, 6, 4435629, 7>,
	Spring<1, 12, 4435703, 7>,
	Spring<1, 15, 4435744, 7>,
	Spring<1, 18, 4435645, 7>,
	Spring<3, 6, 4247668, 7>,
	Spring<6, 9, 4247737, 7>,
	Spring<9, 12, 4247739, 7>,
	Spring<12, 15, 4247668, 7>,
	Spring<15, 18, 4247737, 7>,
	Spring<18, 3, 4247652, 7>,
	Spring<3, 2, 3210282, 7>,
	Spring<6, 5, 3210266, 7>,
	Spring<9, 8, 3210300, 7>,
	Spring<12, 11, 3210280, 7>,
	Spring<15, 14, 3210211, 7>,
	Spring<18, 17, 3210327, 7>,
	Spring<2, 5, 7190884, 7>,
	Spring<5, 8, 7190963, 7>,
	Spring<8, 11, 7190918, 7>,
	Spring<11, 14, 7190884, 7>,
	Spring<14, 17, 7190963, 7>,
	Spring<17, 2, 7190918, 7>,
	Spring<2, 4, 4063347, 7>,
	Spring<5, 7, 4063324, 7>,
	Spring<8, 10, 4063283, 7>,
	Spring<11, 13, 4063280, 7>,
	Spring<14, 16, 4063260, 7>,
	Spring<17, 19, 4063287, 7>,
	Spring<4, 7, 4323876, 7>,
	Spring<7, 10, 4323838, 7>,
	Spring<10, 13, 4323997, 7>,
	Spring<13, 16, 4323925, 7>,
	Spring<16, 19, 4323838, 7>,
	Spring<19, 4, 4323948, 7>,
	Spring<4, 0, 4323910, 7>,
	Spring<7, 0, 4324005, 7>,
	Spring<10, 0, 4324038, 7>,
	Spring<13, 0, 4324046, 7>,
	Spring<16, 0, 4323982, 7>,
	Spring<19, 0, 4323980, 7>,
	Spring<2, 11, 14381827, 7>,
	Spring<5, 14, 14381799, 7>,
	Spring<8, 17, 14381908, 7>,
	Spring<3, 4, 4161955, 7>,
	Spring<6, 7, 4162017, 7>,
	Spring<9, 10, 4161938, 7>,
	Spring<12, 13, 4162000, 7>,
	Spring<15, 16, 4161937, 7>,
	Spring<18, 19, 4161918, 7>,
	Spring<1, 0, 2911556, 7>,
	Spring<1, 4, 5197219, 7>,
	Spring<1, 7, 5197257, 7>,
	Spring<1, 10, 5197266, 7>,
	Spring<1, 13, 5197350, 7>,
	Spring<1, 16, 5197338, 7>,
	Spring<1, 19, 5197299, 7>
>;

using Blood_dust_One_Vertices = mp_list<
	mpFloat3<9599, -2282799, 10400, 7>,
	mpFloat3<-2699, 628700, -2899, 7>,
	mpFloat3<-7193499, 593999, -2699, 7>,
	mpFloat3<-4255700, 1888300, -8599, 7>,
	mpFloat3<-4314300, -2273200, 10300, 7>,
	mpFloat3<-3598099, 580900, -6230199, 7>,
	mpFloat3<-2131900, 1880600, -3687199, 7>,
	mpFloat3<-2152400, -2281100, -3734300, 7>,
	mpFloat3<3592799, 611199, -6230400, 7>,
	mpFloat3<2115799, 1898500, -3687300, 7>,
	mpFloat3<2171400, -2262800, -3734399, 7>,
	mpFloat3<7188199, 654700, -3000, 7>,
	mpFloat3<4239600, 1924200, -8699, 7>,
	mpFloat3<4333400, -2236700, 10199, 7>,
	mpFloat3<3592799, 667800, 6224499, 7>,
	mpFloat3<2115799, 1931899, 3669900, 7>,
	mpFloat3<2171400, -2228800, 3754799, 7>,
	mpFloat3<-3598099, 637499, 6224700, 7>,
	mpFloat3<-2131900, 1914000, 3669900, 7>,
	mpFloat3<-2152400, -2247000, 3754900, 7>
>;

using Blood_dust_One_Indices = mp_list<
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
>;

using Blood_dust_One_Normals = mp_list<
	mpFloat3<41999, -9999799, 44999, 7>,
	mpFloat3<-41999, 9999799, -44999, 7>,
	mpFloat3<-9890199, 1477999, -6999, 7>,
	mpFloat3<-1747999, 9845899, -44999, 7>,
	mpFloat3<-4793199, -8776299, 40000, 7>,
	mpFloat3<-4948199, 1460099, -8566399, 7>,
	mpFloat3<-895000, 9842699, -1523000, 7>,
	mpFloat3<-2378100, -8785200, -4143100, 7>,
	mpFloat3<4934900, 1501999, -8566799, 7>,
	mpFloat3<812000, 9849900, -1523000, 7>,
	mpFloat3<2451999, -8764899, -4142999, 7>,
	mpFloat3<9877200, 1562000, -6999, 7>,
	mpFloat3<1665100, 9860299, -44999, 7>,
	mpFloat3<4866800, -8735700, 40000, 7>,
	mpFloat3<4934900, 1580000, 8552799, 7>,
	mpFloat3<812000, 9863399, 1433099, 7>,
	mpFloat3<2451899, -8726699, 4222899, 7>,
	mpFloat3<-4948000, 1537999, 8552899, 7>,
	mpFloat3<-895000, 9856200, 1432999, 7>,
	mpFloat3<-2378000, -8747100, 4223000, 7>
>;

}