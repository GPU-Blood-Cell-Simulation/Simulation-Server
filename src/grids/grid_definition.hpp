#include "../grids/no_grid.cuh"
#include "../grids/uniform_grid.cuh"

#include <variant>

using Grid = std::variant<UniformGrid*, NoGrid*>;