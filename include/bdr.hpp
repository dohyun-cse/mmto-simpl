#pragma once

#include "mfem.hpp"

namespace mfem
{

/// Get the list of essential true dofs from the essential boundary attributes.
/// @param[in] fes Finite element space
/// @param[in] ess_bdr Array marking the essential boundary attributes
/// @param[out] ess_tdof_list Essential true dofs
void GetEssentialTDofList(const FiniteElementSpace &fes,
                          const Array<int> &ess_bdr,
                          Array<int> &ess_tdof_list);

/// Get the list of essential true dofs of a vector finite element space from
/// the essential boundary attributes of each component.
/// @param[in] fes Finite element space
/// @param[in] ess_bdr Array marking the essential boundary attributes; row 0
///                    marks all components, and row i the i-th component
/// @param[out] ess_tdof_list Essential true dofs
void GetEssentialTDofList(const FiniteElementSpace &fes,
                          const Array2D<int> &ess_bdr,
                          Array<int> &ess_tdof_list);

} // namespace mfem
