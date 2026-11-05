#include "bdr.hpp"

namespace mfem
{

void GetEssentialTDofList(const FiniteElementSpace &fes,
                          const Array<int> &ess_bdr,
                          Array<int> &ess_tdof_list)
{
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
}

void GetEssentialTDofList(const FiniteElementSpace &fes,
                          const Array2D<int> &ess_bdr,
                          Array<int> &ess_tdof_list)
{
   MFEM_VERIFY(fes.GetVSize() == ess_bdr.NumRows(),
               "This overload is only for vector finite element spaces.");
   Array<int> ess_bdr_comp(ess_bdr.NumCols());
   Array<int> ess_tdof_list_comp;
   for (int i = 0; i < fes.GetVectorDim(); i++)
   {
      ess_bdr.GetRow(i, ess_bdr_comp);
      fes.GetEssentialTrueDofs(ess_bdr_comp, ess_tdof_list_comp, i);
      ess_tdof_list.Append(ess_tdof_list_comp);
   }
}
} // namespace mfem
