#include "elasticity.hpp"

namespace mfem
{

AnisotropicHalf2DMaterial::AnisotropicHalf2DMaterial(int simp_pow,
                                                     real_t Ex, real_t Ey,
                                                     real_t nu_xy, real_t G_xy)
   : ElasticMaterialBase(2, 2), simp_pow(simp_pow)
{
   MFEM_VERIFY(simp_pow >= 4, "simp_pow must be >= 4 for orthotropic material");
   C.SetSize(3);
   C = 0.0;

   const real_t nu_yx = nu_xy*Ey/Ex;
   const real_t denom = 1.0 / (1.0 - nu_xy*nu_yx);

   C(0, 0) = Ex*denom;
   C(1, 1) = Ey*denom;
   C(0, 1) = nu_xy*Ey*denom;
   C(1, 0) = C(0, 1);
   C(2, 2) = G_xy;
}

HelmholtzFilter::HelmholtzFilter(ParFiniteElementSpace &fes_,
                                 real_t filter_radius)
   : Operator(fes_.GetTrueVSize())
   , eps2_cf(filter_radius*filter_radius/12.0)
   , fes(fes_)
   , blf(&fes_)
   , pcg(fes_.GetComm())
{
   blf.AddDomainIntegrator(new DiffusionIntegrator(eps2_cf));
   blf.AddDomainIntegrator(new MassIntegrator);
   blf.Assemble();
   blf.Finalize();
   A.reset(blf.ParallelAssemble());

   amg.SetPrintLevel(0);
   pcg.SetTol(1e-09);
   pcg.SetAbsTol(1e-12);
   pcg.SetMaxIter(100000);
   pcg.SetPrintLevel(0);
   pcg.SetPreconditioner(amg);
   pcg.SetOperator(*A);
   pcg.iterative_mode = true;
}

void HelmholtzFilter::Mult(const Vector &F, Vector &X) const
{
   const int siz = fes.GetTrueVSize();
   const int nr_comp = F.Size() / siz;
   MFEM_VERIFY(siz*nr_comp == F.Size() && siz*nr_comp == X.Size(),
               "HelmholtzFilter: vector size must be a multiple of the "
               "number of true dofs");

   for (int i = 0; i < nr_comp; i++)
   {
      const Vector Fi(F.GetData() + i*siz, siz);
      Vector Xi(X.GetData() + i*siz, siz);
      pcg.Mult(Fi, Xi);
   }
}

} // namespace mfem
