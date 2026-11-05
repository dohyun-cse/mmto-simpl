#include "ad.hpp"

namespace mfem
{

void ADFunction::Gradient(const Vector &x, Vector &J) const
{
   MFEM_ASSERT(x.Size() == width, "ADFunction::Gradient: size mismatch");
   J.SetSize(width);
   ADVector x_ad(x);
   for (int i = 0; i < width; i++)
   {
      x_ad[i].gradient = 1.0;
      J[i] = (*this)(x_ad).gradient;
      x_ad[i].gradient = 0.0;
   }
}

void ADFunction::Hessian(const Vector &x, DenseMatrix &H) const
{
   MFEM_ASSERT(x.Size() == width, "ADFunction::Hessian: size mismatch");
   H.SetSize(width);
   AD2Vector x_ad(x);
   for (int i = 0; i < width; i++)
   {
      x_ad[i].value.gradient = 1.0;
      for (int j = 0; j < width; j++)
      {
         x_ad[j].gradient.value = 1.0;
         H(i, j) = (*this)(x_ad).gradient.gradient;
         x_ad[j].gradient.value = 0.0;
      }
      x_ad[i].value.gradient = 0.0;
   }
}

real_t ADEnergyIntegrator::GetElementEnergy(const FiniteElement &el,
                                            ElementTransformation &Tr,
                                            const Vector &elfun)
{
   const int dof = el.GetDof();
   const int dim = Tr.GetSpaceDim();
   const int vdim = elfun.Size() / dof;

   x.SetSize(f.width);
   dshape.SetSize(dof, dim);
   elfun_mat.UseExternalData(const_cast<real_t*>(elfun.GetData()), dof, vdim);
   x_mat.UseExternalData(x.GetData(), dim, vdim);

   real_t energy = 0.0;
   const IntegrationRule *ir = GetIntegrationRule(el, Tr);
   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetIntPoint(&ip);
      el.CalcPhysDShape(Tr, dshape);
      MultAtB(dshape, elfun_mat, x_mat);
      energy += f(x, Tr, ip) * Tr.Weight() * ip.weight;
   }
   return energy;
}

void ADEnergyIntegrator::AssembleElementVector(const FiniteElement &el,
                                               ElementTransformation &Tr,
                                               const Vector &elfun,
                                               Vector &elvect)
{
   const int dof = el.GetDof();
   const int dim = Tr.GetSpaceDim();
   const int vdim = elfun.Size() / dof;

   elvect.SetSize(dof*vdim);
   elvect = 0.0;
   x.SetSize(f.width);
   jac.SetSize(f.width);
   dshape.SetSize(dof, dim);
   elfun_mat.UseExternalData(const_cast<real_t*>(elfun.GetData()), dof, vdim);
   elvect_mat.UseExternalData(elvect.GetData(), dof, vdim);
   x_mat.UseExternalData(x.GetData(), dim, vdim);
   jac_mat.UseExternalData(jac.GetData(), dim, vdim);

   const IntegrationRule *ir = GetIntegrationRule(el, Tr);
   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetIntPoint(&ip);
      el.CalcPhysDShape(Tr, dshape);
      MultAtB(dshape, elfun_mat, x_mat);
      f.Gradient(x, Tr, ip, jac);
      jac *= ip.weight * Tr.Weight();
      AddMult(dshape, jac_mat, elvect_mat);
   }
}

void ADEnergyIntegrator::AssembleElementGrad(const FiniteElement &el,
                                             ElementTransformation &Tr,
                                             const Vector &elfun,
                                             DenseMatrix &elmat)
{
   const int dof = el.GetDof();
   const int dim = Tr.GetSpaceDim();
   const int vdim = elfun.Size() / dof;
   MFEM_ASSERT(dim*vdim == f.width, "ADEnergyIntegrator: size mismatch");

   elmat.SetSize(dof*vdim);
   elmat = 0.0;
   x.SetSize(f.width);
   dshape.SetSize(dof, dim);
   Hsub.SetSize(dim);
   temp.SetSize(dof, dim);
   partelmat.SetSize(dof);
   elfun_mat.UseExternalData(const_cast<real_t*>(elfun.GetData()), dof, vdim);
   x_mat.UseExternalData(x.GetData(), dim, vdim);

   const IntegrationRule *ir = GetIntegrationRule(el, Tr);
   for (int q = 0; q < ir->GetNPoints(); q++)
   {
      const IntegrationPoint &ip = ir->IntPoint(q);
      Tr.SetIntPoint(&ip);
      el.CalcPhysDShape(Tr, dshape);
      MultAtB(dshape, elfun_mat, x_mat);
      f.Hessian(x, Tr, ip, H);
      H *= ip.weight * Tr.Weight();

      // Block (c, r) of the element matrix is dshape H_cr dshape^T
      for (int c = 0; c < vdim; c++)
      {
         for (int r = 0; r <= c; r++)
         {
            for (int j1 = 0; j1 < dim; j1++)
            {
               for (int j2 = 0; j2 < dim; j2++)
               {
                  Hsub(j1, j2) = H(c*dim + j1, r*dim + j2);
               }
            }
            Mult(dshape, Hsub, temp);
            MultABt(temp, dshape, partelmat);
            elmat.AddSubMatrix(c*dof, r*dof, partelmat);
            if (c != r)
            {
               partelmat.Transpose();
               elmat.AddSubMatrix(r*dof, c*dof, partelmat);
            }
         }
      }
   }
}

} // namespace mfem
