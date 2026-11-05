#pragma once

#include "mfem.hpp"

namespace mfem
{

/// Scalar coefficient f(u(x)) of a vector grid function u.
class ScalarMappedVectorGridFunctionCoefficient : public Coefficient
{
   std::function<real_t(const Vector &)> f;
   GridFunction *gf;
   mutable Vector val;
public:
   ScalarMappedVectorGridFunctionCoefficient()
      : f(nullptr), gf(nullptr) {}
   ScalarMappedVectorGridFunctionCoefficient(
      std::function<real_t(const Vector&)>func)
      : f(func), gf(nullptr)
   {}
   ScalarMappedVectorGridFunctionCoefficient(
      GridFunction &grid_func,
      std::function<real_t(const Vector&)>func)
      : f(func)
      , gf(&grid_func)
   {}
   const GridFunction * GetGridFunction() const { return gf; }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      MFEM_ASSERT(f != nullptr, "No mapping function provided.");
      MFEM_ASSERT(GetGridFunction() != nullptr, "No gridfunction provided.");
      val.SetSize(gf->FESpace()->GetVectorDim());
      gf->GetVectorValue(T, ip, val);
      return f(val);
   }
};

/// Vector coefficient f(u(x)) of a vector grid function u.
class VectorMappedVectorGridFunctionCoefficient : public VectorCoefficient
{
   std::function<void(const Vector &, Vector &)> f;
   GridFunction *gf;
   mutable Vector val;
public:
   VectorMappedVectorGridFunctionCoefficient(int dim)
      : VectorCoefficient(dim), f(nullptr), gf(nullptr) {}
   VectorMappedVectorGridFunctionCoefficient(int dim,
                                             std::function<void(const Vector&, Vector&)>func)
      : VectorCoefficient(dim), f(func), gf(nullptr)
   {}
   VectorMappedVectorGridFunctionCoefficient(
      int dim,
      GridFunction &grid_func,
      std::function<void(const Vector&, Vector&)>func)
      : VectorCoefficient(dim)
      , f(func)
      , gf(&grid_func)
   {}

   const GridFunction * GetGridFunction() const { return gf; }

   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      MFEM_ASSERT(f != nullptr, "No mapping function provided.");
      MFEM_ASSERT(GetGridFunction() != nullptr, "No gridfunction provided.");
      val.SetSize(gf->FESpace()->GetVectorDim());
      gf->GetVectorValue(T, ip, val);
      V.SetSize(vdim);
      f(val, V);
   }
};


} // namespace mfem
