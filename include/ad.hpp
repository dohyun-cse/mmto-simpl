#pragma once

#include "mfem.hpp"
#include "miniapps/autodiff/tadvector.hpp"

namespace mfem
{

/// First-order forward-mode dual number and vector.
typedef future::dual<real_t, real_t> ADReal_t;
typedef TAutoDiffVector<ADReal_t> ADVector;

/// Second-order (nested) dual number and vector, used for Hessians.
typedef future::dual<ADReal_t, ADReal_t> AD2Real_t;
typedef TAutoDiffVector<AD2Real_t> AD2Vector;

/// Scalar function f : R^n -> R whose gradient and Hessian are computed by
/// forward-mode automatic differentiation.
///
/// Derived classes implement the function body once with the AD_IMPL macro,
/// which generates the real_t, ADReal_t and AD2Real_t versions. Parameters
/// that depend on the quadrature point, e.g., grid function values, are
/// updated in ProcessParameters().
class ADFunction
{
public:
   const int width; ///< input dimension n

   ADFunction(int n) : width(n) {}
   virtual ~ADFunction() = default;

   /// Update the parameters at the integration point @a ip of @a Tr.
   virtual void ProcessParameters(ElementTransformation &Tr,
                                  const IntegrationPoint &ip) const {}

   virtual real_t operator()(const Vector &x) const = 0;
   virtual ADReal_t operator()(const ADVector &x) const = 0;
   virtual AD2Real_t operator()(const AD2Vector &x) const = 0;

   /// Evaluate f(x) with the parameters at @a ip.
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const
   {
      ProcessParameters(Tr, ip);
      return (*this)(x);
   }

   /// Compute the gradient @a J of f at @a x.
   void Gradient(const Vector &x, Vector &J) const;
   void Gradient(const Vector &x, ElementTransformation &Tr,
                 const IntegrationPoint &ip, Vector &J) const
   {
      ProcessParameters(Tr, ip);
      Gradient(x, J);
   }

   /// Compute the Hessian @a H of f at @a x.
   void Hessian(const Vector &x, DenseMatrix &H) const;
   void Hessian(const Vector &x, ElementTransformation &Tr,
                const IntegrationPoint &ip, DenseMatrix &H) const
   {
      ProcessParameters(Tr, ip);
      Hessian(x, H);
   }
};

/// Implement the three evaluation operators of an ADFunction.
/// @param SCALAR name of the scalar type used in @a body
/// @param var    name of the input vector
/// @param body   function body returning a SCALAR
#define AD_IMPL(SCALAR, var, body)                                   \
   using ADFunction::operator();                                     \
   real_t operator()(const Vector &var) const override               \
   {                                                                 \
      using SCALAR = real_t;                                         \
      body                                                           \
   }                                                                 \
   ADReal_t operator()(const ADVector &var) const override           \
   {                                                                 \
      using SCALAR = ADReal_t;                                       \
      body                                                           \
   }                                                                 \
   AD2Real_t operator()(const AD2Vector &var) const override         \
   {                                                                 \
      using SCALAR = AD2Real_t;                                      \
      body                                                           \
   }

/// Nonlinear form integrator for the energy int f(grad u) dx of a
/// vector-valued H1 field u, where f is an ADFunction of the gradient
/// x[c*dim + j] = du_c/dx_j. The element vector and gradient are the AD
/// gradient and Hessian of f.
class ADEnergyIntegrator : public NonlinearFormIntegrator
{
   ADFunction &f;

   Vector x, jac;
   DenseMatrix H, Hsub;
   DenseMatrix dshape, temp, partelmat;
   DenseMatrix elfun_mat, elvect_mat, x_mat, jac_mat;

public:
   ADEnergyIntegrator(ADFunction &f_) : f(f_) {}

   const IntegrationRule *GetDefaultIntegrationRule(
      const FiniteElement &trial_fe, const FiniteElement &test_fe,
      const ElementTransformation &trans) const override
   {
      const int order = std::max(trial_fe.GetOrder(), test_fe.GetOrder());
      return &IntRules.Get(trans.GetGeometryType(), 2*order + 2);
   }

   real_t GetElementEnergy(const FiniteElement &el, ElementTransformation &Tr,
                           const Vector &elfun) override;

   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override;

   void AssembleElementGrad(const FiniteElement &el, ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override;
};

/// Vector coefficient grad f(x(p)), where x is a vector coefficient.
class ADGradientCoefficient : public VectorCoefficient
{
   ADFunction &f;
   VectorCoefficient &x_cf;
   Vector x;

public:
   ADGradientCoefficient(ADFunction &f_, VectorCoefficient &x_cf_)
      : VectorCoefficient(f_.width), f(f_), x_cf(x_cf_)
   {
      MFEM_VERIFY(x_cf.GetVDim() == f.width,
                  "ADGradientCoefficient: input dimension mismatch");
   }

   void Eval(Vector &J, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      x_cf.Eval(x, T, ip);
      f.Gradient(x, T, ip, J);
   }
};

} // namespace mfem
