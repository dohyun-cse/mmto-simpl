#pragma once

#include "mfem.hpp"

namespace mfem
{

/// Base class for Legendre functions.
///
/// Here, a Legendre function f is
/// - strictly convex on int(dom(f)),
/// - continuously differentiable on int(dom(f)),
/// - such that |grad f| diverges at the boundary of dom(f).
///
/// Then grad f is injective from int(dom(f)) to R^n, and its inverse is the
/// gradient of the convex conjugate, (grad f)^{-1} = grad f^*, where
/// f^*(y) = sup_x (x . y - f(x)). If f is superlinear, then grad f is a
/// bijection onto R^n and gradinv is defined everywhere.
///
/// If f is defined on R, then f(x) for a vector x is sum_i f(x_i).
class LegendreFunction
{
public:
   LegendreFunction() = default;
   virtual ~LegendreFunction() = default;

   /// Evaluate f(x). The overloads with Tr default to the ones without.
   virtual real_t operator()(real_t x) const
   { MFEM_ABORT("LegendreFunction::operator() not implemented."); }

   virtual real_t operator()(real_t x, ElementTransformation &Tr) const
   { return this->operator()(x); }

   virtual real_t operator()(const Vector &x) const
   {
      real_t sum = 0.0;
      for (auto val : x) { sum += this->operator()(val); }
      return sum;
   }

   virtual real_t operator()(const Vector &x, ElementTransformation &Tr) const
   { return this->operator()(x); }

   virtual real_t conj(const real_t psi) const
   { MFEM_ABORT("LegendreFunction::conj not implemented."); }

   virtual real_t conj(const real_t psi, ElementTransformation &Tr) const
   { return this->conj(psi); }
   virtual real_t conj(const Vector &psi) const
   {
      real_t sum = 0.0;
      for (auto val : psi) { sum += this->conj(val); }
      return sum;
   }
   virtual real_t conj(const Vector &psi, ElementTransformation &Tr) const
   { return this->conj(psi); }

   /// Evaluate grad f(x)
   virtual real_t grad(const real_t x) const
   { MFEM_ABORT("LegendreFunction::grad not implemented."); }

   virtual real_t grad(const real_t x, ElementTransformation &Tr) const
   { return this->grad(x); }

   virtual void grad(const Vector &x, Vector &g) const
   {
      g.SetSize(x.Size());
      for (int i=0; i<x.Size(); i++)
      {
         g[i] = this->grad(x[i]);
      }
   }

   virtual void grad(const Vector &x, ElementTransformation &Tr, Vector &g) const
   { this->grad(x, g); }

   /// Evaluate grad f^*(x) = (grad f)^{-1}(x)
   virtual real_t gradinv(const real_t x) const
   { MFEM_ABORT("LegendreFunction::gradinv not implemented."); }

   virtual real_t gradinv(const real_t x, ElementTransformation &Tr) const
   { return this->gradinv(x); }

   virtual void gradinv(const Vector &x, Vector &invg) const
   {
      invg.SetSize(x.Size());
      for (int i=0; i<x.Size(); i++) { invg[i] = this->gradinv(x[i]); }
   }

   virtual void gradinv(const Vector &x, ElementTransformation &Tr,
                        Vector &invg) const
   { this->gradinv(x, invg); }

   /// Evaluate the inverse Hessian of f at x, diagonal for a vector x
   virtual real_t hessinv(const real_t x) const
   { MFEM_ABORT("LegendreFunction::hessinv not implemented."); }

   virtual real_t hessinv(const real_t x, ElementTransformation &Tr) const
   { return this->hessinv(x); }

   virtual void hessinv(const Vector &x, DenseMatrix &H) const
   {
      H.SetSize(x.Size());
      for (int i=0; i<x.Size(); i++) { H(i,i) = this->hessinv(x[i]); }
   }

   virtual void hessinv(const Vector &x, ElementTransformation &Tr,
                        DenseMatrix &H) const
   { this->hessinv(x, H); }
};

/// Gibbs entropy f(x) = sum_i x_i ln(x_i) on the standard simplex.
///
/// The inverse gradient is grad f^*(psi) = softmax(psi), and the conjugate is
/// f^*(psi) = log(sum_i exp(psi_i)).
class Gibbs : public LegendreFunction
{
   real_t tol;
   mutable Vector y;
public:
   /// Entries x_i <= @a tol are treated as zero in the function and gradient.
   Gibbs(real_t tol=1e-10) : tol(tol) {}
   ~Gibbs() = default;

   using LegendreFunction::operator();
   real_t operator()(const Vector &x) const override;
   real_t conj(const Vector &psi) const override
   {
      const real_t M = psi.Max();
      real_t sum = 0.0;
      for (const auto val : psi) { sum += std::exp(val - M); }
      return M + std::log(sum);
   }

   using LegendreFunction::grad;
   void grad(const Vector &x, Vector &g) const override;
   using LegendreFunction::gradinv;
   void gradinv(const Vector &x, Vector &invg) const override;
   using LegendreFunction::hessinv;
   void hessinv(const Vector &x, DenseMatrix &H) const override;
};

/// Gibbs entropy on the polytope conv(v_1, ..., v_m) with vertices given by
/// the columns of V, R(eta) = min { sum_j lambda_j ln(lambda_j) : V lambda =
/// eta, lambda in the standard simplex }.
///
/// Only the inverse gradient grad R^*(psi) = V softmax(V^T psi) is
/// implemented.
class PolytopeGibbs : public LegendreFunction
{
   Gibbs gibbs;
   DenseMatrix V;
   mutable Vector Vtx, lambda;

public:
   PolytopeGibbs(const DenseMatrix &V_)
      : V(V_), Vtx(V_.Width()), lambda(V_.Width()) {}

   using LegendreFunction::gradinv;
   void gradinv(const Vector &psi, Vector &eta) const override
   {
      V.MultTranspose(psi, Vtx);
      gibbs.gradinv(Vtx, lambda);
      V.Mult(lambda, eta);
   }
};

/// Vector coefficient grad f^*(psi(x)) of a latent variable psi, i.e., the
/// primal variable associated with the Legendre function f.
class PrimalVectorCoefficient : public VectorCoefficient
{
   const LegendreFunction &f;
   VectorCoefficient *latent;
   bool own_coeff;
   mutable Vector val;
public:
   PrimalVectorCoefficient(GridFunction &psi, const LegendreFunction &func)
      : VectorCoefficient(psi.VectorDim()), f(func),
        latent(new VectorGridFunctionCoefficient(&psi)), own_coeff(true) {}
   PrimalVectorCoefficient(VectorCoefficient &psi, const LegendreFunction &func)
      : VectorCoefficient(psi.GetVDim()), f(func),
        latent(&psi), own_coeff(false) {}
   virtual ~PrimalVectorCoefficient() { if (own_coeff && latent) { delete latent; } }
   void Eval(Vector &u, ElementTransformation &Tr,
             const IntegrationPoint &ip) override
   {
      MFEM_ASSERT(latent != nullptr, "Latent VectorCoefficient is null.");
      u.SetSize(GetVDim());
      latent->Eval(val, Tr, ip);
      f.gradinv(val, Tr, u);
   }
};

} // namespace mfem
