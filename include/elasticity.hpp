#pragma once

#include "mfem.hpp"
#include "linalg/kernels.hpp"
#include "ad.hpp"

namespace mfem
{

/// Voigt vector v of the symmetric part of the dim x dim matrix A, with
/// engineering shear strains, e.g., v = (A_11, A_22, A_12 + A_21) in 2D.
template <typename T>
inline void ToVoigt(int dim, const T *A, T *v)
{
   switch (dim)
   {
      case 2:
         v[0] = A[0];
         v[1] = A[3];
         v[2] = A[1] + A[2];
         break;
      case 3:
         v[0] = A[0];
         v[1] = A[4];
         v[2] = A[8];
         v[3] = A[7] + A[5];
         v[4] = A[6] + A[2];
         v[5] = A[3] + A[1];
         break;
      default:
         MFEM_ABORT("ToVoigt: unsupported dimension");
   }
}

/// x^T A y for the h x w column-major matrix A.
template <typename T1, typename T2, typename T3>
inline auto InnerProduct(int h, int w, const T1 *A, const T2 *x, const T3 *y)
{
   using T_Result = decltype(
                       std::declval<T1>()*std::declval<T2>()*std::declval<T3>() +
                       std::declval<T1>()*std::declval<T2>()*std::declval<T3>());
   T_Result result{};
   for (int i = 0; i < h; i++)
   {
      for (int j = 0; j < w; j++)
      {
         result = result + x[i]*A[i + j*h]*y[j];
      }
   }
   return result;
}

/// Abstract elastic material with a design-dependent stiffness tensor C(eta).
///
/// CalcStressStrain() returns C(eta) eps(u) : eps(v) at a point, given eta
/// and the displacement gradients grad u and grad v. It is overloaded for
/// real and dual numbers, so that it can be differentiated with respect to
/// either eta (sensitivity) or u (state equation).
class ElasticMaterialBase
{
protected:
   int dim;        ///< spatial dimension
   int design_dim; ///< number of design variables

public:
   ElasticMaterialBase(int dim_, int design_dim_)
      : dim(dim_), design_dim(design_dim_) {}
   virtual ~ElasticMaterialBase() = default;

   int GetDim() const { return dim; }
   int GetDesignDim() const { return design_dim; }

   virtual real_t CalcStressStrain(const real_t *eta, const real_t *gradu,
                                   const real_t *gradv) const = 0;
   virtual ADReal_t CalcStressStrain(const ADReal_t *eta, const real_t *gradu,
                                     const real_t *gradv) const = 0;
   virtual AD2Real_t CalcStressStrain(const AD2Real_t *eta, const real_t *gradu,
                                      const real_t *gradv) const = 0;
   virtual ADReal_t CalcStressStrain(const real_t *eta, const ADReal_t *gradu,
                                     const ADReal_t *gradv) const = 0;
   virtual AD2Real_t CalcStressStrain(const real_t *eta, const AD2Real_t *gradu,
                                      const AD2Real_t *gradv) const = 0;
};

/// Implement all overloads of ElasticMaterialBase::CalcStressStrain().
/// @param T1   name of the type of eta used in @a body
/// @param T2   name of the type of gradu and gradv used in @a body
/// @param T3   name of the result type used in @a body
/// @param body function body returning a T3
#define MATERIAL_IMPL(T1, T2, T3, body)                                       \
   real_t CalcStressStrain(const real_t *eta, const real_t *gradu,           \
                           const real_t *gradv) const override               \
   { using T1 = real_t; using T2 = real_t; using T3 = real_t; body }         \
   ADReal_t CalcStressStrain(const ADReal_t *eta, const real_t *gradu,       \
                             const real_t *gradv) const override             \
   { using T1 = ADReal_t; using T2 = real_t; using T3 = ADReal_t; body }     \
   AD2Real_t CalcStressStrain(const AD2Real_t *eta, const real_t *gradu,     \
                              const real_t *gradv) const override            \
   { using T1 = AD2Real_t; using T2 = real_t; using T3 = AD2Real_t; body }   \
   ADReal_t CalcStressStrain(const real_t *eta, const ADReal_t *gradu,       \
                             const ADReal_t *gradv) const override           \
   { using T1 = real_t; using T2 = ADReal_t; using T3 = ADReal_t; body }     \
   AD2Real_t CalcStressStrain(const real_t *eta, const AD2Real_t *gradu,     \
                              const AD2Real_t *gradv) const override         \
   { using T1 = real_t; using T2 = AD2Real_t; using T3 = AD2Real_t; body }

/// Isotropic material with Young's modulus
/// E(eta) = E_0 (1 - eta) + eta^p E_1 and Poisson's ratio nu, under the plane
/// stress assumption in 2D. The design variable eta is a scalar.
class IsotropicMaterial : public ElasticMaterialBase
{
   real_t simp_pow;
   real_t E_0, E_1;
   real_t nu;

public:
   IsotropicMaterial(int dim_, real_t simp_pow_, real_t E_0_, real_t E_1_,
                     real_t nu_)
      : ElasticMaterialBase(dim_, 1), simp_pow(simp_pow_)
      , E_0(E_0_), E_1(E_1_), nu(nu_) {}

   MATERIAL_IMPL(T1, T2, T3,
   {
      const T1 E = E_0*(1.0 - eta[0]) + pow(eta[0], simp_pow)*E_1;
      const T1 lambda = (E*nu) / (1.0 - nu*nu);
      const T1 mu = E / (2.0*(1.0 + nu));

      T2 divu{0.0};
      T2 divv{0.0};
      for (int i = 0; i < dim; ++i)
      {
         divu += gradu[i*dim + i];
         divv += gradv[i*dim + i];
      }

      T2 eps_prod{0.0};
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            T2 eps_u_ij = 0.5*(gradu[i*dim + j] + gradu[j*dim + i]);
            T2 eps_v_ij = 0.5*(gradv[i*dim + j] + gradv[j*dim + i]);
            eps_prod += eps_u_ij*eps_v_ij;
         }
      }

      T3 val = lambda*divu*divv + 2.0*mu*eps_prod;
      return val;
   })
};

/// Two-dimensional orthotropic material whose orientation is given by the
/// doubled angle. The design variable is eta = (a, b) = r (cos 2θ, sin 2θ),
/// and the stiffness tensor is r^p C(θ), where C(θ) is the orthotropic tensor
/// rotated by θ.
///
/// With the Voigt strain rotation T(θ) = M_0 + r^{-1} M_1(a, b), where M_1 is
/// linear in (a, b), the energy is
///    r^p eps(v)^T T^T C_0 T eps(u)
///    = r^p M_0^T C_0 M_0 + r^{p-1} (cross terms) + r^{p-2} M_1^T C_0 M_1,
/// so it is evaluated from r^2 = a^2 + b^2 without trigonometric functions or
/// 1/r. It is differentiable at r = 0 for p >= 4.
class AnisotropicHalf2DMaterial : public ElasticMaterialBase
{
   int simp_pow;
   DenseMatrix C;

public:
   /// Orthotropic material with Young's moduli @a Ex and @a Ey, Poisson's
   /// ratio @a nu_xy and shear modulus @a G_xy, under plane stress.
   AnisotropicHalf2DMaterial(int simp_pow, real_t Ex, real_t Ey,
                             real_t nu_xy, real_t G_xy);

   MATERIAL_IMPL(T1, T2, T3,
   {
      const T1 a = eta[0];
      const T1 b = eta[1];
      const T1 r2 = a*a + b*b;

      // r-independent part M_0 of the strain rotation
      real_t M_iso[9];
      M_iso[0] = 0.5; M_iso[3] = 0.5; M_iso[6] = 0.0;
      M_iso[1] = 0.5; M_iso[4] = 0.5; M_iso[7] = 0.0;
      M_iso[2] = 0.0; M_iso[5] = 0.0; M_iso[8] = 0.0;

      // part M_1(a, b) of the strain rotation, linear in (a, b)
      T1 M_aniso[9];
      M_aniso[0] =  0.5*a; M_aniso[3] = -0.5*a; M_aniso[6] =  0.5*b;
      M_aniso[1] = -0.5*a; M_aniso[4] =  0.5*a; M_aniso[7] = -0.5*b;
      M_aniso[2] = -1.0*b; M_aniso[5] =  1.0*b; M_aniso[8] =  1.0*a;

      T2 epsu[3]; T2 epsv[3];
      ToVoigt(dim, gradu, epsu);
      ToVoigt(dim, gradv, epsv);

      // Rotated strains and stresses
      T3 eps_m_iso[3]; T3 eps_m_aniso[3];
      ::mfem::kernels::Mult(3, 3, M_iso, epsu, eps_m_iso);
      ::mfem::kernels::Mult(3, 3, M_aniso, epsu, eps_m_aniso);
      T3 stress_iso[3]; T3 stress_aniso[3];
      ::mfem::kernels::Mult(3, 3, C.GetData(), eps_m_iso, stress_iso);
      ::mfem::kernels::Mult(3, 3, C.GetData(), eps_m_aniso, stress_aniso);

      T3 V_iso   = InnerProduct(3, 3, M_iso, stress_iso, epsv);
      T3 V_aniso = InnerProduct(3, 3, M_aniso, stress_aniso, epsv);
      T3 V_cross = InnerProduct(3, 3, M_iso, stress_aniso, epsv) +
      InnerProduct(3, 3, M_aniso, stress_iso, epsv);

      T1 p_iso   = pow(r2, simp_pow*0.5);
      T1 p_cross = pow(r2, simp_pow*0.5 - 0.5);
      // r^(p-2); use r2 for p = 4 to avoid d(pow(0, 1))/dr = pow(0, 0)
      T1 p_aniso = simp_pow == 4 ? r2 : pow(r2, simp_pow*0.5 - 1.0);

      T3 val = V_iso*p_iso + V_cross*p_cross + V_aniso*p_aniso;
      return val;
   });
};

/// Sum of materials, each acting on consecutive design variables.
class CompositeMaterial : public ElasticMaterialBase
{
   std::vector<std::unique_ptr<ElasticMaterialBase>> materials;

public:
   CompositeMaterial(int dim_) : ElasticMaterialBase(dim_, 0) {}

   /// Add @a material, which is owned by this object.
   void AddMaterial(ElasticMaterialBase *material)
   {
      MFEM_VERIFY(material->GetDim() == GetDim(),
                  "All materials must have the same dimension");
      design_dim += material->GetDesignDim();
      materials.emplace_back(material);
   }

   MATERIAL_IMPL(T1, T2, T3,
   {
      T3 total_energy{};
      int idx = 0;
      for (const auto &mat : materials)
      {
         total_energy += mat->CalcStressStrain(eta + idx, gradu, gradv);
         idx += mat->GetDesignDim();
      }
      return total_energy;
   })
};

/// Strain energy density W(grad u) = 1/2 C(eta) eps(u) : eps(u) as a function
/// of the displacement gradient, for a given design field eta.
class ElasticityEnergy : public ADFunction
{
   ElasticMaterialBase &material;
   const GridFunction &eta_gf;
   mutable Vector eta;

public:
   ElasticityEnergy(ElasticMaterialBase &material_, const GridFunction &eta_)
      : ADFunction(material_.GetDim()*material_.GetDim())
      , material(material_), eta_gf(eta_), eta(material_.GetDesignDim())
   {
      MFEM_VERIFY(eta_.VectorDim() == material.GetDesignDim(),
                  "ElasticityEnergy: design dimension mismatch");
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   { eta_gf.GetVectorValue(Tr, ip, eta); }

   AD_IMPL(T, gradu,
   {
      return 0.5*material.CalcStressStrain(eta.GetData(), gradu.GetData(),
                                           gradu.GetData());
   });
};

/// Negative strain energy density -1/2 C(eta) eps(u) : eps(u) as a function of
/// the design variable, for a given displacement u. For the compliance
/// F = 1/2 int C(eta) eps(u) : eps(u) dx, which is self-adjoint, its gradient
/// with respect to eta is the sensitivity dF/deta.
class DesignStrainEnergy : public ADFunction
{
   ElasticMaterialBase &material;
   const GridFunction &u;
   mutable DenseMatrix gradu;

public:
   DesignStrainEnergy(ElasticMaterialBase &material_, const GridFunction &u_)
      : ADFunction(material_.GetDesignDim())
      , material(material_), u(u_), gradu(material_.GetDim())
   {
      MFEM_VERIFY(u_.VectorDim() == material.GetDim(),
                  "DesignStrainEnergy: displacement dimension mismatch");
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   { u.GetVectorGradient(Tr, gradu); }

   AD_IMPL(T, eta,
   {
      return -0.5*material.CalcStressStrain(eta.GetData(), gradu.GetData(),
                                            gradu.GetData());
   });
};

/// PDE (Helmholtz) filter (eps^2 grad u, grad v) + (u, v) = (f, v) with
/// eps = r / (2 sqrt(3)), applied to each component of a vector field.
class HelmholtzFilter : public Operator
{
   ConstantCoefficient eps2_cf;
   ParFiniteElementSpace &fes;
   ParBilinearForm blf;
   std::unique_ptr<HypreParMatrix> A;
   HyprePCG pcg;
   HypreBoomerAMG amg;

public:
   /// Filter on the scalar space @a fes_ with filter radius @a filter_radius.
   HelmholtzFilter(ParFiniteElementSpace &fes_, real_t filter_radius);

   /// Solve the filter equation for each component of the true dual vector
   /// @a F. The input @a X is used as the initial guess.
   void Mult(const Vector &F, Vector &X) const override;
};

} // namespace mfem
