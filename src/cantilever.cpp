//              SiMPL: Multi-Material Compliance Minimization
//
// Compile with: cmake --build <PATH_TO_BUILD> --target cantilever
//
// Sample runs:
//    mpirun -np 8 cantilever
//    mpirun -np 8 cantilever -aa 0
//    mpirun -np 8 cantilever -pv -pvs 10
//
// Description: This example minimizes the compliance of a cantilever beam made
//              of void and three isotropic materials,
//
//                 min F(eta) = int f . u dx,  -div(C(eta_f) eps(u)) = f,
//
//              subject to the volume constraints int eta_i dx <= b_i. The
//              material density eta(x) takes values in the probability
//              simplex, eta_f is its PDE-filtered version, and C(eta_f) is a
//              recursive SIMP interpolation of the material stiffnesses.
//
//              The problem is solved by the SiMPL method: mirror descent on a
//              latent variable psi with eta = softmax(psi), followed by a
//              Bregman projection onto the volume constraints computed by the
//              Dykstra algorithm. The step size is chosen by the
//              Barzilai-Borwein rule with Armijo backtracking.
//
//              The default options reproduce the example of the paper; see
//              README.md.

#include "mfem.hpp"
#include "miniapps/autodiff/tadvector.hpp"
#include "bdr.hpp"
#include "cf.hpp"
#include "legendre.hpp"
#include "logger.hpp"
#include "solvers.hpp"

using namespace std;
using namespace mfem;

/// Evaluate x_q = softmax(psi_q) at the quadrature points of the @a ne
/// elements and the integrals intg = sum_q w_q x_q. The layout of @a psi and
/// @a x is byVDIM with @a np points per element and @a dd components.
static void GibbsVolumes(const real_t *psi, real_t *x, const real_t *weights,
                         const int ne, const int np, const int dd,
                         real_t *intg)
{
   real_t y[16];
   MFEM_ASSERT(dd <= 16, "design dimension too large");
   for (int d = 0; d < dd; d++) { intg[d] = 0.0; }
   for (int e = 0; e < ne; e++)
   {
      for (int j = 0; j < np; j++)
      {
         const int k = (e*np + j)*dd;
         real_t maxval = psi[k];
         for (int d = 0; d < dd; d++)
         {
            y[d] = psi[k + d];
            maxval = std::max(maxval, y[d]);
         }
         real_t sum = 0.0;
         for (int d = 0; d < dd; d++)
         {
            x[k + d] = std::exp(y[d] - maxval);
            sum += x[k + d];
         }
         const real_t m = 1.0 / sum;
         for (int d = 0; d < dd; d++) { x[k + d] *= m; }
         for (int d = 0; d < dd; d++) { intg[d] += x[k + d]*weights[e*np + j]; }
      }
   }
}

/// Cache of E_q = exp(psi_q - max(psi_q)) for evaluating softmax(psi_q + s),
/// which is proportional to E_q exp(s), for many constant shifts s. Points
/// where E_q underflows are evaluated directly.
struct GibbsCache
{
   Vector E;
   Array<bool> exact;
   void Setup(const real_t *psi, const int ne, const int np, const int dd)
   {
      const int npts = ne*np;
      E.SetSize(npts*dd);
      exact.SetSize(npts);
      for (int e = 0; e < ne; e++)
      {
         for (int j = 0; j < np; j++)
         {
            const int p = e*np + j, k = p*dd;
            exact[p] = false;
            real_t maxval = psi[k];
            for (int d = 1; d < dd; d++) { maxval = std::max(maxval, psi[k + d]); }
            for (int d = 0; d < dd; d++)
            {
               E[k + d] = std::exp(psi[k + d] - maxval);
               exact[p] = exact[p] || E[k + d] < 1e-280;
            }
         }
      }
   }
};

/// Same as GibbsVolumes for softmax(psi_q + shift), using @a cache.
static void GibbsVolumesCached(const real_t *psi, const GibbsCache &cache,
                               const real_t *shift, real_t *x,
                               const real_t *weights, const int ne,
                               const int np, const int dd, real_t *intg)
{
   real_t c[16], y[16];
   MFEM_ASSERT(dd <= 16, "design dimension too large");
   real_t smax = shift[0];
   for (int d = 1; d < dd; d++) { smax = std::max(smax, shift[d]); }
   for (int d = 0; d < dd; d++) { c[d] = std::exp(shift[d] - smax); }
   const real_t *E = cache.E.GetData();
   for (int d = 0; d < dd; d++) { intg[d] = 0.0; }
   for (int e = 0; e < ne; e++)
   {
      for (int j = 0; j < np; j++)
      {
         const int p = e*np + j, k = p*dd;
         real_t sum = 0.0;
         if (cache.exact[p])
         {
            real_t maxval = psi[k] + shift[0];
            for (int d = 0; d < dd; d++)
            {
               y[d] = psi[k + d] + shift[d];
               maxval = std::max(maxval, y[d]);
            }
            for (int d = 0; d < dd; d++)
            {
               x[k + d] = std::exp(y[d] - maxval);
               sum += x[k + d];
            }
         }
         else
         {
            for (int d = 0; d < dd; d++)
            {
               x[k + d] = E[k + d]*c[d];
               sum += x[k + d];
            }
         }
         const real_t m = 1.0 / sum;
         for (int d = 0; d < dd; d++) { x[k + d] *= m; }
         for (int d = 0; d < dd; d++) { intg[d] += x[k + d]*weights[p]; }
      }
   }
}

/// Bregman projection onto the half-space { eta : int W . eta dx <= b }.
///
/// Find the smallest mu >= 0 with g(mu) = int W . grad R^*(psi - alpha mu W) dx
/// - b <= 0 by the Illinois algorithm. On exit, @a control and @a qf_psi are
/// shifted by -alpha mu W, and @a qf holds the density. Returns mu. A positive
/// finite @a bracket is used as the initial upper bracket.
real_t BregmanProjection(ParGridFunction &control, real_t alpha, real_t b,
                         const Vector &W, QuadratureFunction &qf,
                         QuadratureFunction &qf_psi, real_t bracket,
                         real_t tol = 1e-8,
                         int max_iter = 1e03)
{
   const int design_dim = control.VectorDim();
   MFEM_VERIFY(design_dim == W.Size(),
               "Incompatible constraint and control dimensions.");
   const int block_size = control.Size() / design_dim;

   Vector wmu(design_dim);
   Vector intg(design_dim);
   const int ne = control.ParFESpace()->GetNE();
   ParMesh *mesh = control.ParFESpace()->GetParMesh();
   const int np = qf.GetSpace()->GetIntRule(0).GetNPoints();
   Vector qf_psi_view(design_dim);
   const Vector &qf_weights = qf.GetSpace()->GetWeights();
   GibbsCache cache;
   cache.Setup(qf_psi.GetData(), ne, np, design_dim);

   // g(mu) with the shift wmu = -alpha mu W
   auto g = [&](real_t mu)
   {
      wmu = W; wmu *= -alpha*mu;
      GibbsVolumesCached(qf_psi.GetData(), cache, wmu.GetData(), qf.Write(),
                         qf_weights.GetData(), ne, np, design_dim,
                         intg.GetData());
      MPI_Allreduce(MPI_IN_PLACE, intg.GetData(), design_dim, MPI_DOUBLE, MPI_SUM,
                    mesh->GetComm());
      return W*intg - b;
   };

   if (g(0.0) <= tol) { wmu = 0.0; return 0.0; }

   // Find a bracket [mu_l, mu_u] with g(mu_l) > 0 >= g(mu_u).
   real_t mu_u;
   if (bracket == mfem::infinity() || std::fabs(bracket) < 1e-08)
   {
      mu_u = ParNormlp(control, mfem::infinity(),
                       control.ParFESpace()->GetComm());
      mu_u = std::max(mu_u, 10.0);
   }
   else
   {
      mu_u = std::fabs(bracket);
   }
   real_t mu_l = 0.0;
   real_t f_l = g(mu_l), f_u = g(mu_u);
   real_t a = mu_u;
   for (int s = 0; f_u > 0.0; s++)
   {
      MFEM_VERIFY(s < max_iter, "BregmanProjection: rightward bracket failed");
      mu_l = mu_u; f_l = f_u;
      mu_u += 2.0 * a;
      a *= 2.0;
      f_u = g(mu_u);
   }
   for (int s = 0; f_l < 0.0; s++)
   {
      MFEM_VERIFY(s < max_iter, "BregmanProjection: leftward bracket failed");
      mu_u = mu_l; f_u = f_l;
      mu_l -= 2.0 * a;
      a *= 2.0;
      f_l = g(mu_l);
   }

   // Illinois algorithm (modified regula falsi)
   real_t mu = mu_u;
   int side = 0, it = 0;
   real_t f;
   for (it = 0; it < max_iter; it++)
   {
      mu = (mu_l * f_u - mu_u * f_l) / (f_u - f_l);
      f = g(mu);
      if (std::abs(f) < tol) { break; }
      if (f > 0.0)
      {
         mu_l = mu; f_l = f;
         if (side == +1) { f_u *= 0.5; }
         side = +1;
      }
      else
      {
         mu_u = mu; f_u = f;
         if (side == -1) { f_l *= 0.5; }
         side = -1;
      }
   }
   MFEM_VERIFY(it < max_iter,
               "BregmanProjection: root finding failed with " << f);

   // psi <- psi - alpha mu W
   for (int d = 0; d < design_dim; d++)
   {
      real_t *data = control.Write() + d * block_size;
      for (int i = 0; i < block_size; i++) { data[i] += wmu[d]; }
   }
   for (int i = 0; i < ne; i++)
   {
      for (int j = 0; j < np; j++)
      {
         qf_psi_view.SetData(qf_psi.GetData() + i*np*design_dim + j*design_dim);
         qf_psi_view += wmu;
      }
   }
   return mu;
}

/// Anderson acceleration (type II, Walker-Ni difference form) for a small
/// fixed-point iteration x <- G(x), replicated on all ranks.
class SmallAnderson
{
   int m;
   real_t reg;
   std::vector<Vector> dF, dG;
   Vector f_prev, g_prev;
   bool have_prev = false;
public:
   /// Use the last @a m_ differences and the relative ridge parameter @a reg_.
   SmallAnderson(int m_, real_t reg_ = 1e-12) : m(m_), reg(reg_) { }

   void Restart() { dF.clear(); dG.clear(); have_prev = false; }

   /// Return in @a x_next the accelerated iterate from @a x_k and
   /// @a g_k = G(x_k).
   void Step(const Vector &x_k, const Vector &g_k, Vector &x_next)
   {
      const int n = x_k.Size();
      Vector f_k(n);
      subtract(g_k, x_k, f_k);
      if (!have_prev)
      {
         x_next = g_k;
         f_prev = f_k; g_prev = g_k;
         have_prev = true;
         return;
      }
      Vector df(n), dg(n);
      subtract(f_k, f_prev, df);
      subtract(g_k, g_prev, dg);
      dF.push_back(df); dG.push_back(dg);
      if ((int)dF.size() > m) { dF.erase(dF.begin()); dG.erase(dG.begin()); }
      f_prev = f_k; g_prev = g_k;

      // gamma = argmin |f_k - dF gamma| by the regularized normal equations
      const int mk = (int)dF.size();
      DenseMatrix A(mk, mk);
      Vector rhs(mk), gamma(mk);
      real_t trace = 0.0;
      for (int i = 0; i < mk; i++)
      {
         rhs(i) = dF[i] * f_k;
         for (int j = i; j < mk; j++)
         {
            A(i, j) = dF[i] * dF[j]; A(j, i) = A(i, j);
         }
         trace += A(i, i);
      }
      const real_t ridge = reg * (trace / mk + 1e-300);
      for (int i = 0; i < mk; i++) { A(i, i) += ridge; }
      DenseMatrixInverse Ainv(A);
      Ainv.Mult(rhs, gamma);

      // x_next = g_k - sum_i gamma_i dG_i
      x_next = g_k;
      for (int i = 0; i < mk; i++) { x_next.Add(-gamma(i), dG[i]); }
   }
};

/// Bregman-Dykstra projection onto the intersection of the half-spaces
/// { eta : int W_i . eta dx <= b_i }.
///
/// The iterate is psi = psi_0 - alpha W^T q with multipliers @a q >= 0, which
/// are optionally accelerated by Anderson acceleration (@a aa_window > 0).
/// On exit, @a qf holds the density. Returns the number of iterations, or -1
/// if the method did not converge in @a max_iter iterations.
int DykstraProjection(ParGridFunction &control, real_t alpha,
                      const Vector &b, const DenseMatrix &W,
                      QuadratureFunction &qf, Vector &q, real_t tol = 1e-6,
                      int max_iter = 1e03, int aa_window = 0)
{
   const int n          = W.Height();
   const int design_dim = control.VectorDim();
   const int block_size = control.Size() / design_dim;
   Vector mu(n); Vector w(design_dim);
   MFEM_VERIFY(W.Width() == design_dim,
               "W width must equal control vdim");
   MFEM_VERIFY(b.Size() == n, "b size must equal W height");

   // The previous multipliers serve as initial brackets.
   if (q.Size() != n) { q.SetSize(n); q = 0.0; mu = mfem::infinity(); }
   else { mu = q; q = 0.0; }

   QuadratureFunction qf_psi(qf);
   qf_psi.ProjectGridFunction(control);
   Vector intg(design_dim);
   Vector val(n);
   const int ne = control.ParFESpace()->GetNE();
   ParMesh *mesh = control.ParFESpace()->GetParMesh();
   const int np = qf.GetSpace()->GetIntRule(0).GetNPoints();
   Vector qf_psi_view(design_dim);
   const Vector &qf_weights = qf.GetSpace()->GetWeights();

   // val = W int eta dx - b
   auto eval = [&]()
   {
      GibbsVolumes(qf_psi.GetData(), qf.Write(), qf_weights.GetData(), ne,
                   np, design_dim, intg.GetData());
      MPI_Allreduce(MPI_IN_PLACE, intg.GetData(), design_dim, MPI_DOUBLE, MPI_SUM,
                    mesh->GetComm());
      W.Mult(intg, val);
      val -= b;
   };

   // Residual of the stopping test: the violation if q_i = 0, and |val_i| if
   // q_i > 0.
   auto residual = [&]()
   {
      real_t r = 0.0;
      for (int i = 0; i < n; i++)
      {
         r = std::max(r, q[i] > 0.0 ? std::abs(val[i]) : std::max(val[i], 0.0));
      }
      return r;
   };

   // psi <- psi - alpha W^T dq, on the control and its quadrature copy
   Vector wdq(design_dim);
   auto shift = [&](const Vector &dq)
   {
      W.MultTranspose(dq, wdq);
      wdq *= -alpha;
      for (int d = 0; d < design_dim; d++)
      {
         real_t *data = control.Write() + d * block_size;
         for (int j = 0; j < block_size; j++) { data[j] += wdq[d]; }
      }
      for (int e = 0; e < ne; e++)
      {
         for (int j = 0; j < np; j++)
         {
            qf_psi_view.SetData(qf_psi.GetData() + e*np*design_dim + j*design_dim);
            qf_psi_view += wdq;
         }
      }
   };

   SmallAnderson aa(aa_window);
   Vector q_prev(n), q_cand(n), dq(n), val_plain(n);
   int it;
   for (it = 0; it < max_iter; it++)
   {
      q_prev = q;
      for (int i = 0; i < n; i++)
      {
         W.GetRow(i, w);
         // Undo the previous correction: psi <- psi + alpha q_i W_i.
         for (int d = 0; d < design_dim; d++)
         {
            const real_t add = alpha*q[i] * w[d];
            real_t *data = control.Write() + d * block_size;
            for (int j = 0; j < block_size; j++) { data[j] += add; }
         }
         for (int e = 0; e < ne; e++)
         {
            for (int j = 0; j < np; j++)
            {
               qf_psi_view.SetData(qf_psi.GetData() + e*np*design_dim + j*design_dim);
               qf_psi_view.Add(q[i]*alpha, w);
            }
         }
         // Project onto the i-th half-space: psi <- psi - alpha mu_i W_i.
         mu[i] = BregmanProjection(control, alpha, b[i], w, qf, qf_psi, mu[i],
                                   tol, max_iter);
         q[i] = mu[i];
      }
      eval();
      real_t res = residual();
      if (aa_window > 0 && res > tol)
      {
         aa.Step(q_prev, q, q_cand);
         for (int i = 0; i < n; i++) { q_cand[i] = std::max(q_cand[i], 0.0); }
         subtract(q_cand, q, dq);
         val_plain = val;
         const Vector q_plain(q);
         shift(dq);
         q = q_cand;
         eval();
         const real_t res_cand = residual();
         if (res_cand < res) { res = res_cand; }
         else
         {
            // Reject the accelerated iterate.
            dq.Neg();
            shift(dq);
            q = q_plain;
            val = val_plain;
            if (!std::isfinite(res_cand) || res_cand > 1e3*res) { aa.Restart(); }
         }
      }
      if (res <= tol) { break; }
   }
   // Make the density in qf match the accepted iterate.
   if (aa_window > 0) { eval(); }
   return it < max_iter ? it : -1;
}

/// Return the components of the vector grid function @a gf (ordered byNODES)
/// as grid functions on @a scalar_fes that share the data of @a gf.
std::vector<std::unique_ptr<ParGridFunction>> ToComponents(
                                              const ParGridFunction &gf,
                                              ParFiniteElementSpace &scalar_fes)
{
   std::vector<std::unique_ptr<ParGridFunction>> comps;
   for (int i = 0; i < gf.VectorDim(); i++)
   {
      comps.emplace_back(
         new ParGridFunction(
            &scalar_fes, gf.GetData() + i*scalar_fes.GetVSize()
         )
      );
   }
   return comps;
}

/// Shift the latent variable so that max_d psi_d = 0 at every point. This does
/// not change softmax(psi).
void Normalize(GridFunction &gf)
{
   VectorGridFunctionCoefficient gf_cf(&gf);
   auto f = [](const Vector &x, Vector &y)
   {
      real_t maxval = x.Max();
      y = x;
      y -= maxval;
   };
   VectorMappedVectorGridFunctionCoefficient normalize_cf(
      gf.FESpace()->GetVectorDim(), gf, f);
   gf.ProjectCoefficient(normalize_cf);
}

/// Pointwise gap function min_j (v_j - eta) . (grad F + W^T lambda), where
/// v_j are the columns of @a V (the vertices of the polytope). Its L1 norm is
/// the residual res_k of the stopping criterion.
class GapCoefficient : public Coefficient
{
   const GridFunction &grad;
   const DenseMatrix &W;
   const Vector &lambda;
   VectorCoefficient &primal;
   const DenseMatrix &V;
   int vdim;
   mutable Vector g, p, v, s;
   real_t *Vdata;
public:
   GapCoefficient(GridFunction &grad_, DenseMatrix &W_, Vector &lambda_,
                  VectorCoefficient &primal_cf, const DenseMatrix &V_mat)
      : grad(grad_), W(W_), lambda(lambda_), primal(primal_cf)
      , V(V_mat), vdim(primal.GetVDim())
      , g(vdim), p(vdim), v(vdim), s(vdim)
      , Vdata(const_cast<real_t*>(V.GetData()))
   {
      MFEM_VERIFY(vdim == V.NumRows(),
                  "GapCoefficient: primal vdim must match V height");
      MFEM_VERIFY(vdim == grad.VectorDim(),
                  "GapCoefficient: gradient vdim must match primal vdim");
   }
   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      grad.GetVectorValue(T, ip, g);
      W.AddMultTranspose(lambda, g);
      primal.Eval(p, T, ip);
      real_t minval = mfem::infinity();
      for (int j = 0; j < V.NumCols(); j++)
      {
         v.SetData(Vdata + j*v.Size());
         subtract(v, p, s);
         minval = std::min(minval, s*g);
      }
      return minval;
   }
};

/// Derivative of the strain energy density with respect to the filtered
/// density, scale * (dlambda (div u)(div v) + 2 dmu eps(u) : eps(v)).
class StrainEnergyCoefficient : public VectorCoefficient
{
   GridFunction &u_gf, &v_gf;
   VectorCoefficient &L_cf, &M_cf;
   mutable Vector L, M;
   mutable DenseMatrix grad_u, grad_v;
   const real_t s;

public:
   StrainEnergyCoefficient(GridFunction &state_gf, GridFunction &adj_state_gf,
                           VectorCoefficient &lambda_cf,
                           VectorCoefficient &mu_cf, real_t scale = 1.0)
      : VectorCoefficient(lambda_cf.GetVDim()), u_gf(state_gf),
        v_gf(adj_state_gf), L_cf(lambda_cf), M_cf(mu_cf),
        L(lambda_cf.GetVDim()), M(mu_cf.GetVDim()), s(scale) {}
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      const int dim = T.GetSpaceDim();
      L_cf.Eval(L, T, ip);
      M_cf.Eval(M, T, ip);

      u_gf.GetVectorGradient(T, grad_u);
      const real_t div_u = grad_u.Trace();
      grad_u.Symmetrize();

      v_gf.GetVectorGradient(T, grad_v);
      const real_t div_v = grad_v.Trace();
      grad_v.Symmetrize();

      V = L;
      V *= div_u * div_v;
      for (int i = 0; i < dim; i++)
      {
         for (int j = 0; j < dim; j++)
         {
            V.Add(2 * grad_u(i, j) * grad_v(i, j), M);
         }
      }
      V *= s;
   }
};

/// Recursive SIMP interpolation of the Young's moduli @a E,
/// E_1 = E[0] x_1, E_j = E_{j-1} (1 - t_j) + E[j-1] t_j, with
/// t_j = (x_j / (x_1 + ... + x_j))^p and x clipped to x >= 0.
real_t SimpInterpolation(const Vector &x, real_t simp_pow, const Vector &E)
{
   real_t cum_rho = std::max(x[0], 0.0);
   real_t E_eff = E[0] * std::max(x[0], 0.0);
   for (int j = 1; j < x.Size(); j++)
   {
      const real_t clipped_x = std::max(x[j], 0.0);
      const real_t t = pow(x[j] / (cum_rho + clipped_x), simp_pow);
      E_eff = E_eff * (1 - t) + E[j] * t;
      cum_rho += clipped_x;
   }
   return E_eff;
}

/// max(x, 0) of a dual number, with the subgradient 1/2 at the kink.
static mfem::future::dual<real_t, real_t>
MaxZero(const mfem::future::dual<real_t, real_t> &x)
{
   if (x.value > 0.0) { return x; }
   if (x.value < 0.0) { return {0.0, 0.0}; }
   return {0.0, 0.5*x.gradient};
}

/// Gradient of SimpInterpolation with respect to @a x, by forward-mode AD.
void SimpInterpolationGrad(const Vector &x, Vector &val, real_t simp_pow,
                           const Vector &E)
{
   using ADReal_t = mfem::future::dual<real_t, real_t>;

   const int n = x.Size();
   val.SetSize(n);
   TAutoDiffVector<ADReal_t> x_ad(x);
   for (int i = 0; i < n; i++)
   {
      x_ad[i].gradient = 1.0;
      ADReal_t cum_rho = MaxZero(x_ad[0]);
      ADReal_t E_eff = E[0] * MaxZero(x_ad[0]);
      for (int j = 1; j < n; j++)
      {
         ADReal_t clipped_xj = MaxZero(x_ad[j]);
         ADReal_t t = pow(x_ad[j] / (cum_rho + clipped_xj), simp_pow);
         E_eff = E_eff * (1 - t) + E[j] * t;
         cum_rho += clipped_xj;
      }
      val[i] = E_eff.gradient;
      x_ad[i].gradient = 0.0;
   }
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;

   // 2. Parse command-line options.
   const int dim = 2;
   int order = 1;
   int ser_ref_levels = 0;
   int par_ref_levels = 6;
   real_t filter_radius = 0.06;
   real_t vol_frac = 0.3;
   real_t abs_tol = 1e-06;
   real_t rel_tol = 1e-04;
   int max_it = 100;
   int max_armijo = 10;
   real_t max_alpha = mfem::infinity();
   real_t armijo_c1 = 1e-04;
   int aa_window = 5;
   bool visualization = false;
   bool paraview = false;
   int pv_step = 10;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ser_ref_levels, "-sr", "--ser-ref",
                  "Number of serial uniform refinements.");
   args.AddOption(&par_ref_levels, "-pr", "--par-ref",
                  "Number of parallel uniform refinements.");
   args.AddOption(&filter_radius, "-fr", "--filter-radius",
                  "Filter radius r, eps = r / (2 sqrt(3)).");
   args.AddOption(&vol_frac, "-vf", "--vol-frac",
                  "Total volume fraction of the materials.");
   args.AddOption(&abs_tol, "-atol", "--abs-tol",
                  "Absolute tolerance for the gap residual.");
   args.AddOption(&rel_tol, "-rtol", "--rel-tol",
                  "Relative tolerance for the gap residual.");
   args.AddOption(&max_it, "-mi", "--max-iter",
                  "Maximum number of optimization iterations.");
   args.AddOption(&max_armijo, "-ma", "--max-armijo",
                  "Maximum number of backtracking steps per iteration.");
   args.AddOption(&max_alpha, "-amax", "--max-alpha",
                  "Upper bound on the step size.");
   args.AddOption(&armijo_c1, "-c1", "--armijo-c1",
                  "Armijo constant c_1.");
   args.AddOption(&aa_window, "-aa", "--anderson-window",
                  "Anderson acceleration window for the Dykstra projection "
                  "(0 disables it).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.AddOption(&pv_step, "-pvs", "--paraview-step",
                  "Save ParaView output every pv_step iterations.");
   args.ParseCheck();

   std::stringstream filename;
   filename << "cantilever"
            << "_ref" << (ser_ref_levels + par_ref_levels)
            << "_o" << order
            << "_rad" << filter_radius;

   // 3. Create the mesh of the domain (0,3) x (0,1).
   auto ser_mesh = std::make_unique<Mesh>(
                      Mesh::MakeCartesian2D(12, 4, Element::QUADRILATERAL, false, 3.0, 1.0)
                   );
   for (int i = 0; i < ser_ref_levels; i++)
   {
      ser_mesh->UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, *ser_mesh);
   ser_mesh->Clear();
   for (int i = 0; i < par_ref_levels; i++)
   {
      mesh.UniformRefinement();
   }
   auto glb_ne = mesh.GetGlobalNE();
   if (myid == 0)
   {
      cout << "Number of elements: " << glb_ne << endl;
   }

   // 4. Mark the boundaries. The beam is clamped on the left. The filter uses
   //    eta_f = e_1 (pure void) on the top and bottom.
   Array<int> state_ess_bdr(mesh.bdr_attributes.Max());
   state_ess_bdr = 0;
   state_ess_bdr[3] = 1; // left
   Array<int> filter_ess_bdr(mesh.bdr_attributes.Max());
   filter_ess_bdr = 1;
   filter_ess_bdr[1] = 0; // right
   filter_ess_bdr[3] = 0; // left

   // 5. Define the materials and the volume constraints
   //    int eta_{i+1} dx <= b_i, i = 1, 2, 3, with b proportional to 1:2:2.
   real_t dom_vol = 0.0;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      dom_vol += mesh.GetElementVolume(i);
   }
   MPI_Allreduce(MPI_IN_PLACE, &dom_vol, 1, MPI_DOUBLE, MPI_SUM, comm);

   DenseMatrix W(
   {
      {0.0, 1.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0},
      {0.0, 0.0, 0.0, 1.0},
   });
   Vector b(3);
   b[0] = vol_frac*dom_vol * (1.0 / 5.0);
   b[1] = vol_frac*dom_vol * (2.0 / 5.0);
   b[2] = vol_frac*dom_vol * (2.0 / 5.0);
   Vector E({1e-06, 1.0, 3.0, 5.0}); // Young's moduli (void, materials 1-3)
   const real_t nu = 0.3;            // Poisson's ratio
   const real_t simp_pow = 3.0;      // SIMP penalty
   DenseMatrix V;                    // vertices of the simplex
   V.Diag(1.0, E.Size());
   int numMaterials = W.NumCols();
   MFEM_VERIFY(E.Size() == W.NumCols(), "Incompatible material properties.");
   MFEM_VERIFY(b.Size() == W.NumRows(), "Incompatible constraint.");

   // 6. Define the finite element spaces: continuous elements for the
   //    displacement and the filtered density, and discontinuous elements of
   //    one order lower for the latent variable psi.
   H1_FECollection state_fec(order, dim);
   H1_FECollection filter_fec(order, dim);
   L2_FECollection control_fec(order-1, dim);

   ParFiniteElementSpace state_fes(&mesh, &state_fec, dim, Ordering::byVDIM);
   ParFiniteElementSpace filter_fes(&mesh, &filter_fec, numMaterials);
   ParFiniteElementSpace control_fes(&mesh, &control_fec, numMaterials);
   ParFiniteElementSpace E_fes(&mesh, &control_fec);

   ParFiniteElementSpace scalar_h1_fes(&mesh, &state_fec);
   ParFiniteElementSpace scalar_l2_fes(&mesh, &control_fec);

   Array<int> state_ess_tdof_list;
   GetEssentialTDofList(state_fes, state_ess_bdr, state_ess_tdof_list);
   Array<int> filter_ess_tdof_list;
   GetEssentialTDofList(scalar_h1_fes, filter_ess_bdr, filter_ess_tdof_list);

   // 7. Define the grid functions. The latent variable psi = 0 corresponds to
   //    the uniform density eta = (1/4, 1/4, 1/4, 1/4).
   ParGridFunction state_gf(&state_fes);
   state_gf = 0.0;
   ParGridFunction filter_gf(&filter_fes);
   filter_gf = 0.0;
   Vector filter_bdr_value(numMaterials);
   filter_bdr_value = 0.0;
   filter_bdr_value[0] = 1.0;
   VectorConstantCoefficient filter_bdr_cf(filter_bdr_value);
   filter_gf.ProjectBdrCoefficient(filter_bdr_cf, filter_ess_bdr);
   auto filter_comps = ToComponents(filter_gf, scalar_h1_fes);
   ParGridFunction control_gf(&control_fes);
   control_gf = 0.0;
   auto control_comps = ToComponents(control_gf, scalar_l2_fes);
   ParGridFunction E_gf(&E_fes);
   ParGridFunction Ephys_gf(&E_fes);
   ParGridFunction gap_gf(&E_fes);

   Gibbs entropy;
   PrimalVectorCoefficient density_cf(control_gf, entropy);
   VectorConstantCoefficient E_cf(E);
   InnerProductCoefficient Erho_cf(E_cf, density_cf);
   QuadratureSpace qspace(&mesh, order);
   QuadratureFunction density_qf(&qspace, V.NumRows());

   // 8. Define the Lame parameters of the filtered density and their
   //    derivatives.
   const real_t lambda_scale = nu / ((1 + nu) * (1 - 2 * nu));
   const real_t mu_scale = 1.0 / (2 * (1 + nu));
   ScalarMappedVectorGridFunctionCoefficient Ephys_cf(
      filter_gf, [E, simp_pow](const Vector &x)
   {
      return SimpInterpolation(x, simp_pow, E);
   });
   ScalarMappedVectorGridFunctionCoefficient lambda_cf(
      filter_gf, [E, simp_pow, lambda_scale](const Vector &x)
   {
      return SimpInterpolation(x, simp_pow, E) * lambda_scale;
   });
   VectorMappedVectorGridFunctionCoefficient der_lambda_cf(
      numMaterials, filter_gf,
      [E, simp_pow, lambda_scale](const Vector &x, Vector &val)
   {
      SimpInterpolationGrad(x, val, simp_pow, E);
      val *= lambda_scale;
   });
   ScalarMappedVectorGridFunctionCoefficient mu_cf(
      filter_gf, [E, simp_pow, mu_scale](const Vector &x)
   {
      return SimpInterpolation(x, simp_pow, E) * mu_scale;
   });
   VectorMappedVectorGridFunctionCoefficient der_mu_cf(
      numMaterials, filter_gf,
      [E, simp_pow, mu_scale](const Vector &x, Vector &val)
   {
      SimpInterpolationGrad(x, val, simp_pow, E);
      val *= mu_scale;
   });

   // 9. Define the load: a downward body force in a disk of radius 0.05 near
   //    the tip.
   const Vector center({2.9, 0.5});
   VectorFunctionCoefficient load_cf(dim, [center](const Vector &x,
                                                   Vector &force)
   {
      force = 0.0;
      force[1] = x.DistanceSquaredTo(center) < 0.05 * 0.05 ? -1.0 : 0.0;
   });

   ParBilinearForm elasticity(&state_fes);
   elasticity.AddDomainIntegrator(new ElasticityIntegrator(lambda_cf, mu_cf));
   ParLinearForm load(&state_fes);
   load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
   load.Assemble();

   // 10. Define the PDE filter, the Riesz map of the L2 gradient, and the
   //     gradient of the compliance.
   ParBilinearForm filter(&scalar_h1_fes);
   ConstantCoefficient K(filter_radius * filter_radius / 12);
   filter.AddDomainIntegrator(new DiffusionIntegrator(K));
   filter.AddDomainIntegrator(new MassIntegrator());
   filter.Assemble();
   ParLinearForm filter_target(&filter_fes);
   filter_target.AddDomainIntegrator(
      new VectorDomainLFIntegrator(density_cf));

   ParBilinearForm riesz(&control_fes);
   {
      auto mass = new VectorMassIntegrator();
      mass->SetVDim(numMaterials);
      riesz.AddDomainIntegrator(new InverseIntegrator(mass));
   }
   riesz.Assemble();
   ParLinearForm dJdrho(&filter_fes);
   StrainEnergyCoefficient strain_energy_cf(state_gf, state_gf, der_lambda_cf,
                                            der_mu_cf, -0.5);
   dJdrho.AddDomainIntegrator(
      new VectorDomainLFIntegrator(strain_energy_cf));
   ParGridFunction fgradJ(&filter_fes), gradJ(&control_fes);
   ParLinearForm fgradJ_form(&control_fes);
   VectorGridFunctionCoefficient fgradJ_cf(&fgradJ);
   fgradJ_form.AddDomainIntegrator(new VectorDomainLFIntegrator(fgradJ_cf));
   fgradJ = 0.0; gradJ = 0.0;

   // 11. Set up the elasticity and filter solvers (PCG with BoomerAMG).
   EllipticLinearSolver elasticity_solver(elasticity, state_ess_tdof_list, true);
   static_cast<HypreBoomerAMG&>
   (elasticity_solver.GetPreconditioner()).SetElasticityOptions(&state_fes);
   elasticity_solver.SetAbsTol(1e-12);
   elasticity_solver.SetRelTol(1e-10);
   elasticity_solver.SetMaxIter(1e09);
   elasticity_solver.SetPrintLevel(0);
   elasticity_solver.iterative_mode = true;
   EllipticLinearSolver filter_solver(filter, filter_ess_tdof_list, false);
   filter_solver.SetRelTol(1e-10);
   filter_solver.SetAbsTol(1e-10);
   filter_solver.SetMaxIter(1e09);
   filter_solver.SetPrintLevel(0);
   filter_solver.iterative_mode = true;

   ParGridFunction prev_control_gf(&control_fes);
   prev_control_gf = 0.0;
   ParGridFunction prev_gradJ(&control_fes);
   prev_gradJ = 0.0;
   PrimalVectorCoefficient prev_density_cf(prev_control_gf, entropy);
   VectorSumCoefficient diff_rho(density_cf, prev_density_cf, 1.0, -1.0);
   ParLinearForm diff_rho_form(&control_fes);
   diff_rho_form.AddDomainIntegrator(new VectorDomainLFIntegrator(diff_rho));

   // 12. Define the objective: project psi onto the volume constraints,
   //     filter the density, solve the elasticity equation, and return the
   //     compliance.
   real_t alpha = 1.0;
   Vector q(W.NumRows()); // multipliers of the volume constraints
   q = 0.0;
   real_t toc_dyk = 0.0, toc_filt = 0.0, toc_elast = 0.0, toc_grad = 0.0;
   int dyk_it = 0, dyk_total_it = 0, dyk_calls = 0;
   real_t dyk_total_time = 0.0;
   const int dyk_max_it = 10000;
   auto eval = [&]() -> real_t
   {
      if (Mpi::Root()) { tic(); }
      int it = DykstraProjection(control_gf, alpha, b, W, density_qf, q,
                                 1e-08, dyk_max_it, aa_window);
      dyk_it = it < 0 ? dyk_max_it : it + 1;
      dyk_total_it += dyk_it;
      dyk_calls++;
      if (it < 0) { MFEM_WARNING("Dykstra projection did not converge."); }
      if (Mpi::Root()) { toc_dyk = toc(); dyk_total_time += toc_dyk; }
      Normalize(control_gf);
      filter_target.Assemble();
      if (Mpi::Root()) { tic(); }
      filter_solver.Mult(filter_target, filter_gf);
      if (Mpi::Root()) { toc_filt = toc(); }

      if (Mpi::Root()) { tic(); }
      elasticity_solver.Mult(load, state_gf);
      if (Mpi::Root()) { toc_elast = toc(); }
      return load(state_gf);
   };
   real_t obj = eval();
   real_t prev_obj = obj;

   // L2 gradient of the compliance with respect to eta
   auto update_grad = [&]()
   {
      if (Mpi::Root()) { tic(); }
      dJdrho.Assemble();
      filter_solver.Mult(dJdrho, fgradJ);
      fgradJ_form.Assemble();
      riesz.Mult(fgradJ_form, gradJ);
      if (Mpi::Root()) { toc_grad = toc(); }
   };
   update_grad();
   alpha = alpha / ParNormlp(gradJ, mfem::infinity(), comm);

   gap_gf = 0.0;
   GapCoefficient gap_cf(gradJ, W, q, density_cf, V);
   ConstantCoefficient zero_cf(0.0);

   // 13. Set up visualization, output and logging.
   std::unique_ptr<GLVis> glvis;
   if (visualization)
   {
      glvis = std::make_unique<GLVis>("localhost", 19916, 400, 350, 4);
      glvis->Append(state_gf, "u", "Rjcl*********");
      Ephys_gf.ProjectCoefficient(Ephys_cf);
      E_gf.ProjectCoefficient(Erho_cf);
      glvis->Append(Ephys_gf, "pE", "Rjcl********");
      glvis->Append(E_gf, "E", "Rjcl********");
      glvis->Append(gap_gf, "gap", "Rjcl********");
   }

   if (myid != 0)
   {
      out.Disable();
   }
   std::unique_ptr<ParaViewDataCollection> paraview_dc;
   if (paraview)
   {
      paraview_dc.reset(new ParaViewDataCollection("ParaView/" + filename.str(),
                                                   &mesh));
      paraview_dc->SetLevelsOfDetail(order < 2 ? order : order + 3);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->RegisterField("u", &state_gf);
      for (int i = 0; i < filter_comps.size(); i++)
      {
         paraview_dc->RegisterField("filter_" + to_string(i), filter_comps[i].get());
      }
      for (int i = 0; i < control_comps.size(); i++)
      {
         paraview_dc->RegisterField("control_" + to_string(i), control_comps[i].get());
      }
      paraview_dc->SetTime(0);
      paraview_dc->SetCycle(0);
      paraview_dc->Save();
   }

   int it = 0;
   int armijo_iter = 0;
   real_t res = 0.0, res_0 = 0.0;
   TableLogger logger;
   logger.Append("it", it);
   logger.Append("obj", obj);
   logger.Append("alpha", alpha);
   logger.Append("res", res);
   logger.Append("retry", armijo_iter);
   logger.Append("dyk_time", toc_dyk);
   logger.Append("filt_time", toc_filt);
   logger.Append("elast_time", toc_elast);
   logger.Append("grad_time", toc_grad);
   logger.Append("dyk_it", dyk_it);
   logger.Print();

   // 14. Optimization loop
   for (it = 1; it < max_it; it++)
   {
      // Generalized Barzilai-Borwein step size
      if (it > 1)
      {
         diff_rho_form.Assemble();
         real_t rhopsi =
            diff_rho_form(control_gf) - diff_rho_form(prev_control_gf);
         real_t rhograd = diff_rho_form(gradJ) - diff_rho_form(prev_gradJ);
         alpha = std::min(abs(rhopsi / rhograd), max_alpha);
      }
      prev_control_gf = control_gf;
      prev_obj = obj;

      // Mirror descent update psi <- psi - alpha grad F with backtracking
      // until the Armijo condition holds
      for (armijo_iter = 0; armijo_iter < max_armijo; armijo_iter++)
      {
         add(prev_control_gf, -alpha, gradJ, control_gf);
         obj = eval();
         diff_rho_form.Assemble();
         real_t target = prev_obj + armijo_c1 * diff_rho_form(gradJ);
         if (target >= obj)
         {
            break;
         }
         alpha *= 0.5;
      }
      if (armijo_iter == max_armijo)
      {
         out << "Line search failed. Stopping optimization." << std::endl;
         break;
      }

      prev_gradJ = gradJ;
      update_grad();

      // Gap residual res = ||min_j (v_j - eta) . (grad F + W^T q)||_L1
      gap_gf.ProjectCoefficient(gap_cf);
      res = gap_gf.ComputeL1Error(zero_cf);
      if (it == 1)
      {
         res_0 = res;
      }

      logger.Print();
      if (glvis)
      {
         Ephys_gf.ProjectCoefficient(Ephys_cf);
         E_gf.ProjectCoefficient(Erho_cf);
         glvis->Update();
      }
      if (paraview_dc && (it % pv_step == 0))
      {
         paraview_dc->SetCycle(it);
         paraview_dc->SetTime(it);
         paraview_dc->Save();
      }

      if (res / res_0 < rel_tol || res < abs_tol)
      {
         out << "Converged." << std::endl;
         break;
      }
   }
   if (paraview_dc)
   {
      paraview_dc->SetCycle(it+1);
      paraview_dc->SetTime(it+1);
      paraview_dc->Save();
   }
   out << "Optimization finished after " << it << " iterations."
       << std::endl;
   out << "Final compliance: " << load(state_gf) << std::endl;
   out << "Dykstra: " << dyk_calls << " calls, " << dyk_total_it
       << " iterations, " << dyk_total_time
       << " s (Anderson window " << aa_window << ")" << std::endl;

   return 0;
}
