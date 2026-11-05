//          SiMPL: Compliance Minimization with Orthotropic Material
//
// Compile with: cmake --build <PATH_TO_BUILD> --target orthotropic
//
// Sample runs:
//    mpirun -np 8 orthotropic
//    mpirun -np 8 orthotropic -r 6 -na 4
//    mpirun -np 8 orthotropic -pv
//
// Description: This example minimizes the compliance of a cantilever beam made
//              of void and an orthotropic material with na orientations,
//
//                 min F(eta) = int f . u dx,  -div(C(eta_f) eps(u)) = f,
//
//              subject to the volume constraint int eta_void dx >= vv |Omega|.
//              The design eta(x) takes values in the polytope whose vertices
//              are the void phase and the material rotated by
//              theta_i = i pi / na, eta_f is its PDE-filtered version, and
//              C(eta_f) is the interpolated stiffness tensor.
//
//              The problem is solved by the SiMPL method: mirror descent on a
//              latent variable psi with eta = V softmax(V^T psi), where the
//              columns of V are the vertices, followed by a Bregman projection
//              onto the volume constraint. The step size is chosen by the
//              Barzilai-Borwein rule with Armijo backtracking.
//
//              The default options reproduce the orthotropic example of the
//              paper; see README.md.

#include "mfem.hpp"
#include "ad.hpp"
#include "elasticity.hpp"
#include "legendre.hpp"
#include "logger.hpp"

using namespace std;
using namespace mfem;

/// Pointwise gap residual min_j (v_j - eta^k) . (psi^k - psi) / alpha, where
/// v_j are the vertices of the polytope (columns of V). Its integral is -res_k.
class GapCoefficient : public Coefficient
{
   GridFunction &prev_control, &control;
   VectorCoefficient &primal;
   const DenseMatrix &V;
   real_t &alpha;
   int vdim;
   mutable Vector psi, psik, g, p, v, s;
   real_t *Vdata;

public:
   GapCoefficient(GridFunction &prev_control_, GridFunction &control_,
                  VectorCoefficient &primal_cf, const DenseMatrix &V_,
                  real_t &alpha_)
      : prev_control(prev_control_), control(control_), primal(primal_cf)
      , V(V_), alpha(alpha_), vdim(primal.GetVDim())
      , psi(vdim), psik(vdim), g(vdim), p(vdim), v(vdim), s(vdim)
      , Vdata(const_cast<real_t*>(V.GetData()))
   {
      MFEM_VERIFY(vdim == V.NumRows(),
                  "GapCoefficient: primal vdim must match V height");
      MFEM_VERIFY(vdim == control.VectorDim(),
                  "GapCoefficient: control vdim must match primal vdim");
   }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      control.GetVectorValue(T, ip, psi);
      prev_control.GetVectorValue(T, ip, psik);
      subtract(psik, psi, g);
      g /= alpha;
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

/// Vector coefficient c(x) + shift.
class ShiftedVectorCoefficient : public VectorCoefficient
{
   VectorCoefficient &c;
   const Vector &shift;

public:
   ShiftedVectorCoefficient(VectorCoefficient &c_, const Vector &shift_)
      : VectorCoefficient(c_.GetVDim()), c(c_), shift(shift_) {}

   void Eval(Vector &val, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      c.Eval(val, T, ip);
      val += shift;
   }
};

/// Bregman projection of the latent variable @a control onto the constraint
/// int W . eta dx >= @a target, i.e., control <- control + mu W with the
/// smallest mu >= 0. The integral is monotone in mu, and the root is found by
/// the Illinois algorithm. Returns mu.
real_t VolumeProjection(ParGridFunction &control,
                        const LegendreFunction &entropy, real_t target,
                        QuadratureFunction &qf, const Vector &W)
{
   const int design_dim = control.VectorDim();
   Vector wmu(design_dim);
   wmu = 0.0;
   VectorGridFunctionCoefficient control_cf(&control);
   ShiftedVectorCoefficient shifted_control_cf(control_cf, wmu);
   PrimalVectorCoefficient eta_cf(shifted_control_cf, entropy);
   Vector intgs(design_dim);
   eta_cf.Project(qf);
   qf.Integrate(intgs);
   if (W*intgs > target) { return 0.0; } // already feasible

   // Find a bracket [mu_l, mu_u] of the root
   real_t mu_u = 5*ParNormlp(control, mfem::infinity(),
                             control.ParFESpace()->GetComm());
   mu_u = std::max(mu_u, 1.0);
   real_t mu_l = -mu_u;
   wmu.Set(mu_l, W);
   eta_cf.Project(qf); qf.Integrate(intgs);
   real_t lower = W*intgs - target;
   wmu.Set(mu_u, W);
   eta_cf.Project(qf); qf.Integrate(intgs);
   real_t upper = W*intgs - target;
   if (lower*upper > 0)
   {
      int num_tries = 0;
      if (lower > 0)
      {
         for (num_tries = 0; num_tries < 100; num_tries++)
         {
            mu_u = mu_l;
            mu_l *= 2.0;
            wmu.Set(mu_l, W);
            eta_cf.Project(qf); qf.Integrate(intgs);
            upper = lower;
            lower = W*intgs - target;
            if (lower < 1e-08) { break; }
         }
      }
      else
      {
         for (num_tries = 0; num_tries < 100; num_tries++)
         {
            mu_l = mu_u;
            mu_u *= 2.0;
            wmu.Set(mu_u, W);
            eta_cf.Project(qf); qf.Integrate(intgs);
            lower = upper;
            upper = W*intgs - target;
            if (upper > -1e-08) { break; }
         }
      }
      MFEM_VERIFY(num_tries < 100, "VolumeProjection: no bracket found");
   }

   // Illinois algorithm
   real_t fa = lower;
   real_t fb = upper;
   int side = 0; // -1: mu_u replaced last, +1: mu_l replaced last
   real_t mu = (mu_l*fb - mu_u*fa) / (fb - fa);
   wmu.Set(mu, W);
   while (std::fabs(mu_u - mu_l) > 1e-08)
   {
      eta_cf.Project(qf);
      intgs = 0.0;
      qf.Integrate(intgs);
      const real_t fc = W*intgs - target;
      if (std::fabs(fc) < 1e-10) { break; }
      if (fc*fb > 0)
      {
         mu_u = mu; fb = fc;
         if (side == -1) { fa *= 0.5; }
         side = -1;
      }
      else
      {
         mu_l = mu; fa = fc;
         if (side == 1) { fb *= 0.5; }
         side = 1;
      }
      mu = (mu_l*fb - mu_u*fa) / (fb - fa);
      wmu.Set(mu, W);
   }

   const int block_size = control.Size() / design_dim;
   for (int j = 0; j < design_dim; j++)
   {
      for (int i = 0; i < block_size; i++)
      {
         control[j*block_size + i] += wmu[j];
      }
   }
   return mu;
}

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;

   // 2. Parse command-line options.
   int order = 1;
   int ref_levels = 8;
   int num_angles = 8;
   real_t psi0_a = 0.5;
   real_t psi0_void = 0.0;
   real_t filter_radius = 0.05;
   real_t void_frac = 0.7;
   real_t rel_tol = 0.0;
   real_t abs_tol = 1e-06;
   int max_it = 10000;
   int max_armijo = 10;
   real_t armijo_c1 = 1e-04;
   bool visualization = false;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Polynomial degree of the displacement.");
   args.AddOption(&ref_levels, "-r", "--ref",
                  "Number of uniform refinements of the 3 x 1 mesh.");
   args.AddOption(&num_angles, "-na", "--num-angles",
                  "Number of orientations theta_i = i pi / na.");
   args.AddOption(&psi0_a, "-psi", "--psi0-a",
                  "Initial latent variable psi^0 = (psi0_void, psi0_a, 0).");
   args.AddOption(&psi0_void, "-psiv", "--psi0-void",
                  "Void component of the initial latent variable.");
   args.AddOption(&filter_radius, "-fr", "--filter-radius",
                  "Filter radius r, eps = r / (2 sqrt(3)).");
   args.AddOption(&void_frac, "-vv", "--void-frac",
                  "Lower bound of the void volume fraction.");
   args.AddOption(&rel_tol, "-rtol", "--rel-tol",
                  "Relative tolerance on the gap residual.");
   args.AddOption(&abs_tol, "-atol", "--abs-tol",
                  "Absolute tolerance on the gap residual.");
   args.AddOption(&max_it, "-mi", "--max-iter",
                  "Maximum number of mirror descent iterations.");
   args.AddOption(&max_armijo, "-ma", "--max-armijo",
                  "Maximum number of Armijo backtracking steps.");
   args.AddOption(&armijo_c1, "-c1", "--armijo-c1",
                  "Armijo sufficient decrease parameter.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization", "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable or disable ParaView output.");
   args.ParseCheck();
   MFEM_VERIFY(num_angles > 0, "num_angles must be positive");
   MFEM_VERIFY(filter_radius > 0, "filter_radius must be positive");

   const int design_dim = 3; // (eta_void, r cos(2 theta), r sin(2 theta))
   real_t alpha = 1e-02;     // alpha_0 = 1e-2 / ||grad F(eta^0)||_inf

   // 3. Create the mesh of Omega = (0, 3) x (0, 1). Boundary attributes are
   //    1 (bottom), 2 (right), 3 (top) and 4 (left).
   Mesh ser_mesh = Mesh::MakeCartesian2D(3, 1, Element::QUADRILATERAL, false,
                                         3.0, 1.0);
   const int dim = ser_mesh.Dimension();
   for (int i = 0; i < ref_levels; i++) { ser_mesh.UniformRefinement(); }
   ParMesh mesh(comm, ser_mesh);
   ser_mesh.Clear();

   // 4. Define the load, a downward force on the disk of radius 0.05 at
   //    (2.9, 0.5).
   Vector center({2.9, 0.5});
   VectorFunctionCoefficient load_cf(dim, [dim, center](const Vector &x,
                                                        Vector &f)
   {
      f = 0.0;
      if (center.DistanceSquaredTo(x) < 0.05*0.05) { f(dim - 1) = -1.0; }
   });

   // 5. Define the finite element spaces: H1 displacement, H1 filtered design
   //    and L2 latent variable.
   H1_FECollection h1_fec(order, dim);
   L2_FECollection l2_fec(order - 1, dim);
   ParFiniteElementSpace fes_state(&mesh, &h1_fec, dim, Ordering::byVDIM);
   ParFiniteElementSpace fes_filter(&mesh, &h1_fec, design_dim);
   ParFiniteElementSpace fes_filter_scalar(&mesh, &h1_fec);
   ParFiniteElementSpace fes_control(&mesh, &l2_fec, design_dim);
   QuadratureSpace qspace(&mesh, order == 0 ? 0 : 2*order + 3);
   QuadratureFunction qf_vec(qspace, design_dim);
   QuadratureFunction qf_scalar(qspace);

   // 6. Define the state, the filtered design, the latent variable and the
   //    gradients. The left edge is clamped.
   ParGridFunction u(&fes_state); u = 0.0;
   HypreParVector u_tvec(&fes_state), load_tvec(&fes_state);
   u.GetTrueDofs(u_tvec);

   Array<int> is_bdr_ess(mesh.bdr_attributes.Max());
   is_bdr_ess = 0;
   is_bdr_ess[3] = 1;

   ParGridFunction feta(&fes_filter); feta = 0.0;
   HypreParVector feta_tvec(&fes_filter);
   feta.GetTrueDofs(feta_tvec);

   ParGridFunction fgrad(&fes_filter); fgrad = 0.0;
   HypreParVector fgrad_tvec(&fes_filter);
   fgrad.GetTrueDofs(fgrad_tvec);
   VectorGridFunctionCoefficient fgrad_cf(&fgrad);

   ParGridFunction control(&fes_control);
   ParGridFunction grad(&fes_control); grad = 0.0;
   Vector control_init(design_dim);
   control_init = 0.0;
   control_init[0] = psi0_void;
   control_init[1] = psi0_a;
   VectorConstantCoefficient control_init_cf(control_init);
   control.ProjectCoefficient(control_init_cf);

   ParGridFunction prev_control(&fes_control); prev_control = 0.0;
   ParGridFunction prev_grad(&fes_control); prev_grad = 0.0;

   // 7. Define the polytope P = conv(e_1, (0, cos(2 theta_i), sin(2 theta_i)))
   //    and the Gibbs entropy on P.
   DenseMatrix vertex_matrix(design_dim, 1 + num_angles);
   vertex_matrix = 0.0;
   vertex_matrix(0, 0) = 1.0;
   for (int i = 0; i < num_angles; i++)
   {
      const real_t theta = M_PI*i / num_angles;
      vertex_matrix(1, 1 + i) = std::cos(2*theta);
      vertex_matrix(2, 1 + i) = std::sin(2*theta);
   }
   if (Mpi::Root())
   {
      out << "Vertex matrix:" << endl;
      vertex_matrix.Print(out, vertex_matrix.NumCols());
   }
   PolytopeGibbs entropy(vertex_matrix);
   PrimalVectorCoefficient eta_cf(control, entropy);
   PrimalVectorCoefficient prev_eta_cf(prev_control, entropy);
   VectorSumCoefficient eta_diff_cf(eta_cf, prev_eta_cf, 1.0, -1.0);
   ParLinearForm eta_diff_lf(&fes_control);
   eta_diff_lf.AddDomainIntegrator(new VectorDomainLFIntegrator(eta_diff_cf));
   GapCoefficient gap_cf(prev_control, control, prev_eta_cf, vertex_matrix,
                         alpha);

   // 8. Define the materials: the void phase with stiffness eta_void^4 C_v and
   //    the orthotropic phase with stiffness r^4 C(theta), plane stress.
   const real_t E_v = 1e-06, nu_v = 0.3;
   const real_t E_x = 5.0, E_y = 0.5, nu_xy = 0.3;
   const real_t nu_yx = nu_xy*E_y/E_x;
   const real_t G_xy = sqrt(E_x*E_y) / (2*(1.0 + sqrt(nu_xy*nu_yx)));
   CompositeMaterial material(dim);
   material.AddMaterial(new IsotropicMaterial(dim, 4, 0.0, E_v, nu_v));
   material.AddMaterial(new AnisotropicHalf2DMaterial(4, E_x, E_y, nu_xy,
                                                      G_xy));

   // 9. Define the elasticity problem. The stiffness matrix is the AD Hessian
   //    of the strain energy, solved by PCG with BoomerAMG.
   ParLinearForm load(&fes_state);
   load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
   load.Assemble();
   load.ParallelAssemble(load_tvec);

   ElasticityEnergy energy(material, feta);
   ParNonlinearForm nlf(&fes_state);
   nlf.SetGradientType(Operator::Type::Hypre_ParCSR);
   nlf.AddDomainIntegrator(new ADEnergyIntegrator(energy));
   nlf.SetEssentialBC(is_bdr_ess, &load_tvec);

   HypreBoomerAMG amg;
   amg.SetElasticityOptions(&fes_state);
   amg.SetPrintLevel(0);
   HyprePCG pcg(comm);
   pcg.SetTol(1e-8);
   pcg.SetAbsTol(1e-12);
   pcg.SetMaxIter(10000);
   pcg.SetPrintLevel(0);
   pcg.SetPreconditioner(amg);
   pcg.iterative_mode = true;

   // 10. Define the filter and the sensitivity of the compliance.
   HelmholtzFilter filter(fes_filter_scalar, filter_radius);
   ParLinearForm filter_targ(&fes_filter);
   HypreParVector filter_targ_tvec(&fes_filter);
   filter_targ.AddDomainIntegrator(new VectorDomainLFIntegrator(eta_cf));

   // dF/d(eta_f) = -1/2 C'(eta_f) eps(u) : eps(u), at the filtered design
   VectorGridFunctionCoefficient feta_cf(&feta);
   DesignStrainEnergy design_energy(material, u);
   ADGradientCoefficient design_grad_cf(design_energy, feta_cf);
   ParLinearForm fgrad_form(&fes_filter);
   HypreParVector fgrad_form_tvec(&fes_filter);
   fgrad_form.AddDomainIntegrator(new VectorDomainLFIntegrator(design_grad_cf));

   // L2 Riesz representation of the filtered gradient
   BilinearForm riesz(&fes_control);
   {
      auto mass = new VectorMassIntegrator();
      mass->SetVDim(design_dim);
      riesz.AddDomainIntegrator(new InverseIntegrator(mass));
   }
   riesz.Assemble();
   riesz.Finalize();
   LinearForm riesz_targ(&fes_control);
   riesz_targ.AddDomainIntegrator(new VectorDomainLFIntegrator(fgrad_cf));

   // Compute the compliance: filter, then solve the state equation.
   auto eval = [&]() -> real_t
   {
      filter_targ.Assemble();
      filter_targ.ParallelAssemble(filter_targ_tvec);
      filter.Mult(filter_targ_tvec, feta_tvec);
      feta.SetFromTrueDofs(feta_tvec);

      pcg.SetOperator(static_cast<HypreParMatrix&>(nlf.GetGradient(u_tvec)));
      pcg.Mult(load_tvec, u_tvec);
      u.SetFromTrueDofs(u_tvec);
      return nlf.GetEnergy(u);
   };

   // Compute the gradient: sensitivity, adjoint filter, then Riesz map.
   auto update_grad = [&]()
   {
      fgrad_form.Assemble();
      fgrad_form.ParallelAssemble(fgrad_form_tvec);
      filter.Mult(fgrad_form_tvec, fgrad_tvec);
      fgrad.SetFromTrueDofs(fgrad_tvec);

      riesz_targ.Assemble();
      riesz.Mult(riesz_targ, grad);
   };

   // 11. Evaluate the initial design and set alpha_0.
   const real_t target = 3.0*void_frac; // lower bound of int eta_void dx
   Vector W_void(design_dim); W_void = 0.0; W_void[0] = 1.0;
   int it = 0, armijo_it = 0;
   real_t res = 0.0, res0 = 0.0;
   VolumeProjection(control, entropy, target, qf_vec, W_void);
   real_t obj = eval();
   update_grad();
   real_t grad_max = grad.Normlinf();
   MPI_Allreduce(MPI_IN_PLACE, &grad_max, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_MAX, comm);
   alpha /= grad_max;

   TableLogger logger;
   logger.Append("it", it);
   logger.Append("obj", obj);
   logger.Append("res", res);
   logger.Append("alpha", alpha);
   logger.Append("retry", armijo_it);
   logger.Print();

   // 12. Set up GLVis and ParaView output of the filtered design and the
   //     displacement.
   vector<unique_ptr<ParGridFunction>> filter_comps;
   for (int i = 0; i < design_dim; i++)
   {
      filter_comps.emplace_back(new ParGridFunction(
                                   &fes_filter_scalar,
                                   feta.GetData() + i*fes_filter_scalar.GetVSize()));
   }
   const char *names[] = {"void", "a", "b"};

   unique_ptr<GLVis> glvis;
   if (visualization)
   {
      glvis.reset(new GLVis("localhost", 19916, 400, 350, design_dim + 1));
      glvis->Append(*filter_comps[0], names[0],
                    "RjcPPPPPPPPPPl\n autoscale off\n valuerange 0 1\n");
      for (int i = 1; i < design_dim; i++)
      {
         glvis->Append(*filter_comps[i], names[i],
                       "Rjc***\n autoscale off\n valuerange -1 1\n");
      }
      glvis->Append(u, "u", "Rjc");
   }

   unique_ptr<ParaViewDataCollection> paraview_dc;
   if (paraview)
   {
      paraview_dc.reset(new ParaViewDataCollection("ParaView/orthotropic",
                                                   &mesh));
      paraview_dc->SetLevelsOfDetail(order < 2 ? order : order + 3);
      if (order > 1) { paraview_dc->SetHighOrderOutput(true); }
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      for (int i = 0; i < design_dim; i++)
      {
         paraview_dc->RegisterField(names[i], filter_comps[i].get());
      }
      paraview_dc->RegisterField("u", &u);
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->Save();
   }

   // 13. Mirror descent loop.
   for (it = 0; it < max_it; it++)
   {
      // Generalized Barzilai-Borwein step size
      if (it > 0)
      {
         eta_diff_lf.Assemble();
         prev_control -= control;
         const real_t numer = eta_diff_lf(prev_control);
         prev_grad -= grad;
         const real_t denom = eta_diff_lf(prev_grad);
         alpha = std::fabs(numer / denom);
      }

      // Mirror descent step with Armijo backtracking
      prev_control = control;
      const real_t prev_obj = obj;
      for (armijo_it = 0; armijo_it < max_armijo; armijo_it++)
      {
         add(prev_control, -alpha, grad, control);
         VolumeProjection(control, entropy, target, qf_vec, W_void);
         obj = eval();
         eta_diff_lf.Assemble();
         const real_t diff = eta_diff_lf(grad); // int grad F . (eta - eta^k)
         if (diff < 0 && obj <= prev_obj + armijo_c1*diff) { break; }
         alpha *= 0.5;
      }
      if (armijo_it == max_armijo)
      {
         if (Mpi::Root()) { out << "Armijo line search failed." << endl; }
         break;
      }

      gap_cf.Project(qf_scalar);
      res = -qf_scalar.Integrate();

      if (glvis) { glvis->Update(); }
      if (paraview_dc)
      {
         paraview_dc->SetCycle(it + 1);
         paraview_dc->SetTime(it + 1);
         paraview_dc->Save();
      }
      logger.Print();

      if (it == 0) { res0 = res; }
      if (res < rel_tol*res0 || std::fabs(res) < abs_tol)
      {
         if (Mpi::Root())
         {
            out << "Converged after " << it << " iterations." << endl;
         }
         break;
      }

      prev_grad = grad;
      update_grad();
   }
   return EXIT_SUCCESS;
}
