#pragma once
#include "mfem.hpp"
#include <linalg/solvers.hpp>

namespace mfem
{

/// PCG solver for a symmetric positive definite bilinear form with essential
/// boundary conditions, preconditioned by BoomerAMG in parallel and by
/// Gauss-Seidel in serial. If reassemble is true, the form is reassembled at
/// each call of Mult(), e.g., when its coefficients change.
class EllipticLinearSolver : public IterativeSolver
{
   BilinearForm &a;
   bool reassemble;
   mutable OperatorHandle A;
   std::unique_ptr<CGSolver> cg;
   std::unique_ptr<Solver> prec;
   std::unique_ptr<GridFunction> x_gf;
   Array<int> &ess_tdof_list;
   mutable Vector X, b_, B;

public:
   EllipticLinearSolver(BilinearForm &a_,
                        Array<int> &ess_tdof_list_,
                        bool reassemble_=false)
      : IterativeSolver()
      , a(a_), reassemble(reassemble_)
      , x_gf(std::make_unique<GridFunction>(a.FESpace(), (real_t*)nullptr))
      , ess_tdof_list(ess_tdof_list_)
   {
      iterative_mode = true;
      prec = std::make_unique<GSSmoother>();
      cg = std::make_unique<CGSolver>();
      cg->SetPreconditioner(*prec);

      a.Update();
      a.Assemble();
      a.FormSystemMatrix(ess_tdof_list, A);
      cg->SetOperator(*A.As<SparseMatrix>());
      cg->iterative_mode = true;
   }
#ifdef MFEM_USE_MPI
   EllipticLinearSolver(ParBilinearForm &a_,
                        Array<int> &ess_tdof_list_,
                        bool reassemble_=false)
      : IterativeSolver(a_.ParFESpace()->GetComm())
      , a(a_), reassemble(reassemble_)
      , x_gf(std::make_unique<ParGridFunction>(a_.ParFESpace(), (real_t*)nullptr))
      , ess_tdof_list(ess_tdof_list_)
   {
      iterative_mode = true;
      MFEM_VERIFY(GetComm() != MPI_COMM_NULL,
                  "EllipticLinearSolver: Invalid MPI communicator.");
      prec = std::make_unique<HypreBoomerAMG>();
      static_cast<HypreBoomerAMG*>(prec.get())->SetPrintLevel(0);
      cg = std::make_unique<CGSolver>(GetComm());
      cg->SetPreconditioner(*prec);
      a.Update();
      a.Assemble(0);
      a.FormSystemMatrix(ess_tdof_list, A);
      cg->SetOperator(*A.As<HypreParMatrix>());
      cg->iterative_mode = true;
   }
   bool IsParallel() const { return GetComm() != MPI_COMM_NULL; }
#else
   bool IsParallel() const { return false; }
#endif

   Solver &GetPreconditioner() { return *prec; }
   const Solver &GetPreconditioner() const { return *prec; }
   CGSolver &GetSolver() { return *cg; }
   const CGSolver &GetSolver() const { return *cg; }

   void Mult(const Vector &b, Vector &x) const override
   {
      MFEM_VERIFY(b.Size() == x.Size(),
                  "Incompatible right-hand side and solution vector sizes.");
      MFEM_VERIFY(x.Size() % x_gf->Size() == 0, "Incompatible solution vector size.");
      if (x.Size() / x_gf->Size() > 1)
      {
         Vector b_sub, x_sub;
         for (int i=0; i<x.Size() / x_gf->Size(); i++)
         {
            b_sub.MakeRef(const_cast<Vector&>(b), i*x_gf->Size(), x_gf->Size());
            x_sub.MakeRef(x, i*x_gf->Size(), x_gf->Size());
            Mult(b_sub, x_sub);
         }
         return;
      }

      x_gf->MakeRef(x, 0, x.Size());
      x_gf->SetTrueVector();
      cg->SetMaxIter(max_iter);
      cg->SetRelTol(rel_tol);
      cg->SetAbsTol(abs_tol);
      cg->SetPrintLevel(print_level);
      cg->iterative_mode = iterative_mode;
      if (reassemble)
      {
         b_ = b;
         delete a.LoseMat();
         a.Update();
         a.Assemble(0);
         a.FormLinearSystem(ess_tdof_list, x, b_, A, x_gf->GetTrueVector(), B, 1);
         if (IsParallel())
         {
#ifdef MFEM_USE_MPI
            cg->SetOperator(*A.As<HypreParMatrix>());
#endif
         }
         else
         {
            cg->SetOperator(*A.As<SparseMatrix>());
         }
      }
      else
      {
         if (IsParallel())
         {
#ifdef MFEM_USE_MPI
            auto a_par = dynamic_cast<ParBilinearForm*>(&a);
            auto pfes = a_par->ParFESpace();
            const Operator &P = *pfes->GetProlongationMatrix();
            const SparseMatrix &R = *pfes->GetRestrictionMatrix();
            X.SetSize(P.Width());
            B.SetSize(X.Size());
            P.MultTranspose(b, B);
            R.Mult(x, x_gf->GetTrueVector());
            a_par->ParallelEliminateTDofsInRHS(ess_tdof_list, x_gf->GetTrueVector(), B);
#endif
         }
         else
         {
            // Variational restriction with P
            const SparseMatrix *P = a.FESpace()->GetConformingProlongation();
            if (!P)
            {
               b_ = b;
               a.EliminateVDofsInRHS(ess_tdof_list, x, b_);
               X.MakeRef(x, 0, x.Size());
               b_ = b;
               B.MakeRef(b_, 0, b.Size());
            }
            else
            {
               const SparseMatrix *R = a.FESpace()->GetConformingRestriction();
               B.SetSize(P->Width());
               P->MultTranspose(b, B);
               X.SetSize(R->Height());
               R->Mult(x, X);
               a.EliminateVDofsInRHS(ess_tdof_list, X, B);
            }
         }
      }
      cg->Mult(B, x_gf->GetTrueVector());
      if (!cg->GetConverged())
      {
         MFEM_WARNING("EllipticLinearSolver: CG did not converge in "
                      << cg->GetNumIterations() << " iterations, "
                      << "residual norm = " << cg->GetFinalNorm());
      }
      x_gf->SetFromTrueVector();
   }
};

} // namespace mfem
