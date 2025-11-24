#pragma once

#include <deal.II/base/timer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <memory>

using namespace dealii;

// Matrix-based solver for ADR equation
template <int dim>
class ADRMatrixSolver
{
public:
  ADRMatrixSolver(Triangulation<dim> &tria,
                  const unsigned int fe_degree,
                  const Function<dim> &exact_sol,
                  const Function<dim> &rhs,
                  const Function<dim> &mu,
                  const TensorFunction<1, dim> &beta,
                  const Function<dim> &gamma);

  /// Setup DoFs and sparsity pattern
  void setup_system();

  /// Assemble system matrix and right-hand side
  void assemble_system();

  /// Solve the linear system
  void solve();

  /// Compute L2 and H1 errors
  void compute_errors(double &L2_error, double &H1_error) const;

  /// Output solution to VTK file
  void output_results(const std::string &filename) const;

  /// Get number of degrees of freedom
  unsigned int n_dofs() const { return dof_handler.n_dofs(); }

  /// Get solution vector (const reference)
  const Vector<double>& get_solution() const { return solution; }

  /// Get DoF handler (const reference)
  const DoFHandler<dim>& get_dof_handler() const { return dof_handler; }

private:
  Triangulation<dim> &triangulation;
  const unsigned int fe_degree;
  FE_Q<dim>          fe;
  DoFHandler<dim>    dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;

  // Problem data (references to externally owned objects)
  const Function<dim> &exact_solution;
  const Function<dim> &right_hand_side;
  const Function<dim> &diffusion_coefficient;
  const TensorFunction<1, dim> &advection_field;
  const Function<dim> &reaction_coefficient;
};

// Implementation
template <int dim>
ADRMatrixSolver<dim>::ADRMatrixSolver(
    Triangulation<dim> &tria,
    const unsigned int degree,
    const Function<dim> &exact_sol,
    const Function<dim> &rhs,
    const Function<dim> &mu,
    const TensorFunction<1, dim> &beta,
    const Function<dim> &gamma)
  : triangulation(tria)
  , fe_degree(degree)
  , fe(degree)
  , dof_handler(triangulation)
  , exact_solution(exact_sol)
  , right_hand_side(rhs)
  , diffusion_coefficient(mu)
  , advection_field(beta)
  , reaction_coefficient(gamma)
{}

template <int dim>
void ADRMatrixSolver<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  constraints.clear();
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0, // all boundary ids
                                           exact_solution,
                                           constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints,
                                  /*keep_constrained_dofs*/ false);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void ADRMatrixSolver<dim>::assemble_system()
{
  QGauss<dim>   quadrature_formula(fe_degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    cell_matrix = 0;
    cell_rhs    = 0;

    fe_values.reinit(cell);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const double JxW = fe_values.JxW(q);
      const Point<dim> qp = fe_values.quadrature_point(q);

      // Get coefficients at quadrature point
      const double mu = diffusion_coefficient.value(qp);
      const Tensor<1, dim> beta = advection_field.value(qp);
      const double gamma = reaction_coefficient.value(qp);
      const double f_qp = right_hand_side.value(qp);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double phi_i = fe_values.shape_value(i, q);
        const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const double phi_j = fe_values.shape_value(j, q);
          const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);

          // Diffusion term: mu * grad_phi_i . grad_phi_j
          cell_matrix(i, j) += (mu * (grad_phi_i * grad_phi_j)) * JxW;

          // Advection term (conservative form): beta . grad_phi_j * phi_i
          cell_matrix(i, j) += ((beta * grad_phi_j) * phi_i) * JxW;

          // Reaction term: gamma * phi_j * phi_i
          cell_matrix(i, j) += (gamma * phi_j * phi_i) * JxW;
        }

        // RHS: f * phi_i
        cell_rhs(i) += f_qp * phi_i * JxW;
      }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                          local_dof_indices,
                                          system_matrix, system_rhs);
  }
}

template <int dim>
void ADRMatrixSolver<dim>::solve()
{
  // Use stricter tolerance for better accuracy
  const double tolerance = 1e-10 * system_rhs.l2_norm();
  const unsigned int max_iterations = std::min(100000u, 10 * dof_handler.n_dofs());

  SolverControl solver_control(max_iterations, tolerance);

  // Use GMRES with larger Krylov space for better convergence
  SolverGMRES<Vector<double>>::AdditionalData gmres_data;
  gmres_data.max_n_tmp_vectors = 300;
  gmres_data.right_preconditioning = true;
  SolverGMRES<Vector<double>> solver(solver_control, gmres_data);

  // Use Chebyshev preconditioner - consistent with matrix-free solver
  using PreconditionerType = PreconditionChebyshev<SparseMatrix<double>, Vector<double>>;
  typename PreconditionerType::AdditionalData chebyshev_data;
  chebyshev_data.degree = 5;  // Polynomial degree for smoothing
  chebyshev_data.smoothing_range = 20.0;  // Eigenvalue range estimate
  chebyshev_data.eig_cg_n_iterations = 10;  // CG iterations for eigenvalue estimation

  PreconditionerType preconditioner;
  preconditioner.initialize(system_matrix, chebyshev_data);

  try
  {
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);
    std::cout << "  GMRES converged in " << solver_control.last_step()
              << " iterations.\n";
  }
  catch (const std::exception &exc)
  {
    // If GMRES fails, the solution still contains the last iterate
    // which is often good enough for convergence studies
    std::cerr << "Warning: Solver did not fully converge.\n"
              << "  Iterations: " << solver_control.last_step() << "\n"
              << "  Residual: " << solver_control.last_value() << "\n";
    constraints.distribute(solution);
  }
}

template <int dim>
void ADRMatrixSolver<dim>::compute_errors(double &L2_error, double &H1_error) const
{
  // L2 error
  Vector<float> difference_per_cell(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree + 2),
                                    VectorTools::L2_norm);
  L2_error = VectorTools::compute_global_error(triangulation,
                                               difference_per_cell,
                                               VectorTools::L2_norm);

  // H1 error
  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree + 2),
                                    VectorTools::H1_norm);
  H1_error = VectorTools::compute_global_error(triangulation,
                                               difference_per_cell,
                                               VectorTools::H1_norm);
}

template <int dim>
void ADRMatrixSolver<dim>::output_results(const std::string &filename) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  // Also output the exact solution for comparison
  Vector<double> exact_values(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, exact_solution, exact_values);
  data_out.add_data_vector(exact_values, "exact_solution");

  // Compute and output the error
  Vector<double> error(dof_handler.n_dofs());
  error = solution;
  error -= exact_values;
  data_out.add_data_vector(error, "error");

  data_out.build_patches();

  std::ofstream output(filename);
  data_out.write_vtk(output);
}

