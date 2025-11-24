#pragma once

#include "adr_matrix_free_operator.hpp"

#include <deal.II/base/timer.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

using namespace dealii;

// Matrix-free solver for ADR equation using lifting method for non-homogeneous BC
template <int dim, int fe_degree>
class ADRMatrixFreeSolver
{
public:
  ADRMatrixFreeSolver(Triangulation<dim> &tria,
                      const Function<dim> &exact_sol,
                      const Function<dim> &rhs,
                      const Function<dim> &mu,
                      const TensorFunction<1, dim> &beta,
                      const Function<dim> &gamma);

  void setup_system();
  void assemble_rhs();
  void solve();
  void compute_errors(double &L2_error, double &H1_error) const;
  void output_results(const std::string &filename) const;

  unsigned int n_dofs() const { return dof_handler.n_dofs(); }
  const DoFHandler<dim>& get_dof_handler() const { return dof_handler; }

private:
  Triangulation<dim> &triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraints;

  std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;
  std::shared_ptr<ADRMatrixFreeOperator<dim, fe_degree, double>> system_operator;

  LinearAlgebra::distributed::Vector<double> solution;
  LinearAlgebra::distributed::Vector<double> system_rhs;
  LinearAlgebra::distributed::Vector<double> lifting;

  Table<2, VectorizedArray<double>> forcing_term_values;

  const Function<dim> &exact_solution;
  const Function<dim> &right_hand_side;
  const Function<dim> &diffusion_coefficient;
  const TensorFunction<1, dim> &advection_field;
  const Function<dim> &reaction_coefficient;
};


template <int dim, int fe_degree>
ADRMatrixFreeSolver<dim, fe_degree>::ADRMatrixFreeSolver(
  Triangulation<dim> &tria,
  const Function<dim> &exact_sol,
  const Function<dim> &rhs,
  const Function<dim> &mu,
  const TensorFunction<1, dim> &beta,
  const Function<dim> &gamma)
  : triangulation(tria)
  , fe(fe_degree)
  , dof_handler(triangulation)
  , exact_solution(exact_sol)
  , right_hand_side(rhs)
  , diffusion_coefficient(mu)
  , advection_field(beta)
  , reaction_coefficient(gamma)
{}

template <int dim, int fe_degree>
void ADRMatrixFreeSolver<dim, fe_degree>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  // Setup homogeneous constraints for lifting method
  constraints.clear();
  Functions::ZeroFunction<dim> zero_function;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
                                           zero_function, constraints);
  constraints.close();

  // Setup MatrixFree
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    (update_values | update_gradients | update_JxW_values | update_quadrature_points);
  QGauss<1> quadrature(fe_degree + 1);

  matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();
  matrix_free_storage->reinit(MappingQ1<dim>(),
                              dof_handler,
                              constraints,
                              quadrature,
                              additional_data);

  system_operator = std::make_shared<ADRMatrixFreeOperator<dim, fe_degree, double>>();
  system_operator->initialize(matrix_free_storage);
  system_operator->evaluate_coefficients(diffusion_coefficient,
                                        advection_field,
                                        reaction_coefficient);

  // Pre-compute forcing term values
  FEEvaluation<dim, fe_degree> phi(*matrix_free_storage);
  const unsigned int n_cells = matrix_free_storage->n_cell_batches();
  forcing_term_values.reinit(n_cells, phi.n_q_points);

  for (unsigned int cell = 0; cell < n_cells; ++cell)
  {
    phi.reinit(cell);
    for (unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      const auto q_point_vectorized = phi.quadrature_point(q);
      for (unsigned int v = 0; v < matrix_free_storage->n_active_entries_per_cell_batch(cell); ++v)
      {
        Point<dim> q_point;
        for (unsigned int d = 0; d < dim; ++d)
          q_point[d] = q_point_vectorized[d][v];
        forcing_term_values(cell, q)[v] = right_hand_side.value(q_point);
      }
    }
  }

  // Initialization
  matrix_free_storage->initialize_dof_vector(solution);
  matrix_free_storage->initialize_dof_vector(system_rhs);
  matrix_free_storage->initialize_dof_vector(lifting);

  solution = 0;
  system_rhs = 0;
  lifting = 0;
}

template <int dim, int fe_degree>
void ADRMatrixFreeSolver<dim, fe_degree>::assemble_rhs()
{
  system_rhs = 0;
  lifting = 0;

  // Interpolate Dirichlet BC into lifting vector
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
                                           exact_solution, boundary_values);
  for (const auto &pair : boundary_values)
    if (lifting.locally_owned_elements().is_element(pair.first))
      lifting(pair.first) = pair.second;
  lifting.update_ghost_values();

  // Get coefficient tables from operator
  const auto &diffusion_table = system_operator->get_diffusion_coefficient();
  const auto &advection_table = system_operator->get_advection_field();
  const auto &reaction_table = system_operator->get_reaction_coefficient();

  // Assemble RHS = f - A*lifting
  FEEvaluation<dim, fe_degree> fe_eval(*matrix_free_storage);

  for (unsigned int cell = 0; cell < matrix_free_storage->n_cell_batches(); ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values_plain(lifting);
    fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      const auto value = fe_eval.get_value(q);
      const auto gradient = fe_eval.get_gradient(q);

      const auto mu = diffusion_table(cell, q);
      const auto beta = advection_table(cell, q);
      const auto gamma = reaction_table(cell, q);

      // Submit -A*lifting + f
      fe_eval.submit_gradient(-mu * gradient, q);
      fe_eval.submit_value(-beta * gradient - gamma * value + forcing_term_values(cell, q), q);
    }

    fe_eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                              system_rhs);
  }

  system_rhs.compress(VectorOperation::add);
}

template <int dim, int fe_degree>
void ADRMatrixFreeSolver<dim, fe_degree>::solve()
{
  solution = 0;
  system_rhs.update_ghost_values();

  const double tolerance = 1e-10 * system_rhs.l2_norm();
  const unsigned int max_iterations = std::min(100000u, 10 * dof_handler.n_dofs());

  SolverControl solver_control(max_iterations, tolerance);
  SolverGMRES<LinearAlgebra::distributed::Vector<double>>::AdditionalData gmres_data;
  gmres_data.max_n_tmp_vectors = 300;  // Increased from 200
  gmres_data.right_preconditioning = true;  // Right preconditioning

  SolverGMRES<LinearAlgebra::distributed::Vector<double>> solver(solver_control, gmres_data);

  // Compute diagonal for Chebyshev preconditioner
  system_operator->compute_diagonal();

  // Use Chebyshev preconditioner - much better than plain diagonal for ADR problems
  using PreconditionerType = PreconditionChebyshev<ADRMatrixFreeOperator<dim, fe_degree, double>,
                                                    LinearAlgebra::distributed::Vector<double>>;
  typename PreconditionerType::AdditionalData chebyshev_data;
  chebyshev_data.preconditioner = system_operator->get_matrix_diagonal_inverse();
  chebyshev_data.degree = 5;  // Polynomial degree for smoothing
  chebyshev_data.smoothing_range = 20.0;  // Eigenvalue range estimate
  chebyshev_data.eig_cg_n_iterations = 10;  // CG iterations for eigenvalue estimation

  PreconditionerType preconditioner;
  preconditioner.initialize(*system_operator, chebyshev_data);

  constraints.set_zero(solution);

  try
  {
    solver.solve(*system_operator, solution, system_rhs, preconditioner);
    std::cout << "  GMRES converged in " << solver_control.last_step()
              << " iterations.\n";
  }
  catch (const std::exception &exc)
  {
    std::cerr << "Warning: GMRES not fully converge.\n"
              << "  Iterations: " << solver_control.last_step() << "\n"
              << "  Residual: " << solver_control.last_value() << "\n";
  }

  constraints.distribute(solution);
  solution += lifting;  // Add lifting to restore non-homogeneous BC
}

template <int dim, int fe_degree>
void ADRMatrixFreeSolver<dim, fe_degree>::compute_errors(double &L2_error, double &H1_error) const
{
  Vector<double> solution_local(solution.size());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution_local(i) = solution(i);

  Vector<float> difference_per_cell(triangulation.n_active_cells());

  // L2 error
  VectorTools::integrate_difference(dof_handler,
                                    solution_local,
                                    exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree + 2),
                                    VectorTools::L2_norm);
  L2_error = VectorTools::compute_global_error(triangulation,
                                               difference_per_cell,
                                               VectorTools::L2_norm);

  // H1 error
  VectorTools::integrate_difference(dof_handler,
                                    solution_local,
                                    exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree + 2),
                                    VectorTools::H1_norm);
  H1_error = VectorTools::compute_global_error(triangulation,
                                               difference_per_cell,
                                               VectorTools::H1_norm);
}

template <int dim, int fe_degree>
void ADRMatrixFreeSolver<dim, fe_degree>::output_results(const std::string &filename) const
{
  Vector<double> solution_local(solution.size());
  for (unsigned int i = 0; i < solution.size(); ++i)
    solution_local(i) = solution(i);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_local, "solution");

  Vector<double> exact_values(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, exact_solution, exact_values);
  data_out.add_data_vector(exact_values, "exact_solution");

  Vector<double> error = solution_local;
  error -= exact_values;
  data_out.add_data_vector(error, "error");

  data_out.build_patches();

  std::ofstream output(filename);
  data_out.write_vtk(output);
}

