#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <set>
#include <limits>

#ifdef ENABLE_PERF_COUNTERS
#include "traffic_counters.hpp"
#endif

using namespace dealii;

/**
 * MPI-parallel matrix-free operator for ADR equation.
 * NO PRECOMPUTATION - coefficients evaluated on-the-fly.
 */
template <int dim, int fe_degree, typename number = double>
class ADRMatrixFreeOperatorMPI
  : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
{
public:
  using value_type = number;
  using VectorType = LinearAlgebra::distributed::Vector<number>;

  ADRMatrixFreeOperatorMPI()
    : diffusion_function(nullptr)
    , advection_function(nullptr)
    , reaction_function(nullptr)
  {}

  void clear() override
  {
    diffusion_function = nullptr;
    advection_function = nullptr;
    reaction_function = nullptr;
    MatrixFreeOperators::Base<dim, VectorType>::clear();
  }

  void set_coefficients(const Function<dim> &diffusion,
                        const TensorFunction<1, dim> &advection,
                        const Function<dim> &reaction)
  {
    diffusion_function = &diffusion;
    advection_function = &advection;
    reaction_function = &reaction;
  }

  void compute_diagonal() override
  {
    Assert(diffusion_function != nullptr &&
           advection_function != nullptr &&
           reaction_function != nullptr,
           ExcMessage("Coefficients must be set before computing diagonal"));

    this->inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
    auto &diagonal = this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(diagonal);

    MatrixFreeOperators::Base<dim, VectorType>::set_constrained_entries_to_one(diagonal);

    const auto &matrix_free = *this->get_matrix_free();
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(matrix_free);

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      phi.reinit(cell);

      VectorizedArray<number> local_diagonal[phi.tensor_dofs_per_cell];
      for (unsigned int i = 0; i < phi.tensor_dofs_per_cell; ++i)
        local_diagonal[i] = 0.0;

      for (unsigned int i = 0; i < phi.tensor_dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < phi.tensor_dofs_per_cell; ++j)
          phi.begin_dof_values()[j] = (i == j) ? 1.0 : 0.0;

        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          // Evaluate coefficients on-the-fly
          const auto q_points = phi.quadrature_point(q);
          VectorizedArray<number> mu = make_vectorized_array<number>(0.0);
          Tensor<1, dim, VectorizedArray<number>> beta;
          VectorizedArray<number> gamma = make_vectorized_array<number>(0.0);

          for (unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
          {
            Point<dim> q_point;
            for (unsigned int d = 0; d < dim; ++d)
              q_point[d] = q_points[d][v];

            mu[v] = diffusion_function->value(q_point);
            gamma[v] = reaction_function->value(q_point);
            const auto beta_tensor = advection_function->value(q_point);
            for (unsigned int d = 0; d < dim; ++d)
              beta[d][v] = beta_tensor[d];
          }

          phi.submit_gradient(mu * phi.get_gradient(q), q);
          phi.submit_value(beta * phi.get_gradient(q) + gamma * phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        local_diagonal[i] = phi.begin_dof_values()[i];
      }

      for (unsigned int i = 0; i < phi.tensor_dofs_per_cell; ++i)
        phi.begin_dof_values()[i] = local_diagonal[i];

      phi.distribute_local_to_global(diagonal);
    }

    diagonal.compress(VectorOperation::add);

    for (unsigned int i = 0; i < diagonal.locally_owned_size(); ++i)
      if (diagonal.local_element(i) != 0.0)
        diagonal.local_element(i) = 1.0 / diagonal.local_element(i);
  }

private:
  void apply_add(VectorType &dst, const VectorType &src) const override
  {
    this->data->cell_loop(&ADRMatrixFreeOperatorMPI::local_apply, this, dst, src);
  }

  void local_apply(const MatrixFree<dim, number> &data,
                   VectorType &dst,
                   const VectorType &src,
                   const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    Assert(diffusion_function != nullptr &&
           advection_function != nullptr &&
           reaction_function != nullptr,
           ExcMessage("Coefficients must be set before applying operator"));

    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

    uint64_t local_processed_dofs = 0;

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      local_processed_dofs += phi.tensor_dofs_per_cell;

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        const auto value = phi.get_value(q);
        const auto gradient = phi.get_gradient(q);

        // Evaluate coefficients on-the-fly
        const auto q_points = phi.quadrature_point(q);
        VectorizedArray<number> mu = make_vectorized_array<number>(0.0);
        Tensor<1, dim, VectorizedArray<number>> beta;
        VectorizedArray<number> gamma = make_vectorized_array<number>(0.0);

        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
        {
          Point<dim> q_point;
          for (unsigned int d = 0; d < dim; ++d)
            q_point[d] = q_points[d][v];

          mu[v] = diffusion_function->value(q_point);
          gamma[v] = reaction_function->value(q_point);
          const auto beta_tensor = advection_function->value(q_point);
          for (unsigned int d = 0; d < dim; ++d)
            beta[d][v] = beta_tensor[d];
        }

        phi.submit_gradient(mu * gradient, q);
        phi.submit_value(beta * gradient + gamma * value, q);
      }

      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }

#ifdef ENABLE_PERF_COUNTERS
    traffic_counters::add_mf(local_processed_dofs);
#endif
  }

  const Function<dim> *diffusion_function;
  const TensorFunction<1, dim> *advection_function;
  const Function<dim> *reaction_function;
};


/**
 * MPI-parallel matrix-free solver for ADR equation with mixed boundary conditions.
 * Uses Geometric Multigrid (GMG) with Chebyshev smoothers for preconditioning.
 * NO PRECOMPUTATION - all coefficients evaluated on-the-fly.
 */
template <int dim, int fe_degree>
class ADRMatrixFreeSolverMPI
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<double>;
  using LevelMatrixType = ADRMatrixFreeOperatorMPI<dim, fe_degree, double>;
  struct SolveDiagnostics
  {
    unsigned int iterations = 0;
    bool converged = false;
    double preconditioner_setup_time = 0.0;
    double krylov_time = 0.0;
    double final_residual = 0.0;
  };

  ADRMatrixFreeSolverMPI(MPI_Comm mpi_comm,
                         const Function<dim> &exact_sol,
                         const Function<dim> &rhs,
                         const Function<dim> &mu,
                         const TensorFunction<1, dim> &beta,
                         const Function<dim> &gamma,
                         const Function<dim> &neumann_bc,
                         const std::set<types::boundary_id> &dirichlet_ids,
                         const std::set<types::boundary_id> &neumann_ids);

  void make_grid(unsigned int n_refinements);
  void setup_system();
  void setup_multigrid();
  void assemble_rhs();
  void solve(unsigned int fixed_iterations = 0);
  void compute_errors(double &L2_error, double &H1_error) const;
  void output_results(const std::string &filename) const;

  types::global_dof_index n_dofs() const { return dof_handler.n_dofs(); }
  unsigned int n_cells() const { return triangulation.n_global_active_cells(); }
  types::global_dof_index n_locally_owned_dofs() const
  {
    return dof_handler.locally_owned_dofs().n_elements();
  }
  unsigned int n_locally_owned_cells() const
  {
    return triangulation.n_locally_owned_active_cells();
  }
  const SolveDiagnostics &get_solve_diagnostics() const
  {
    return solve_diagnostics;
  }

private:
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;
  MappingQ1<dim> mapping;

  AffineConstraints<double> constraints;

  std::shared_ptr<MatrixFree<dim, double>> matrix_free_data;
  ADRMatrixFreeOperatorMPI<dim, fe_degree, double> system_operator;

  MGLevelObject<std::shared_ptr<MatrixFree<dim, double>>> mg_matrix_free;
  MGLevelObject<LevelMatrixType> mg_operators;
  MGLevelObject<AffineConstraints<double>> mg_constraints;
  MGConstrainedDoFs mg_constrained_dofs;

  VectorType solution;
  VectorType system_rhs;
  VectorType lifting;

  const Function<dim> &exact_solution;
  const Function<dim> &right_hand_side;
  const Function<dim> &diffusion_coefficient;
  const TensorFunction<1, dim> &advection_field;
  const Function<dim> &reaction_coefficient;
  const Function<dim> &neumann_boundary_function;

  std::set<types::boundary_id> dirichlet_boundary_ids;
  std::set<types::boundary_id> neumann_boundary_ids;
  SolveDiagnostics solve_diagnostics;
};


template <int dim, int fe_degree>
ADRMatrixFreeSolverMPI<dim, fe_degree>::ADRMatrixFreeSolverMPI(
  MPI_Comm mpi_comm,
  const Function<dim> &exact_sol,
  const Function<dim> &rhs,
  const Function<dim> &mu,
  const TensorFunction<1, dim> &beta,
  const Function<dim> &gamma,
  const Function<dim> &neumann_bc,
  const std::set<types::boundary_id> &dirichlet_ids,
  const std::set<types::boundary_id> &neumann_ids)
  : mpi_communicator(mpi_comm)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm))
  , pcout(std::cout, this_mpi_process == 0)
  , triangulation(mpi_comm,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening),
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
  , fe(fe_degree)
  , dof_handler(triangulation)
  , exact_solution(exact_sol)
  , right_hand_side(rhs)
  , diffusion_coefficient(mu)
  , advection_field(beta)
  , reaction_coefficient(gamma)
  , neumann_boundary_function(neumann_bc)
  , dirichlet_boundary_ids(dirichlet_ids)
  , neumann_boundary_ids(neumann_ids)
{}


template <int dim, int fe_degree>
void ADRMatrixFreeSolverMPI<dim, fe_degree>::make_grid(const unsigned int n_refinements)
{
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0, true);
  triangulation.refine_global(n_refinements);
}


template <int dim, int fe_degree>
void ADRMatrixFreeSolverMPI<dim, fe_degree>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  pcout << "  Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
  pcout << "  Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  Functions::ZeroFunction<dim> zero_function;
  for (const auto &boundary_id : dirichlet_boundary_ids)
  {
    VectorTools::interpolate_boundary_values(dof_handler, boundary_id,
                                             zero_function, constraints);
  }
  constraints.close();

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    (update_values | update_gradients | update_JxW_values | update_quadrature_points);
  additional_data.mapping_update_flags_boundary_faces =
    (update_values | update_JxW_values | update_quadrature_points);
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim, double>::AdditionalData::TasksParallelScheme::partition_partition;

  matrix_free_data = std::make_shared<MatrixFree<dim, double>>();
  matrix_free_data->reinit(mapping,
                           dof_handler,
                           constraints,
                           QGauss<1>(fe_degree + 1),
                           additional_data);

  system_operator.clear();
  system_operator.initialize(matrix_free_data);
  system_operator.set_coefficients(diffusion_coefficient,
                                   advection_field,
                                   reaction_coefficient);

  matrix_free_data->initialize_dof_vector(solution);
  matrix_free_data->initialize_dof_vector(system_rhs);
  matrix_free_data->initialize_dof_vector(lifting);

  setup_multigrid();
}


template <int dim, int fe_degree>
void ADRMatrixFreeSolverMPI<dim, fe_degree>::setup_multigrid()
{
  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(dof_handler);
  for (const auto &boundary_id : dirichlet_boundary_ids)
  {
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, {boundary_id});
  }

  const unsigned int min_level = 0;
  const unsigned int max_level = triangulation.n_global_levels() - 1;

  mg_matrix_free.resize(min_level, max_level);
  mg_operators.resize(min_level, max_level);
  mg_constraints.resize(min_level, max_level);

  for (unsigned int level = min_level; level <= max_level; ++level)
  {
    const IndexSet relevant_dofs =
      DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);

    mg_constraints[level].clear();
    mg_constraints[level].reinit(relevant_dofs);
    mg_constraints[level].add_lines(
      mg_constrained_dofs.get_boundary_indices(level));
    mg_constraints[level].close();

    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
      (update_values | update_gradients | update_JxW_values | update_quadrature_points);
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, double>::AdditionalData::TasksParallelScheme::partition_partition;
    additional_data.mg_level = level;

    mg_matrix_free[level] = std::make_shared<MatrixFree<dim, double>>();
    mg_matrix_free[level]->reinit(mapping,
                                   dof_handler,
                                   mg_constraints[level],
                                   QGauss<1>(fe_degree + 1),
                                   additional_data);

    mg_operators[level].clear();
    mg_operators[level].initialize(mg_matrix_free[level]);
    mg_operators[level].set_coefficients(diffusion_coefficient,
                                         advection_field,
                                         reaction_coefficient);
    mg_operators[level].compute_diagonal();
  }
}


template <int dim, int fe_degree>
void ADRMatrixFreeSolverMPI<dim, fe_degree>::assemble_rhs()
{
  system_rhs = 0;
  lifting = 0;

  std::map<types::global_dof_index, double> boundary_values;
  for (const auto &boundary_id : dirichlet_boundary_ids)
  {
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             boundary_id,
                                             exact_solution,
                                             boundary_values);
  }
  for (const auto &pair : boundary_values)
  {
    if (lifting.locally_owned_elements().is_element(pair.first))
      lifting(pair.first) = pair.second;
  }
  lifting.update_ghost_values();

  // Assemble RHS = f - A*lifting (NO precomputed tables)
  FEEvaluation<dim, fe_degree> fe_eval(*matrix_free_data);

  for (unsigned int cell = 0; cell < matrix_free_data->n_cell_batches(); ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values_plain(lifting);
    fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      const auto value = fe_eval.get_value(q);
      const auto gradient = fe_eval.get_gradient(q);

      // Evaluate ALL coefficients on-the-fly (including forcing term)
      const auto q_points = fe_eval.quadrature_point(q);
      VectorizedArray<double> mu = make_vectorized_array<double>(0.0);
      Tensor<1, dim, VectorizedArray<double>> beta;
      VectorizedArray<double> gamma = make_vectorized_array<double>(0.0);
      VectorizedArray<double> forcing = make_vectorized_array<double>(0.0);

      for (unsigned int v = 0; v < matrix_free_data->n_active_entries_per_cell_batch(cell); ++v)
      {
        Point<dim> q_point;
        for (unsigned int d = 0; d < dim; ++d)
          q_point[d] = q_points[d][v];

        mu[v] = diffusion_coefficient.value(q_point);
        gamma[v] = reaction_coefficient.value(q_point);
        forcing[v] = right_hand_side.value(q_point);
        const auto beta_tensor = advection_field.value(q_point);
        for (unsigned int d = 0; d < dim; ++d)
          beta[d][v] = beta_tensor[d];
      }

      // Submit -A*lifting + f
      fe_eval.submit_gradient(-mu * gradient, q);
      fe_eval.submit_value(-beta * gradient - gamma * value + forcing, q);
    }

    fe_eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, system_rhs);
  }

  // Add Neumann boundary contribution
  if (!neumann_boundary_ids.empty())
  {
    FEFaceEvaluation<dim, fe_degree> fe_face_eval(*matrix_free_data);

    for (unsigned int face = 0; face < matrix_free_data->n_boundary_face_batches(); ++face)
    {
      fe_face_eval.reinit(face);
      const auto boundary_id = fe_face_eval.boundary_id();

      if (neumann_boundary_ids.count(boundary_id) > 0)
      {
        for (const unsigned int q : fe_face_eval.quadrature_point_indices())
        {
          const auto q_point = fe_face_eval.quadrature_point(q);
          VectorizedArray<double> neumann_value = make_vectorized_array(0.0);

          for (unsigned int v = 0; v < matrix_free_data->n_active_entries_per_face_batch(face); ++v)
          {
            Point<dim> point;
            for (unsigned int d = 0; d < dim; ++d)
              point[d] = q_point[d][v];
            neumann_value[v] = neumann_boundary_function.value(point);
          }
          fe_face_eval.submit_value(neumann_value, q);
        }

        fe_face_eval.integrate_scatter(EvaluationFlags::values, system_rhs);
      }
    }
  }

  system_rhs.compress(VectorOperation::add);
}


template <int dim, int fe_degree>
void ADRMatrixFreeSolverMPI<dim, fe_degree>::solve(const unsigned int fixed_iterations)
{
  solution = 0;
  system_rhs.update_ghost_values();
  const double rhs_norm = system_rhs.l2_norm();
  const double abs_tol = (fixed_iterations > 0)
    ? std::numeric_limits<double>::min()
    : std::max(1e-18, 1e-16 * rhs_norm);
  const unsigned int max_iterations = (fixed_iterations > 0)
    ? fixed_iterations
    : std::min(100000u, static_cast<unsigned int>(10 * dof_handler.n_dofs()));

  SolverControl solver_control(max_iterations, abs_tol);
  solve_diagnostics = SolveDiagnostics{};

  using SolverType = SolverGMRES<VectorType>;
  typename SolverType::AdditionalData gmres_data;
  gmres_data.max_n_tmp_vectors = 300;
  gmres_data.right_preconditioning = true;

  SolverType solver(solver_control, gmres_data);

  const unsigned int min_level = 0;
  const unsigned int max_level = triangulation.n_global_levels() - 1;
  Timer precond_setup_timer;
  precond_setup_timer.start();

  MGTransferMatrixFree<dim, double> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  using SmootherType = PreconditionChebyshev<LevelMatrixType, VectorType>;
  mg::SmootherRelaxation<SmootherType, VectorType> mg_smoother;

  MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level, max_level);
  for (unsigned int level = min_level; level <= max_level; ++level)
  {
    smoother_data[level].smoothing_range = 15.0;
    smoother_data[level].degree = 5;
    smoother_data[level].eig_cg_n_iterations = 10;
    smoother_data[level].preconditioner = mg_operators[level].get_matrix_diagonal_inverse();
  }
  mg_smoother.initialize(mg_operators, smoother_data);
  MGCoarseGridApplySmoother<VectorType> mg_coarse;
  mg_coarse.initialize(mg_smoother);
  mg::Matrix<VectorType> mg_matrix(mg_operators);
  Multigrid<VectorType> mg(mg_matrix,
                            mg_coarse,
                            mg_transfer,
                            mg_smoother,
                            mg_smoother,
                            min_level,
                            max_level);

  PreconditionMG<dim, VectorType, MGTransferMatrixFree<dim, double>>
  preconditioner(dof_handler, mg, mg_transfer);
  solve_diagnostics.preconditioner_setup_time = precond_setup_timer.wall_time();
  constraints.set_zero(solution);
  Timer krylov_timer;
  krylov_timer.start();

  try
  {
    solver.solve(system_operator, solution, system_rhs, preconditioner);
    solve_diagnostics.converged = true;
    pcout << "  GMRES converged in " << solver_control.last_step()
          << " iterations.\n";
  }
  catch (const SolverControl::NoConvergence &)
  {
    solve_diagnostics.converged = false;
    pcout << "  GMRES reached max iterations (" << solver_control.last_step()
          << ") without convergence.\n";
  }
  catch (const std::exception &exc)
  {
    solve_diagnostics.converged = false;
    pcout << "  Warning: GMRES not fully converged.\n"
          << "  Iterations: " << solver_control.last_step() << "\n"
          << "  Residual: " << solver_control.last_value() << "\n";
  }
  solve_diagnostics.krylov_time = krylov_timer.wall_time();
  solve_diagnostics.iterations = solver_control.last_step();
  solve_diagnostics.final_residual = solver_control.last_value();

  constraints.distribute(solution);
  solution += lifting;
  solution.update_ghost_values();
}


template <int dim, int fe_degree>
void ADRMatrixFreeSolverMPI<dim, fe_degree>::compute_errors(double &L2_error, double &H1_error) const
{
  const unsigned int n_locally_owned_cells = triangulation.n_locally_owned_active_cells();
  Vector<float> difference_per_cell(n_locally_owned_cells);

  VectorTools::integrate_difference(dof_handler,
                                    solution,
                                    exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe_degree + 2),
                                    VectorTools::L2_norm);
  L2_error = VectorTools::compute_global_error(triangulation,
                                               difference_per_cell,
                                               VectorTools::L2_norm);

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


template <int dim, int fe_degree>
void ADRMatrixFreeSolverMPI<dim, fe_degree>::output_results(const std::string &filename) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./", filename, 0,
                                      mpi_communicator, 2, 1);
}
