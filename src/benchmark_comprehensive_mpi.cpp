/**
 * Pure MPI benchmark: Matrix-Free vs Matrix-Based solvers.
 *
 * Matrix-Free: No precomputation, all coefficients on-the-fly, MF GMG
 * Matrix-Based: Assembled matrices at ALL levels (fine + MG), matrix-based GMG
 *
 * Run with: mpirun -np N ./benchmark_pure_mpi [options]
 */

#include "adr_matrix_free_solver_mpi.hpp"
#include "manufactured_solution_mixed_bc.hpp"

#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/solver_gmres.h> 
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <limits>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

#ifdef ENABLE_PERF_COUNTERS
#include "traffic_counters.hpp"
#endif

using namespace dealii;

constexpr unsigned int dim = 2;


// Get memory usage
struct MemorySnapshot
{
  double rss_mb = 0.0;
  double hwm_mb = 0.0;
  double vmsize_mb = 0.0;
  double vmdata_mb = 0.0;
  double heap_alloc_mb = 0.0;
  double heap_reserved_mb = 0.0;
  double memtotal_mb = 0.0;
  double memavail_mb = 0.0;
};

struct MemoryInfo
{
  static double read_kb_from_file(const std::string &path, const std::string &key)
  {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
    {
      std::istringstream iss(line);
      std::string label;
      double value = 0.0;
      if (!(iss >> label >> value))
        continue;
      if (label == key)
        return value;
    }
    return 0.0;
  }

  static MemorySnapshot snapshot()
  {
    MemorySnapshot snap;
    snap.rss_mb = read_kb_from_file("/proc/self/status", "VmRSS:") / 1024.0;
    snap.hwm_mb = read_kb_from_file("/proc/self/status", "VmHWM:") / 1024.0;
    snap.vmsize_mb = read_kb_from_file("/proc/self/status", "VmSize:") / 1024.0;
    snap.vmdata_mb = read_kb_from_file("/proc/self/status", "VmData:") / 1024.0;
#if defined(__GLIBC__)
    const struct mallinfo2 mi = mallinfo2();
    snap.heap_alloc_mb = static_cast<double>(mi.uordblks) / (1024.0 * 1024.0);
    snap.heap_reserved_mb = static_cast<double>(mi.arena + mi.hblkhd) / (1024.0 * 1024.0);
#endif
    snap.memtotal_mb = read_kb_from_file("/proc/meminfo", "MemTotal:") / 1024.0;
    snap.memavail_mb = read_kb_from_file("/proc/meminfo", "MemAvailable:") / 1024.0;
    return snap;
  }
};

struct Stats
{
  double min = 0.0;
  double max = 0.0;
  double avg = 0.0;
};

static Stats mpi_stats(double value, MPI_Comm comm)
{
  const auto mma = Utilities::MPI::min_max_avg(value, comm);
  return {mma.min, mma.max, mma.avg};
}

static double mpi_sum(double value, MPI_Comm comm)
{
  return Utilities::MPI::sum(value, comm);
}


// Analytical memory estimates
struct MemoryEstimate
{
  static double matrix_free_mb(types::global_dof_index n_dofs,
                               unsigned int n_cells,
                               unsigned int fe_degree)
  {
    const unsigned int n_q_points = std::pow(fe_degree + 1, dim);

    // Vectors (with ghost overhead ~20%)
    double vectors = 3.0 * n_dofs * sizeof(double) * 1.2;

    // MatrixFree cell data
    double jacobians = n_cells * n_q_points * sizeof(double);
    double inv_jacobians = n_cells * n_q_points * dim * dim * sizeof(double);
    double jxw_values = n_cells * n_q_points * sizeof(double);

    // Cell-to-DoF indexing (approximate)
    unsigned int dofs_per_cell = std::pow(fe_degree + 1, dim);
    double cell_dof_indices = n_cells * dofs_per_cell * sizeof(unsigned int);

    // Constraint matrix data (sparse, hard to estimate - use 10% of DoFs)
    double constraints = 0.1 * n_dofs * sizeof(double);

    // MG structures (very rough - each level is ~1/4 size in 2D)
    unsigned int n_mg_levels = std::log2(n_cells / 4) + 1;  // Approximate
    double mg_overhead = (vectors + jacobians + inv_jacobians + jxw_values) * 0.33; // ~1/3 for MG

    return (vectors + jacobians + inv_jacobians + jxw_values +
            cell_dof_indices + constraints + mg_overhead) / (1024.0 * 1024.0);
  }
  static double matrix_based_mb(types::global_dof_index n_dofs,
                                 unsigned int fe_degree,
                                 unsigned int n_mg_levels)
  {
    // Vectors: solution, rhs (2 vectors)
    double vectors = 2.0 * n_dofs * sizeof(double);

    // Fine level sparse matrix
    unsigned int avg_nnz_per_row = std::pow(2 * fe_degree + 1, dim) * 0.7; // ~70% due to boundaries

    double fine_matrix = n_dofs * avg_nnz_per_row * (sizeof(double) + sizeof(int));

    // MG level matrices (roughly 1/4 size per level for 2D)
    double mg_matrices = 0.0;
    types::global_dof_index level_dofs = n_dofs;
    for (unsigned int l = 0; l < n_mg_levels - 1; ++l)
    {
      level_dofs /= 4;  // Approximate for 2D
      mg_matrices += level_dofs * avg_nnz_per_row * (sizeof(double) + sizeof(int));
    }

    return (vectors + fine_matrix + mg_matrices) / (1024.0 * 1024.0);
  }
};


/**
 * Pure matrix-based solver with assembled matrices at ALL levels.
 */
struct MatrixBasedSolverOptions
{
  unsigned int chebyshev_degree = 5;
  double chebyshev_smoothing_range = 15.0;
  unsigned int chebyshev_eig_iterations = 10;
  double jacobi_omega = 1.0;
  unsigned int coarse_chebyshev_degree = 12;
};

template <int dim, int fe_degree>
class ADRPureMatrixSolverMPI
{
public:
  struct SolveDiagnostics
  {
    unsigned int iterations = 0;
    bool converged = false;
    double preconditioner_setup_time = 0.0;
    double krylov_time = 0.0;
    double final_residual = 0.0;
  };

  ADRPureMatrixSolverMPI(MPI_Comm mpi_comm,
                         const Function<dim> &exact_sol,
                         const Function<dim> &rhs,
                         const Function<dim> &mu,
                         const TensorFunction<1, dim> &beta,
                         const Function<dim> &gamma,
                         const Function<dim> &neumann_bc,
                         const std::set<types::boundary_id> &dirichlet_ids,
                         const std::set<types::boundary_id> &neumann_ids,
                         const MatrixBasedSolverOptions &solver_options);

  void make_grid(unsigned int n_refinements);
  void setup_system();
  void setup_multigrid();
  void assemble_system();
  void assemble_multigrid_matrices();
  void solve(unsigned int fixed_iterations = 0);
  void compute_errors(double &L2_error, double &H1_error) const;

  types::global_dof_index n_dofs() const { return dof_handler.n_dofs(); }
  unsigned int n_cells() const { return triangulation.n_global_active_cells(); }
  types::global_dof_index n_locally_owned_dofs() const
  {
    return locally_owned_dofs.n_elements();
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
  void assemble_level_matrix(unsigned int level,
                             TrilinosWrappers::SparseMatrix &level_matrix);

  MPI_Comm mpi_communicator;
  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;
  MappingQ1<dim> mapping;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector system_rhs;
  TrilinosWrappers::MPI::Vector solution;

  // Pure matrix-based GMG
  MGLevelObject<TrilinosWrappers::SparseMatrix> mg_matrices;
  MGLevelObject<AffineConstraints<double>> mg_constraints;
  MGConstrainedDoFs mg_constrained_dofs;

  const Function<dim> &exact_solution;
  const Function<dim> &right_hand_side;
  const Function<dim> &diffusion_coefficient;
  const TensorFunction<1, dim> &advection_field;
  const Function<dim> &reaction_coefficient;
  const Function<dim> &neumann_boundary_function;

  std::set<types::boundary_id> dirichlet_boundary_ids;
  std::set<types::boundary_id> neumann_boundary_ids;
  MatrixBasedSolverOptions options;
  SolveDiagnostics solve_diagnostics;
};


template <int dim, int fe_degree>
ADRPureMatrixSolverMPI<dim, fe_degree>::ADRPureMatrixSolverMPI(
  MPI_Comm mpi_comm,
  const Function<dim> &exact_sol,
  const Function<dim> &rhs,
  const Function<dim> &mu,
  const TensorFunction<1, dim> &beta,
  const Function<dim> &gamma,
  const Function<dim> &neumann_bc,
  const std::set<types::boundary_id> &dirichlet_ids,
  const std::set<types::boundary_id> &neumann_ids,
  const MatrixBasedSolverOptions &solver_options)
  : mpi_communicator(mpi_comm)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0)
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
  , options(solver_options)
{}


template <int dim, int fe_degree>
void ADRPureMatrixSolverMPI<dim, fe_degree>::make_grid(unsigned int n_refinements)
{
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0, true);
  triangulation.refine_global(n_refinements);
}


template <int dim, int fe_degree>
void ADRPureMatrixSolverMPI<dim, fe_degree>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  for (const auto &boundary_id : dirichlet_boundary_ids)
  {
    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_id,
                                             exact_solution,
                                             constraints);
  }
  constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  solution.reinit(locally_owned_dofs, mpi_communicator);

  setup_multigrid();
}



template <int dim, int fe_degree>
void ADRPureMatrixSolverMPI<dim, fe_degree>::assemble_system()
{
  system_matrix = 0;
  system_rhs = 0;

  const QGauss<dim> quadrature(fe.degree + 1);
  const QGauss<dim - 1> face_quadrature(fe.degree + 1);

  FEValues<dim> fe_values(fe, quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  FEFaceValues<dim> fe_face_values(fe, face_quadrature,
                                   update_values | update_quadrature_points |
                                   update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature.size();
  const unsigned int n_face_q_points = face_quadrature.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    cell_matrix = 0;
    cell_rhs = 0;
    fe_values.reinit(cell);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const auto &q_point = fe_values.quadrature_point(q);
      const double mu = diffusion_coefficient.value(q_point);
      const Tensor<1, dim> beta = advection_field.value(q_point);
      const double gamma_val = reaction_coefficient.value(q_point);
      const double f = right_hand_side.value(q_point);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          cell_matrix(i, j) += (mu * fe_values.shape_grad(i, q) *
                                    fe_values.shape_grad(j, q)
                                + beta * fe_values.shape_grad(j, q) *
                                    fe_values.shape_value(i, q)
                                + gamma_val * fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q))
                               * fe_values.JxW(q);
        }
        cell_rhs(i) += f * fe_values.shape_value(i, q) * fe_values.JxW(q);
      }
    }

    // Neumann BC
    for (const auto &face : cell->face_iterators())
    {
      if (face->at_boundary() &&
          neumann_boundary_ids.count(face->boundary_id()) > 0)
      {
        fe_face_values.reinit(cell, face);
        for (unsigned int q = 0; q < n_face_q_points; ++q)
        {
          const double g_N = neumann_boundary_function.value(
            fe_face_values.quadrature_point(q));
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            cell_rhs(i) += g_N * fe_face_values.shape_value(i, q) *
                           fe_face_values.JxW(q);
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                           local_dof_indices,
                                           system_matrix, system_rhs);
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}


template <int dim, int fe_degree>
void ADRPureMatrixSolverMPI<dim, fe_degree>::assemble_level_matrix(
  unsigned int level,
  TrilinosWrappers::SparseMatrix &level_matrix)
{
  level_matrix = 0;

  const QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.cell_iterators_on_level(level))
  {
    if (!cell->is_locally_owned_on_level())
      continue;

    cell_matrix = 0;
    fe_values.reinit(cell);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      const auto &q_point = fe_values.quadrature_point(q);
      const double mu = diffusion_coefficient.value(q_point);
      const Tensor<1, dim> beta = advection_field.value(q_point);
      const double gamma_val = reaction_coefficient.value(q_point);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          cell_matrix(i, j) += (mu * fe_values.shape_grad(i, q) *
                                    fe_values.shape_grad(j, q)
                                + beta * fe_values.shape_grad(j, q) *
                                    fe_values.shape_value(i, q)
                                + gamma_val * fe_values.shape_value(i, q) *
                                    fe_values.shape_value(j, q))
                               * fe_values.JxW(q);
        }
      }
    }

    cell->get_mg_dof_indices(local_dof_indices);
    mg_constraints[level].distribute_local_to_global(cell_matrix,
                                                     local_dof_indices,
                                                     level_matrix);
  }

  level_matrix.compress(VectorOperation::add);
}


template <int dim, int fe_degree>
void ADRPureMatrixSolverMPI<dim, fe_degree>::assemble_multigrid_matrices()
{
  const unsigned int min_level = 0;
  const unsigned int max_level = triangulation.n_global_levels() - 1;

  for (unsigned int level = min_level; level <= max_level; ++level)
  {
    assemble_level_matrix(level, mg_matrices[level]);
  }
}

template <int dim, int fe_degree>
void ADRPureMatrixSolverMPI<dim, fe_degree>::solve(const unsigned int fixed_iterations)
{
  TrilinosWrappers::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

  const double rhs_norm = system_rhs.l2_norm();
  const double abs_tol = (fixed_iterations > 0)
    ? std::numeric_limits<double>::min()
    : std::max(1e-18, 1e-16 * rhs_norm);
  const unsigned int max_iterations = (fixed_iterations > 0)
    ? fixed_iterations
    : static_cast<unsigned int>(dof_handler.n_dofs());
  SolverControl solver_control(max_iterations, abs_tol);
  solve_diagnostics = SolveDiagnostics{};

  const unsigned int min_level = 0;
  const unsigned int max_level = triangulation.n_global_levels() - 1;
  Timer precond_setup_timer;
  precond_setup_timer.start();

  // --- MG TRANSFER ---
  MGTransferPrebuilt<TrilinosWrappers::MPI::Vector> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  // --- SMOOTHER (Chebyshev) ---
  using SmootherType = PreconditionChebyshev<TrilinosWrappers::SparseMatrix,
                                             TrilinosWrappers::MPI::Vector,
                                             TrilinosWrappers::PreconditionJacobi>;
  mg::SmootherRelaxation<SmootherType, TrilinosWrappers::MPI::Vector> mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level, max_level);

  for (unsigned int level = min_level; level <= max_level; ++level)
  {
    smoother_data[level].smoothing_range = options.chebyshev_smoothing_range;
    smoother_data[level].degree = (level == min_level)
      ? options.coarse_chebyshev_degree
      : options.chebyshev_degree;
    smoother_data[level].eig_cg_n_iterations = options.chebyshev_eig_iterations;
    smoother_data[level].constraints.copy_from(mg_constraints[level]);

    auto level_jacobi = std::make_shared<TrilinosWrappers::PreconditionJacobi>();
    TrilinosWrappers::PreconditionJacobi::AdditionalData jacobi_data;
    jacobi_data.omega = options.jacobi_omega;
    level_jacobi->initialize(mg_matrices[level], jacobi_data);
    smoother_data[level].preconditioner = level_jacobi;
  }
  mg_smoother.initialize(mg_matrices, smoother_data);

  dealii::MGCoarseGridApplySmoother<TrilinosWrappers::MPI::Vector> mg_coarse;
  mg_coarse.initialize(mg_smoother);

  // --- MG SOLVER ---
  mg::Matrix<TrilinosWrappers::MPI::Vector> mg_matrix(mg_matrices);
  Multigrid<TrilinosWrappers::MPI::Vector> mg(mg_matrix,
                                               mg_coarse,
                                               mg_transfer,
                                               mg_smoother,
                                               mg_smoother,
                                               min_level,
                                               max_level);

  PreconditionMG<dim, TrilinosWrappers::MPI::Vector, MGTransferPrebuilt<TrilinosWrappers::MPI::Vector>>
    preconditioner(dof_handler, mg, mg_transfer);
  solve_diagnostics.preconditioner_setup_time = precond_setup_timer.wall_time();

  // --- OUTER GMRES SOLVER ---
  SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData gmres_data;
  gmres_data.max_n_tmp_vectors = 300;
  gmres_data.right_preconditioning = true;
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control, gmres_data);
  Timer krylov_timer;
  krylov_timer.start();

  try {
    solver.solve(system_matrix, completely_distributed_solution, system_rhs, preconditioner);
    solve_diagnostics.converged = true;
    pcout << "  Matrix GMRES converged in " << solver_control.last_step() << " iterations.\n";
  } catch (const SolverControl::NoConvergence &) {
    solve_diagnostics.converged = false;
    pcout << "  Matrix GMRES reached max iterations (" << solver_control.last_step()
          << ") without convergence.\n";
  } catch (std::exception &e) {
    solve_diagnostics.converged = false;
    pcout << "  Error: Matrix solver failed: " << e.what() << "\n";
  }
  solve_diagnostics.krylov_time = krylov_timer.wall_time();
  solve_diagnostics.iterations = solver_control.last_step();
  solve_diagnostics.final_residual = solver_control.last_value();

  constraints.distribute(completely_distributed_solution);
  solution = completely_distributed_solution;
}

template <int dim, int fe_degree>
void ADRPureMatrixSolverMPI<dim, fe_degree>::setup_multigrid()
{
  mg_constrained_dofs.clear();
  // Ensure MG constraints know about Dirichlet boundaries globally
  mg_constrained_dofs.initialize(dof_handler);


  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary_ids);

  const unsigned int min_level = 0;
  const unsigned int max_level = triangulation.n_global_levels() - 1;

  mg_matrices.resize(min_level, max_level);
  mg_constraints.resize(min_level, max_level);

  for (unsigned int level = min_level; level <= max_level; ++level)
  {
    const IndexSet relevant_dofs = DoFTools::extract_locally_relevant_level_dofs(dof_handler, level);
    const IndexSet locally_owned_level_dofs = dof_handler.locally_owned_mg_dofs(level);

    mg_constraints[level].clear();
    mg_constraints[level].reinit(locally_owned_level_dofs, relevant_dofs);
    // Ensure we only add indices that are relevant to this process
    mg_constraints[level].add_lines(mg_constrained_dofs.get_boundary_indices(level));
    mg_constraints[level].close();

    DynamicSparsityPattern level_dsp(relevant_dofs);
    MGTools::make_sparsity_pattern(dof_handler, level_dsp, level, mg_constraints[level]);
    SparsityTools::distribute_sparsity_pattern(level_dsp,
                                               locally_owned_level_dofs,
                                               mpi_communicator,
                                               relevant_dofs);

    // Important: SparseMatrix reinit for Trilinos in MPI
    mg_matrices[level].reinit(locally_owned_level_dofs,
                              locally_owned_level_dofs,
                              level_dsp,
                              mpi_communicator);
  }
}

template <int dim, int fe_degree>
void ADRPureMatrixSolverMPI<dim, fe_degree>::compute_errors(double &L2_error, double &H1_error) const
{
  TrilinosWrappers::MPI::Vector solution_with_ghosts(locally_owned_dofs,
                                                      locally_relevant_dofs,
                                                      mpi_communicator);
  solution_with_ghosts = solution;

  Vector<float> difference_per_cell(triangulation.n_active_cells());

  VectorTools::integrate_difference(dof_handler,
                                    solution_with_ghosts,
                                    exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::L2_norm);
  L2_error = VectorTools::compute_global_error(triangulation,
                                               difference_per_cell,
                                               VectorTools::L2_norm);

  VectorTools::integrate_difference(dof_handler,
                                    solution_with_ghosts,
                                    exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(fe.degree + 2),
                                    VectorTools::H1_norm);
  H1_error = VectorTools::compute_global_error(triangulation,
                                               difference_per_cell,
                                               VectorTools::H1_norm);
}


// Benchmark result structure
struct BenchmarkResult
{
  std::string run_mode;
  unsigned int amortized_solves;
  std::string solver_type;
  unsigned int fe_degree;
  unsigned int refinement;
  types::global_dof_index n_dofs;
  unsigned int n_cells;
  unsigned int n_procs;
  unsigned int n_mg_levels;

  double setup_time;
  double assemble_time;
  double solve_time;
  double total_time;
  double solve_time_batch;
  double total_time_batch;

  double setup_time_avg;
  double assemble_time_avg;
  double solve_time_avg;
  double total_time_avg;

  double memory_before_mb;       // RSS sum across ranks
  double memory_after_mb;        // RSS sum across ranks
  double memory_used_mb;         // RSS delta sum across ranks
  double memory_after_mb_max;    // RSS max across ranks
  double memory_after_mb_avg;    // RSS avg across ranks
  double memory_hwm_mb_max;      // HWM max across ranks
  double memory_hwm_mb_avg;      // HWM avg across ranks
  double memory_vmdata_used_mb;  // VmData delta sum across ranks
  double memory_heap_used_mb;    // heap alloc delta sum across ranks
  double memory_heap_after_max_mb;
  double memory_heap_after_avg_mb;
  double memory_heap_reserved_after_max_mb;
  double memory_heap_reserved_after_avg_mb;
  double memory_occupancy_pct_avg;
  double memory_occupancy_pct_max;
  double memory_estimate_ratio;
  double memory_estimate_ratio_incremental;
  double memory_actual_per_dof_bytes;
  double memory_incremental_per_dof_bytes;

  double local_dofs_min;
  double local_dofs_avg;
  double local_dofs_max;
  double local_cells_min;
  double local_cells_avg;
  double local_cells_max;

  double memory_estimated_mb;

  double L2_error;
  double H1_error;

  double time_per_dof_us;
  double memory_per_dof_bytes;
  unsigned int solve_iterations;
  double solve_iterations_avg;
  unsigned int solve_converged_ranks;
  double preconditioner_setup_time;
  double preconditioner_setup_time_avg;
  double krylov_time;
  double krylov_time_avg;
  double time_per_iteration_ms;
};


template <int fe_degree>
BenchmarkResult run_matrix_free_benchmark(MPI_Comm mpi_comm,
                                          unsigned int n_refinements,
                                          unsigned int fixed_iterations,
                                          unsigned int amortized_solves)
{
  BenchmarkResult result;
  const bool amortized = (amortized_solves > 1);
  result.run_mode = amortized ? ((fixed_iterations > 0) ? "amortized_fixed_iter" : "amortized_end")
                              : ((fixed_iterations > 0) ? "fixed_iter" : "end_to_end");
  result.amortized_solves = amortized_solves;
  result.solver_type = "Matrix-Free";
  result.fe_degree = fe_degree;
  result.refinement = n_refinements;
  result.n_procs = Utilities::MPI::n_mpi_processes(mpi_comm);

  MPI_Barrier(mpi_comm);
  const MemorySnapshot mem_before = MemoryInfo::snapshot();
  const double rss_before_sum = mpi_sum(mem_before.rss_mb, mpi_comm);
  const double vmdata_before_sum = mpi_sum(mem_before.vmdata_mb, mpi_comm);
  const double heap_before_sum = mpi_sum(mem_before.heap_alloc_mb, mpi_comm);

  ExactSolutionMixed<dim> exact_solution;
  RightHandSideMixed<dim> right_hand_side;
  DiffusionCoefficientMixed<dim> diffusion;
  AdvectionFieldMixed<dim> advection;
  ReactionCoefficientMixed<dim> reaction;
  NeumannBCMixed<dim> neumann_bc;

  std::set<types::boundary_id> dirichlet_ids = {0, 2};
  std::set<types::boundary_id> neumann_ids = {1, 3};

  ADRMatrixFreeSolverMPI<dim, fe_degree> solver(
    mpi_comm, exact_solution, right_hand_side, diffusion, advection,
    reaction, neumann_bc, dirichlet_ids, neumann_ids);

  Timer timer;

  MPI_Barrier(mpi_comm);
  timer.start();
  solver.make_grid(n_refinements);
  solver.setup_system();
  const double setup_local = timer.wall_time();
  const auto setup_stats = mpi_stats(setup_local, mpi_comm);
  result.setup_time = setup_stats.max;
  result.setup_time_avg = setup_stats.avg;
  result.n_dofs = solver.n_dofs();
  result.n_cells = solver.n_cells();
  result.n_mg_levels = n_refinements + 1;
  const double local_dofs = static_cast<double>(solver.n_locally_owned_dofs());
  const double local_cells = static_cast<double>(solver.n_locally_owned_cells());
  const auto local_dof_stats = mpi_stats(local_dofs, mpi_comm);
  const auto local_cell_stats = mpi_stats(local_cells, mpi_comm);
  result.local_dofs_min = local_dof_stats.min;
  result.local_dofs_avg = local_dof_stats.avg;
  result.local_dofs_max = local_dof_stats.max;
  result.local_cells_min = local_cell_stats.min;
  result.local_cells_avg = local_cell_stats.avg;
  result.local_cells_max = local_cell_stats.max;

  MPI_Barrier(mpi_comm);
  timer.restart();
  solver.assemble_rhs();
  const double assemble_local = timer.wall_time();
  const auto assemble_stats = mpi_stats(assemble_local, mpi_comm);
  result.assemble_time = assemble_stats.max;
  result.assemble_time_avg = assemble_stats.avg;

  double solve_local_batch = 0.0;
  double iteration_local_sum = 0.0;
  double precond_local_sum = 0.0;
  double krylov_local_sum = 0.0;
  double converged_local_sum = 0.0;
  for (unsigned int solve_idx = 0; solve_idx < amortized_solves; ++solve_idx)
  {
    MPI_Barrier(mpi_comm);
    timer.restart();
    solver.solve(fixed_iterations);
    solve_local_batch += timer.wall_time();
    const auto &solve_diag = solver.get_solve_diagnostics();
    iteration_local_sum += solve_diag.iterations;
    precond_local_sum += solve_diag.preconditioner_setup_time;
    krylov_local_sum += solve_diag.krylov_time;
    converged_local_sum += solve_diag.converged ? 1.0 : 0.0;
  }
  const double solve_local = solve_local_batch / amortized_solves;
  const auto solve_stats = mpi_stats(solve_local, mpi_comm);
  const auto solve_batch_stats = mpi_stats(solve_local_batch, mpi_comm);
  result.solve_time = solve_stats.max;
  result.solve_time_avg = solve_stats.avg;
  result.solve_time_batch = solve_batch_stats.max;

  const double iteration_local_avg = iteration_local_sum / amortized_solves;
  const double precond_local_avg = precond_local_sum / amortized_solves;
  const double krylov_local_avg = krylov_local_sum / amortized_solves;
  const auto iteration_stats = mpi_stats(iteration_local_avg, mpi_comm);
  const auto precond_setup_stats = mpi_stats(precond_local_avg, mpi_comm);
  const auto krylov_stats = mpi_stats(krylov_local_avg, mpi_comm);
  result.solve_iterations = static_cast<unsigned int>(std::round(iteration_stats.max));
  result.solve_iterations_avg = iteration_stats.avg;
  result.preconditioner_setup_time = precond_setup_stats.max;
  result.preconditioner_setup_time_avg = precond_setup_stats.avg;
  result.krylov_time = krylov_stats.max;
  result.krylov_time_avg = krylov_stats.avg;
  result.solve_converged_ranks = static_cast<unsigned int>(std::round(
    mpi_sum(converged_local_sum / amortized_solves, mpi_comm)));

  const double total_local_batch = setup_local + assemble_local + solve_local_batch;
  const double total_local = total_local_batch / amortized_solves;
  const auto total_stats = mpi_stats(total_local, mpi_comm);
  const auto total_batch_stats = mpi_stats(total_local_batch, mpi_comm);
  result.total_time = total_stats.max;
  result.total_time_avg = total_stats.avg;
  result.total_time_batch = total_batch_stats.max;

  MPI_Barrier(mpi_comm);
  const MemorySnapshot mem_after = MemoryInfo::snapshot();
  const double rss_after_sum = mpi_sum(mem_after.rss_mb, mpi_comm);
  const double vmdata_after_sum = mpi_sum(mem_after.vmdata_mb, mpi_comm);
  const double heap_after_sum = mpi_sum(mem_after.heap_alloc_mb, mpi_comm);
  const auto rss_after_stats = mpi_stats(mem_after.rss_mb, mpi_comm);
  const auto hwm_stats = mpi_stats(mem_after.hwm_mb, mpi_comm);
  const auto heap_stats = mpi_stats(mem_after.heap_alloc_mb, mpi_comm);
  const auto heap_reserved_stats = mpi_stats(mem_after.heap_reserved_mb, mpi_comm);
  const double occupancy_pct =
    (mem_after.memtotal_mb > 0.0) ? (mem_after.rss_mb / mem_after.memtotal_mb * 100.0) : 0.0;
  const auto occupancy_stats = mpi_stats(occupancy_pct, mpi_comm);

  result.memory_before_mb = rss_before_sum;
  result.memory_after_mb = rss_after_sum;
  result.memory_used_mb = rss_after_sum - rss_before_sum;
  result.memory_after_mb_max = rss_after_stats.max;
  result.memory_after_mb_avg = rss_after_stats.avg;
  result.memory_hwm_mb_max = hwm_stats.max;
  result.memory_hwm_mb_avg = hwm_stats.avg;
  result.memory_vmdata_used_mb = vmdata_after_sum - vmdata_before_sum;
  result.memory_heap_used_mb = heap_after_sum - heap_before_sum;
  result.memory_heap_after_max_mb = heap_stats.max;
  result.memory_heap_after_avg_mb = heap_stats.avg;
  result.memory_heap_reserved_after_max_mb = heap_reserved_stats.max;
  result.memory_heap_reserved_after_avg_mb = heap_reserved_stats.avg;
  result.memory_occupancy_pct_avg = occupancy_stats.avg;
  result.memory_occupancy_pct_max = occupancy_stats.max;

  result.memory_estimated_mb = MemoryEstimate::matrix_free_mb(
    result.n_dofs, result.n_cells, fe_degree);
  result.memory_estimate_ratio =
    (result.memory_estimated_mb > 0.0) ? (result.memory_after_mb / result.memory_estimated_mb) : 0.0;
  result.memory_estimate_ratio_incremental =
    (result.memory_estimated_mb > 0.0) ? (result.memory_used_mb / result.memory_estimated_mb) : 0.0;

  if (fixed_iterations == 0)
    solver.compute_errors(result.L2_error, result.H1_error);
  else
  {
    result.L2_error = std::numeric_limits<double>::quiet_NaN();
    result.H1_error = std::numeric_limits<double>::quiet_NaN();
  }

  result.time_per_dof_us = (result.total_time * 1e6) / result.n_dofs;
  result.memory_per_dof_bytes = (result.memory_estimated_mb * 1024.0 * 1024.0) / result.n_dofs;
  result.memory_actual_per_dof_bytes = (result.memory_after_mb * 1024.0 * 1024.0) / result.n_dofs;
  result.memory_incremental_per_dof_bytes = (result.memory_used_mb * 1024.0 * 1024.0) / result.n_dofs;
  result.time_per_iteration_ms = (result.solve_iterations > 0)
    ? (result.krylov_time * 1e3 / result.solve_iterations)
    : 0.0;

  return result;
}


template <int fe_degree>
BenchmarkResult run_matrix_benchmark(MPI_Comm mpi_comm,
                                     unsigned int n_refinements,
                                     unsigned int fixed_iterations,
                                     const MatrixBasedSolverOptions &solver_options,
                                     unsigned int amortized_solves)
{
  BenchmarkResult result;
  const bool amortized = (amortized_solves > 1);
  result.run_mode = amortized ? ((fixed_iterations > 0) ? "amortized_fixed_iter" : "amortized_end")
                              : ((fixed_iterations > 0) ? "fixed_iter" : "end_to_end");
  result.amortized_solves = amortized_solves;
  result.solver_type = "Matrix-Based";
  result.fe_degree = fe_degree;
  result.refinement = n_refinements;
  result.n_procs = Utilities::MPI::n_mpi_processes(mpi_comm);

  MPI_Barrier(mpi_comm);
  const MemorySnapshot mem_before = MemoryInfo::snapshot();
  const double rss_before_sum = mpi_sum(mem_before.rss_mb, mpi_comm);
  const double vmdata_before_sum = mpi_sum(mem_before.vmdata_mb, mpi_comm);
  const double heap_before_sum = mpi_sum(mem_before.heap_alloc_mb, mpi_comm);

  ExactSolutionMixed<dim> exact_solution;
  RightHandSideMixed<dim> right_hand_side;
  DiffusionCoefficientMixed<dim> diffusion;
  AdvectionFieldMixed<dim> advection;
  ReactionCoefficientMixed<dim> reaction;
  NeumannBCMixed<dim> neumann_bc;

  std::set<types::boundary_id> dirichlet_ids = {0, 2};
  std::set<types::boundary_id> neumann_ids = {1, 3};

  ADRPureMatrixSolverMPI<dim, fe_degree> solver(
    mpi_comm, exact_solution, right_hand_side, diffusion, advection,
    reaction, neumann_bc, dirichlet_ids, neumann_ids, solver_options);

  Timer timer;

  MPI_Barrier(mpi_comm);
  timer.start();
  solver.make_grid(n_refinements);
  solver.setup_system();
  const double setup_local = timer.wall_time();
  const auto setup_stats = mpi_stats(setup_local, mpi_comm);
  result.setup_time = setup_stats.max;
  result.setup_time_avg = setup_stats.avg;
  result.n_dofs = solver.n_dofs();
  result.n_cells = solver.n_cells();
  result.n_mg_levels = n_refinements + 1;
  const double local_dofs = static_cast<double>(solver.n_locally_owned_dofs());
  const double local_cells = static_cast<double>(solver.n_locally_owned_cells());
  const auto local_dof_stats = mpi_stats(local_dofs, mpi_comm);
  const auto local_cell_stats = mpi_stats(local_cells, mpi_comm);
  result.local_dofs_min = local_dof_stats.min;
  result.local_dofs_avg = local_dof_stats.avg;
  result.local_dofs_max = local_dof_stats.max;
  result.local_cells_min = local_cell_stats.min;
  result.local_cells_avg = local_cell_stats.avg;
  result.local_cells_max = local_cell_stats.max;

  MPI_Barrier(mpi_comm);
  timer.restart();
  solver.assemble_system();
  solver.assemble_multigrid_matrices();  // Assemble ALL MG level matrices!
  const double assemble_local = timer.wall_time();
  const auto assemble_stats = mpi_stats(assemble_local, mpi_comm);
  result.assemble_time = assemble_stats.max;
  result.assemble_time_avg = assemble_stats.avg;

  double solve_local_batch = 0.0;
  double iteration_local_sum = 0.0;
  double precond_local_sum = 0.0;
  double krylov_local_sum = 0.0;
  double converged_local_sum = 0.0;
  for (unsigned int solve_idx = 0; solve_idx < amortized_solves; ++solve_idx)
  {
    MPI_Barrier(mpi_comm);
    timer.restart();
    solver.solve(fixed_iterations);
    solve_local_batch += timer.wall_time();
    const auto &solve_diag = solver.get_solve_diagnostics();
    iteration_local_sum += solve_diag.iterations;
    precond_local_sum += solve_diag.preconditioner_setup_time;
    krylov_local_sum += solve_diag.krylov_time;
    converged_local_sum += solve_diag.converged ? 1.0 : 0.0;
  }
  const double solve_local = solve_local_batch / amortized_solves;
  const auto solve_stats = mpi_stats(solve_local, mpi_comm);
  const auto solve_batch_stats = mpi_stats(solve_local_batch, mpi_comm);
  result.solve_time = solve_stats.max;
  result.solve_time_avg = solve_stats.avg;
  result.solve_time_batch = solve_batch_stats.max;

  const double iteration_local_avg = iteration_local_sum / amortized_solves;
  const double precond_local_avg = precond_local_sum / amortized_solves;
  const double krylov_local_avg = krylov_local_sum / amortized_solves;
  const auto iteration_stats = mpi_stats(iteration_local_avg, mpi_comm);
  const auto precond_setup_stats = mpi_stats(precond_local_avg, mpi_comm);
  const auto krylov_stats = mpi_stats(krylov_local_avg, mpi_comm);
  result.solve_iterations = static_cast<unsigned int>(std::round(iteration_stats.max));
  result.solve_iterations_avg = iteration_stats.avg;
  result.preconditioner_setup_time = precond_setup_stats.max;
  result.preconditioner_setup_time_avg = precond_setup_stats.avg;
  result.krylov_time = krylov_stats.max;
  result.krylov_time_avg = krylov_stats.avg;
  result.solve_converged_ranks = static_cast<unsigned int>(std::round(
    mpi_sum(converged_local_sum / amortized_solves, mpi_comm)));

  const double total_local_batch = setup_local + assemble_local + solve_local_batch;
  const double total_local = total_local_batch / amortized_solves;
  const auto total_stats = mpi_stats(total_local, mpi_comm);
  const auto total_batch_stats = mpi_stats(total_local_batch, mpi_comm);
  result.total_time = total_stats.max;
  result.total_time_avg = total_stats.avg;
  result.total_time_batch = total_batch_stats.max;

  MPI_Barrier(mpi_comm);
  const MemorySnapshot mem_after = MemoryInfo::snapshot();
  const double rss_after_sum = mpi_sum(mem_after.rss_mb, mpi_comm);
  const double vmdata_after_sum = mpi_sum(mem_after.vmdata_mb, mpi_comm);
  const double heap_after_sum = mpi_sum(mem_after.heap_alloc_mb, mpi_comm);
  const auto rss_after_stats = mpi_stats(mem_after.rss_mb, mpi_comm);
  const auto hwm_stats = mpi_stats(mem_after.hwm_mb, mpi_comm);
  const auto heap_stats = mpi_stats(mem_after.heap_alloc_mb, mpi_comm);
  const auto heap_reserved_stats = mpi_stats(mem_after.heap_reserved_mb, mpi_comm);
  const double occupancy_pct =
    (mem_after.memtotal_mb > 0.0) ? (mem_after.rss_mb / mem_after.memtotal_mb * 100.0) : 0.0;
  const auto occupancy_stats = mpi_stats(occupancy_pct, mpi_comm);

  result.memory_before_mb = rss_before_sum;
  result.memory_after_mb = rss_after_sum;
  result.memory_used_mb = rss_after_sum - rss_before_sum;
  result.memory_after_mb_max = rss_after_stats.max;
  result.memory_after_mb_avg = rss_after_stats.avg;
  result.memory_hwm_mb_max = hwm_stats.max;
  result.memory_hwm_mb_avg = hwm_stats.avg;
  result.memory_vmdata_used_mb = vmdata_after_sum - vmdata_before_sum;
  result.memory_heap_used_mb = heap_after_sum - heap_before_sum;
  result.memory_heap_after_max_mb = heap_stats.max;
  result.memory_heap_after_avg_mb = heap_stats.avg;
  result.memory_heap_reserved_after_max_mb = heap_reserved_stats.max;
  result.memory_heap_reserved_after_avg_mb = heap_reserved_stats.avg;
  result.memory_occupancy_pct_avg = occupancy_stats.avg;
  result.memory_occupancy_pct_max = occupancy_stats.max;

  result.memory_estimated_mb = MemoryEstimate::matrix_based_mb(
    result.n_dofs, fe_degree, result.n_mg_levels);
  result.memory_estimate_ratio =
    (result.memory_estimated_mb > 0.0) ? (result.memory_after_mb / result.memory_estimated_mb) : 0.0;
  result.memory_estimate_ratio_incremental =
    (result.memory_estimated_mb > 0.0) ? (result.memory_used_mb / result.memory_estimated_mb) : 0.0;

  if (fixed_iterations == 0)
    solver.compute_errors(result.L2_error, result.H1_error);
  else
  {
    result.L2_error = std::numeric_limits<double>::quiet_NaN();
    result.H1_error = std::numeric_limits<double>::quiet_NaN();
  }

  result.time_per_dof_us = (result.total_time * 1e6) / result.n_dofs;
  result.memory_per_dof_bytes = (result.memory_estimated_mb * 1024.0 * 1024.0) / result.n_dofs;
  result.memory_actual_per_dof_bytes = (result.memory_after_mb * 1024.0 * 1024.0) / result.n_dofs;
  result.memory_incremental_per_dof_bytes = (result.memory_used_mb * 1024.0 * 1024.0) / result.n_dofs;
  result.time_per_iteration_ms = (result.solve_iterations > 0)
    ? (result.krylov_time * 1e3 / result.solve_iterations)
    : 0.0;

  return result;
}


void print_comparison_table(const std::vector<BenchmarkResult> &mf_results,
                           const std::vector<BenchmarkResult> &mat_results,
                           ConditionalOStream &pcout)
{
  pcout << "\n" << std::string(160, '=') << "\n";
  pcout << "PURE COMPARISON: MATRIX-FREE (no precomp) vs MATRIX-BASED (all levels)\n";
  pcout << std::string(160, '=') << "\n\n";

  pcout << std::setw(4) << "P"
        << std::setw(5) << "Ref"
        << std::setw(10) << "DOFs"
        << std::setw(6) << "Proc"
        << std::setw(5) << "MG-L"
        << std::setw(10) << "MF_Time"
        << std::setw(10) << "Mat_Time"
        << std::setw(9) << "Speedup"
        << std::setw(10) << "MF_RSS"
        << std::setw(10) << "Mat_RSS"
        << std::setw(9) << "MemRatio"
        << std::setw(12) << "MF_L2"
        << std::setw(12) << "Mat_L2"
        << "\n";
  pcout << std::string(160, '-') << "\n";

  for (size_t i = 0; i < mf_results.size() && i < mat_results.size(); ++i)
  {
    const auto &mf = mf_results[i];
    const auto &mat = mat_results[i];

    if (mat.total_time <= 0) continue;

    double speedup = mat.total_time / mf.total_time;
    // Compare measured memory instead of analytical estimates.
    double mem_ratio = mat.memory_after_mb_max / std::max(mf.memory_after_mb_max, 0.01);

    pcout << std::fixed << std::setprecision(2);
    pcout << std::setw(4) << mf.fe_degree
          << std::setw(5) << mf.refinement
          << std::setw(10) << mf.n_dofs
          << std::setw(6) << mf.n_procs
          << std::setw(5) << mf.n_mg_levels
          << std::setw(10) << std::setprecision(4) << mf.total_time
          << std::setw(10) << mat.total_time
          << std::setw(8) << std::setprecision(2) << speedup << "x"
          << std::setw(10) << std::setprecision(1) << mf.memory_after_mb_max
          << std::setw(10) << mat.memory_after_mb_max
          << std::setw(8) << std::setprecision(1) << mem_ratio << "x"
          << std::setw(12) << std::scientific << std::setprecision(2) << mf.L2_error
          << std::setw(12) << mat.L2_error
          << "\n";
  }
}


void write_csv(const std::vector<BenchmarkResult> &results,
               const std::string &filename,
               unsigned int this_mpi_process)
{
  if (this_mpi_process != 0)
    return;

  std::ofstream out(filename);
  out << "run_mode,solver_type,degree,refinement,dofs,cells,n_procs,n_mg_levels,"
      << "amortized_solves,"
      << "setup_time,assemble_time,solve_time,total_time,solve_time_batch,total_time_batch,"
      << "setup_time_avg,assemble_time_avg,solve_time_avg,total_time_avg,"
      << "solve_iterations,solve_iterations_avg,solve_converged_ranks,"
      << "preconditioner_setup_time,preconditioner_setup_time_avg,"
      << "krylov_time,krylov_time_avg,time_per_iteration_ms,"
      << "memory_used_mb,memory_estimated_mb,memory_per_dof_bytes,"
      << "memory_rss_after_max_mb,memory_rss_after_avg_mb,"
      << "memory_hwm_max_mb,memory_hwm_avg_mb,"
      << "memory_vmdata_used_mb,memory_heap_used_mb,"
      << "memory_heap_after_max_mb,memory_heap_after_avg_mb,"
      << "memory_heap_reserved_after_max_mb,memory_heap_reserved_after_avg_mb,"
      << "memory_occupancy_pct_avg,memory_occupancy_pct_max,"
      << "memory_estimate_ratio,memory_estimate_ratio_incremental,"
      << "memory_actual_per_dof_bytes,memory_incremental_per_dof_bytes,"
      << "local_dofs_min,local_dofs_avg,local_dofs_max,"
      << "local_cells_min,local_cells_avg,local_cells_max,"
      << "L2_error,H1_error,time_per_dof_us\n";

  for (const auto &r : results)
  {
    out << r.run_mode << ","
        << r.solver_type << ","
        << r.fe_degree << ","
        << r.refinement << ","
        << r.n_dofs << ","
        << r.n_cells << ","
        << r.n_procs << ","
        << r.n_mg_levels << ","
        << r.amortized_solves << ","
        << r.setup_time << ","
        << r.assemble_time << ","
        << r.solve_time << ","
        << r.total_time << ","
        << r.solve_time_batch << ","
        << r.total_time_batch << ","
        << r.setup_time_avg << ","
        << r.assemble_time_avg << ","
        << r.solve_time_avg << ","
        << r.total_time_avg << ","
        << r.solve_iterations << ","
        << r.solve_iterations_avg << ","
        << r.solve_converged_ranks << ","
        << r.preconditioner_setup_time << ","
        << r.preconditioner_setup_time_avg << ","
        << r.krylov_time << ","
        << r.krylov_time_avg << ","
        << r.time_per_iteration_ms << ","
        << r.memory_used_mb << ","
        << r.memory_estimated_mb << ","
        << r.memory_per_dof_bytes << ","
        << r.memory_after_mb_max << ","
        << r.memory_after_mb_avg << ","
        << r.memory_hwm_mb_max << ","
        << r.memory_hwm_mb_avg << ","
        << r.memory_vmdata_used_mb << ","
        << r.memory_heap_used_mb << ","
        << r.memory_heap_after_max_mb << ","
        << r.memory_heap_after_avg_mb << ","
        << r.memory_heap_reserved_after_max_mb << ","
        << r.memory_heap_reserved_after_avg_mb << ","
        << r.memory_occupancy_pct_avg << ","
        << r.memory_occupancy_pct_max << ","
        << r.memory_estimate_ratio << ","
        << r.memory_estimate_ratio_incremental << ","
        << r.memory_actual_per_dof_bytes << ","
        << r.memory_incremental_per_dof_bytes << ","
        << r.local_dofs_min << ","
        << r.local_dofs_avg << ","
        << r.local_dofs_max << ","
        << r.local_cells_min << ","
        << r.local_cells_avg << ","
        << r.local_cells_max << ","
        << r.L2_error << ","
        << r.H1_error << ","
        << r.time_per_dof_us << "\n";
  }
}


template <int fe_degree>
void run_degree_benchmark(MPI_Comm mpi_comm,
                          unsigned int min_ref,
                          unsigned int max_ref,
                          std::vector<BenchmarkResult> &mf_results,
                          std::vector<BenchmarkResult> &mat_results,
                          ConditionalOStream &pcout,
                          bool skip_matrix_large,
                          bool only_mf,
                          bool only_mat,
                          unsigned int fixed_iterations,
                          unsigned int amortized_solves,
                          const MatrixBasedSolverOptions &solver_options)
{
  pcout << "\n" << std::string(70, '#') << "\n";
  pcout << "# Benchmarking P" << fe_degree << " elements (ref " << min_ref << "-" << max_ref << ")\n";
  pcout << std::string(70, '#') << "\n";

  for (unsigned int ref = min_ref; ref <= max_ref; ++ref)
  {
    pcout << "\n--- Refinement " << ref << " ---\n";

    // Matrix-free benchmark
    BenchmarkResult mf_result;
    if (!only_mat)
    {
      pcout << "  Running matrix-free (no precomp)... " << std::flush;
      mf_result = run_matrix_free_benchmark<fe_degree>(mpi_comm, ref, fixed_iterations, amortized_solves);
      pcout << "done (" << std::fixed << std::setprecision(2)
            << mf_result.total_time << "s, "
            << "rss_max=" << mf_result.memory_after_mb_max << " MB, "
            << "heap_max=" << mf_result.memory_heap_after_max_mb << " MB, "
            << mf_result.solve_iterations << " iters)\n";
      mf_results.push_back(mf_result);
    }

    // Matrix-based benchmark
    bool run_matrix = !only_mf;
    if (run_matrix && !only_mat)
    {
      const types::global_dof_index estimated_dofs = mf_result.n_dofs;
      run_matrix = !skip_matrix_large || estimated_dofs < 300000;
    }

    if (run_matrix)
    {
      pcout << "  Running matrix-based (all levels)... " << std::flush;
      auto mat_result = run_matrix_benchmark<fe_degree>(mpi_comm,
                                                        ref,
                                                        fixed_iterations,
                                                        solver_options,
                                                        amortized_solves);
      pcout << "done (" << std::fixed << std::setprecision(2)
            << mat_result.total_time << "s, "
            << "rss_max=" << mat_result.memory_after_mb_max << " MB, "
            << "heap_max=" << mat_result.memory_heap_after_max_mb << " MB, "
            << mat_result.solve_iterations << " iters)\n";
      mat_results.push_back(mat_result);

      if (!only_mat && !only_mf)
      {
        double speedup = mat_result.total_time / mf_result.total_time;
        double mem_ratio =
          mat_result.memory_after_mb_max / std::max(mf_result.memory_after_mb_max, 0.01);
        pcout << "  Speedup: " << std::setprecision(2) << speedup << "x"
              << ", RSS(max-rank) ratio: " << mem_ratio << "x\n";
      }
    }
    else
    {
      if (only_mf)
        pcout << "  Skipping matrix-based (--only-mf)\n";
      else if (skip_matrix_large)
        pcout << "  Skipping matrix-based (problem too large)\n";
      else
        pcout << "  Skipping matrix-based (disabled)\n";
    }
  }
}


int main(int argc, char *argv[])
{
  try
  {
#if defined(FORCE_ONLY_MF) && defined(FORCE_ONLY_MAT)
#  error "FORCE_ONLY_MF and FORCE_ONLY_MAT cannot both be defined."
#endif

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    const unsigned int this_mpi_process = Utilities::MPI::this_mpi_process(mpi_comm);
    const unsigned int n_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

    ConditionalOStream pcout(std::cout, this_mpi_process == 0);

    unsigned int min_ref = 4;
    unsigned int max_ref = 7;
    int single_degree = -1;
    bool skip_matrix_large = false;
    bool only_mf = false;
    bool only_mat = false;
#if defined(FORCE_ONLY_MF)
    only_mf = true;
#elif defined(FORCE_ONLY_MAT)
    only_mat = true;
#endif
    unsigned int fixed_iterations = 0;
    unsigned int amortized_solves = 1;
    MatrixBasedSolverOptions matrix_options;
    std::string csv_file = "benchmark_pure_mpi.csv";

    for (int i = 1; i < argc; ++i)
    {
      std::string arg = argv[i];
      if (arg == "--min-ref" && i + 1 < argc)
        min_ref = std::stoi(argv[++i]);
      else if (arg == "--max-ref" && i + 1 < argc)
        max_ref = std::stoi(argv[++i]);
      else if (arg == "--degree" && i + 1 < argc)
        single_degree = std::stoi(argv[++i]);
      else if (arg == "--all-matrix")
        skip_matrix_large = false;
      else if (arg == "--skip-matrix-large")
        skip_matrix_large = true;
      else if (arg == "--only-mf")
        only_mf = true;
      else if (arg == "--only-mat")
        only_mat = true;
      else if (arg == "--fixed-iters" && i + 1 < argc)
        fixed_iterations = static_cast<unsigned int>(std::stoul(argv[++i]));
      else if (arg == "--amortize-solves" && i + 1 < argc)
        amortized_solves = std::max(1u, static_cast<unsigned int>(std::stoul(argv[++i])));
      else if (arg == "--mat-cheb-degree" && i + 1 < argc)
        matrix_options.chebyshev_degree = static_cast<unsigned int>(std::stoul(argv[++i]));
      else if (arg == "--mat-cheb-range" && i + 1 < argc)
        matrix_options.chebyshev_smoothing_range = std::stod(argv[++i]);
      else if (arg == "--mat-cheb-eig-iters" && i + 1 < argc)
        matrix_options.chebyshev_eig_iterations = static_cast<unsigned int>(std::stoul(argv[++i]));
      else if (arg == "--mat-jacobi-omega" && i + 1 < argc)
        matrix_options.jacobi_omega = std::stod(argv[++i]);
      else if (arg == "--mat-coarse-degree" && i + 1 < argc)
        matrix_options.coarse_chebyshev_degree = static_cast<unsigned int>(std::stoul(argv[++i]));
      else if (arg == "--csv" && i + 1 < argc)
        csv_file = argv[++i];
      else if (arg == "--help")
      {
        pcout << "Usage: mpirun -np N " << argv[0] << " [options]\n"
              << "Options:\n"
              << "  --min-ref N      Minimum refinement (default: 4)\n"
              << "  --max-ref N      Maximum refinement (default: 7)\n"
              << "  --degree N       Run only degree N (2, 3, 4, or 5)\n"
              << "  --all-matrix     Alias for --skip-matrix-large=off (default behavior)\n"
              << "  --skip-matrix-large  Skip matrix-based runs beyond internal DoF threshold\n"
              << "  --only-mf        Run only matrix-free solver\n"
              << "  --only-mat       Run only matrix-based solver\n"
              << "  --fixed-iters N  Force exactly N GMRES iterations (backend comparison mode)\n"
              << "  --amortize-solves N  Reuse setup+assembly and solve N times; report per-solve amortized timings\n"
              << "  --mat-cheb-degree N      Matrix-based Chebyshev degree on non-coarsest levels (default: 5)\n"
              << "  --mat-cheb-range R       Matrix-based Chebyshev smoothing range (default: 15)\n"
              << "  --mat-cheb-eig-iters N   Matrix-based Chebyshev eigenvalue iterations (default: 10)\n"
              << "  --mat-jacobi-omega W     Matrix-based Jacobi omega (default: 1.0)\n"
              << "  --mat-coarse-degree N    Matrix-based coarsest-level Chebyshev degree (default: 12)\n"
              << "  --csv FILE       Output CSV file\n"
              << "  --help           Show this help\n";
        return 0;
      }
    }

    if (only_mf && only_mat)
    {
      pcout << "Error: --only-mf and --only-mat are mutually exclusive.\n";
      return 1;
    }

#if defined(FORCE_ONLY_MF)
    if (only_mat)
    {
      pcout << "Error: this executable is matrix-free only.\n";
      return 1;
    }
    only_mf = true;
    only_mat = false;
#elif defined(FORCE_ONLY_MAT)
    if (only_mf)
    {
      pcout << "Error: this executable is matrix-based only.\n";
      return 1;
    }
    only_mf = false;
    only_mat = true;
#endif

    pcout << "\n" << std::string(80, '=') << "\n";
    pcout << "PURE BENCHMARK: MATRIX-FREE (no precomp) vs MATRIX-BASED (all levels)\n";
    pcout << std::string(80, '=') << "\n";
    pcout << "MPI processes: " << n_mpi_processes << "\n";
    pcout << "Refinement range: " << min_ref << " - " << max_ref << "\n";
    if (fixed_iterations > 0)
      pcout << "Mode: fixed-iteration backend comparison (" << fixed_iterations << " iterations)\n";
    else
      pcout << "Mode: end-to-end convergence solve\n";
    if (amortized_solves > 1)
      pcout << "Amortized solves: " << amortized_solves << " (reported times are per solve)\n";
#if defined(FORCE_ONLY_MF)
    pcout << "Binary mode: matrix-free only\n";
#elif defined(FORCE_ONLY_MAT)
    pcout << "Binary mode: matrix-based only\n";
#else
    pcout << "Binary mode: combined (both solvers available)\n";
#endif
    pcout << "Matrix-based smoother: cheb_degree=" << matrix_options.chebyshev_degree
          << ", coarse_degree=" << matrix_options.coarse_chebyshev_degree
          << ", range=" << matrix_options.chebyshev_smoothing_range
          << ", eig_iters=" << matrix_options.chebyshev_eig_iterations
          << ", jacobi_omega=" << matrix_options.jacobi_omega << "\n";

    std::vector<BenchmarkResult> mf_results_p2, mat_results_p2;
    std::vector<BenchmarkResult> mf_results_p3, mat_results_p3;
    std::vector<BenchmarkResult> mf_results_p4, mat_results_p4;

    if (single_degree < 0 || single_degree == 2)
    {
      run_degree_benchmark<2>(mpi_comm, min_ref, std::min(max_ref, 10u),
                              mf_results_p2, mat_results_p2, pcout,
                              skip_matrix_large, only_mf, only_mat,
                              fixed_iterations, amortized_solves, matrix_options);
    }
    if (single_degree < 0 || single_degree == 3)
    {
      run_degree_benchmark<3>(mpi_comm, min_ref, std::min(max_ref, 9u),
                              mf_results_p3, mat_results_p3, pcout,
                              skip_matrix_large, only_mf, only_mat,
                              fixed_iterations, amortized_solves, matrix_options);
    }
    if (single_degree < 0 || single_degree == 4)
    {
      run_degree_benchmark<4>(mpi_comm, min_ref, std::min(max_ref, 8u),
                              mf_results_p4, mat_results_p4, pcout,
                              skip_matrix_large, only_mf, only_mat,
                              fixed_iterations, amortized_solves, matrix_options);
    }

    if (!only_mf && !only_mat)
    {
      if (single_degree < 0 || single_degree == 2)
      {
        pcout << "\n\n--- P2 Elements ---";
        print_comparison_table(mf_results_p2, mat_results_p2, pcout);
      }
      if (single_degree < 0 || single_degree == 3)
      {
        pcout << "\n\n--- P3 Elements ---";
        print_comparison_table(mf_results_p3, mat_results_p3, pcout);
      }
      if (single_degree < 0 || single_degree == 4)
      {
        pcout << "\n\n--- P4 Elements ---";
        print_comparison_table(mf_results_p4, mat_results_p4, pcout);
      }
    }

    std::vector<BenchmarkResult> all_results;
    all_results.insert(all_results.end(), mf_results_p2.begin(), mf_results_p2.end());
    all_results.insert(all_results.end(), mat_results_p2.begin(), mat_results_p2.end());
    all_results.insert(all_results.end(), mf_results_p3.begin(), mf_results_p3.end());
    all_results.insert(all_results.end(), mat_results_p3.begin(), mat_results_p3.end());
    all_results.insert(all_results.end(), mf_results_p4.begin(), mf_results_p4.end());
    all_results.insert(all_results.end(), mat_results_p4.begin(), mat_results_p4.end());

    write_csv(all_results, csv_file, this_mpi_process);
    pcout << "\nResults written to: " << csv_file << "\n";

    return 0;
  }
  catch (std::exception &exc)
  {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return 1;
  }
}
