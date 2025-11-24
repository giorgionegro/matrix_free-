// Matrix-free ADR solver test driver
#include "manufactured_solution.hpp"
#include "adr_matrix_free_solver.hpp"
#include "convergence_test.hpp"

#include <deal.II/base/timer.h>

using namespace dealii;

constexpr unsigned int dim = 2;

template <int fe_degree>
void run_matrix_free_test_impl(const unsigned int n_refinements)
{
  print_convergence_header("Matrix-Free ADR Solver - Single Test");

  std::cout << "FE degree: " << fe_degree << std::endl;
  std::cout << "Refinements: " << n_refinements << std::endl;
  print_separator();

  Triangulation<dim> triangulation;
  create_mesh(triangulation, n_refinements);

  ExactSolution<dim> exact_solution;
  RightHandSide<dim> right_hand_side;
  DiffusionCoefficient<dim> diffusion_coefficient;
  AdvectionField<dim> advection_field;
  ReactionCoefficient<dim> reaction_coefficient;

  ADRMatrixFreeSolver<dim, fe_degree> solver(triangulation,
                                              exact_solution,
                                              right_hand_side,
                                              diffusion_coefficient,
                                              advection_field,
                                              reaction_coefficient);

  Timer timer;
  timer.start();

  solver.setup_system();
  std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
  std::cout << "Number of degrees of freedom: " << solver.n_dofs() << std::endl;
  print_separator();

  const double setup_time = timer.wall_time();

  timer.restart();
  solver.assemble_rhs();
  const double rhs_time = timer.wall_time();

  timer.restart();
  solver.solve();
  const double solve_time = timer.wall_time();

  std::cout << "Setup time (s): " << std::fixed << setup_time << std::endl;
  std::cout << "RHS assembly time (s): " << rhs_time << std::endl;
  std::cout << "Solve time (s): " << solve_time << std::endl;
  std::cout << "Total time (s): " << setup_time + rhs_time + solve_time << std::endl;
  print_separator();

  double L2_error, H1_error;
  solver.compute_errors(L2_error, H1_error);

  std::cout << std::scientific << std::setprecision(4);
  std::cout << "L2 error: " << L2_error << std::endl;
  std::cout << "H1 error: " << H1_error << std::endl;
  print_separator();

  solver.output_results("solution_mf.vtk");
  std::cout << "Output written to solution_mf.vtk" << std::endl;
}

// Runtime dispatch to compile-time degree template (supports degrees 1-4)
void run_matrix_free_test(const unsigned int fe_degree,
                          const unsigned int n_refinements)
{
  switch (fe_degree)
  {
    case 1:
      run_matrix_free_test_impl<1>(n_refinements);
      break;
    case 2:
      run_matrix_free_test_impl<2>(n_refinements);
      break;
    case 3:
      run_matrix_free_test_impl<3>(n_refinements);
      break;
    case 4:
      run_matrix_free_test_impl<4>(n_refinements);
      break;
    default:
      AssertThrow(false,
                  ExcMessage("Unsupported FE degree. Supported degrees: 1, 2, 3, 4"));
  }
}

template <int fe_degree>
void run_h_convergence_mf_impl(const unsigned int min_refinements,
                               const unsigned int max_refinements)
{
  print_convergence_header("h-Convergence Test (Matrix-Free)");

  std::cout << "FE degree: " << fe_degree << std::endl;
  std::cout << "Refinement levels: " << min_refinements
            << " to " << max_refinements << std::endl;
  print_separator();

  ConvergenceTest<dim> convergence;

  ExactSolution<dim> exact_solution;
  RightHandSide<dim> right_hand_side;
  DiffusionCoefficient<dim> diffusion_coefficient;
  AdvectionField<dim> advection_field;
  ReactionCoefficient<dim> reaction_coefficient;

  for (unsigned int refinement = min_refinements;
       refinement <= max_refinements;
       ++refinement)
  {
    std::cout << "\nRefinement level " << refinement << ":" << std::endl;

    Triangulation<dim> triangulation;
    create_mesh(triangulation, refinement);

    ADRMatrixFreeSolver<dim, fe_degree> solver(triangulation,
                                                exact_solution,
                                                right_hand_side,
                                                diffusion_coefficient,
                                                advection_field,
                                                reaction_coefficient);

    Timer timer;
    solver.setup_system();

    std::cout << "  Cells: " << triangulation.n_active_cells()
              << ", DoFs: " << solver.n_dofs() << std::endl;

    timer.start();
    solver.assemble_rhs();
    solver.solve();
    const double total_time = timer.wall_time();

    double L2_error, H1_error;
    solver.compute_errors(L2_error, H1_error);

    std::cout << "  L2 error: " << std::scientific << L2_error
              << ", H1 error: " << H1_error << std::endl;
    std::cout << "  Time: " << std::fixed << total_time << " s" << std::endl;

    const double h = 1.0 / std::sqrt(triangulation.n_active_cells());
    convergence.add_value("cells", triangulation.n_active_cells());
    convergence.add_value("dofs", solver.n_dofs());
    convergence.add_value("h", h);
    convergence.add_value("L2", L2_error);
    convergence.add_value("H1", H1_error);
    convergence.add_value("time", total_time);
  }

  convergence.set_precision("L2", 3);
  convergence.set_precision("H1", 3);
  convergence.set_precision("time", 3);
  convergence.set_scientific("L2", true);
  convergence.set_scientific("H1", true);

  convergence.evaluate_convergence_rates("L2", "cells",
                                        ConvergenceTable::reduction_rate_log2);
  convergence.evaluate_convergence_rates("H1", "cells",
                                        ConvergenceTable::reduction_rate_log2);

  std::cout << "\n";
  print_separator();
  std::cout << "Convergence Table:\n";
  print_separator();
  convergence.write_text(std::cout);
  std::cout << std::endl;

  std::ofstream convergence_file("h_convergence_mf.txt");
  convergence.write_text(convergence_file);
  std::cout << "Convergence data written to h_convergence_mf.txt" << std::endl;
}

// Runtime dispatch for h-convergence test
void run_h_convergence_mf(const unsigned int fe_degree,
                         const unsigned int min_refinements,
                         const unsigned int max_refinements)
{
  switch (fe_degree)
  {
    case 1:
      run_h_convergence_mf_impl<1>(min_refinements, max_refinements);
      break;
    case 2:
      run_h_convergence_mf_impl<2>(min_refinements, max_refinements);
      break;
    case 3:
      run_h_convergence_mf_impl<3>(min_refinements, max_refinements);
      break;
    case 4:
      run_h_convergence_mf_impl<4>(min_refinements, max_refinements);
      break;
    default:
      AssertThrow(false,
                  ExcMessage("Unsupported FE degree. Supported degrees: 1, 2, 3, 4"));
  }
}

int main(int argc, char **argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    std::cout << "================================================================\n";
    std::cout << "Matrix-Free ADR Solver\n";
    std::cout << "================================================================\n";
    //TODO: for mpi we might want to print only from the root process

    if (argc == 1)
    {
      // Default: single test
      run_matrix_free_test(/*fe_degree=*/2, /*refinements=*/4);
    }
    else if (argc == 2 && std::string(argv[1]) == "h")
    {
      // h-convergence test
      run_h_convergence_mf(/*fe_degree=*/2,
                          /*min_refinements=*/3,
                          /*max_refinements=*/6);
    }
    else if (argc == 3)
    {
      // Single test: degree refinements
      const unsigned int fe_degree = std::atoi(argv[1]);
      const unsigned int refinements = std::atoi(argv[2]);
      run_matrix_free_test(fe_degree, refinements);
    }
    else if (argc == 5 && std::string(argv[1]) == "h")
    {
      // h-convergence: h degree min_ref max_ref
      const unsigned int fe_degree = std::atoi(argv[2]);
      const unsigned int min_ref = std::atoi(argv[3]);
      const unsigned int max_ref = std::atoi(argv[4]);
      run_h_convergence_mf(fe_degree, min_ref, max_ref);
    }
    else
    {
      std::cout << "\nUsage:\n";
      std::cout << "  " << argv[0] << "                    # Run default test\n";
      std::cout << "  " << argv[0] << " h                  # Run h-convergence\n";
      std::cout << "  " << argv[0] << " degree refinements # Single test\n";
      std::cout << "  " << argv[0] << " h degree min max   # Custom h-convergence\n";
      std::cout << "\nExamples:\n";
      std::cout << "  " << argv[0] << " 2 4       # Q2, 4 refinements\n";
      std::cout << "  " << argv[0] << " h 2 3 6   # h-convergence: Q2, levels 3-6\n";
      return 0;
    }

    std::cout << "\n================================================================\n";
    std::cout << "Matrix-free test completed successfully!\n";
    std::cout << "================================================================\n";
  }
  catch (std::exception &exc) //Not sure what exception could be thrown here, but better be safe than sorry I guess
  {
    std::cerr << "\n\n"
              << "----------------------------------------------------\n";
    std::cerr << "Exception: " << exc.what() << std::endl;
    std::cerr << "----------------------------------------------------\n";
    return 1;
  }
  catch (...)
  {
    std::cerr << "\n\n"
              << "----------------------------------------------------\n";
    std::cerr << "Unknown exception!" << std::endl;
    std::cerr << "----------------------------------------------------\n";
    return 1;
  }
  return 0;
}

