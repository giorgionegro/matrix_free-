#include "manufactured_solution.hpp"
#include "adr_solver.hpp"
#include "convergence_test.hpp"
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_generator.h>
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace dealii;
constexpr unsigned int dim = 2;

void run_single_test(const unsigned int fe_degree,
                    const unsigned int n_refinements)
{
  print_convergence_header("Single Test Run");
  std::cout << "FE degree: " << fe_degree << std::endl;
  std::cout << "Refinements: " << n_refinements << std::endl;
  print_separator();

  Triangulation<dim> triangulation;
  create_mesh(triangulation, n_refinements);

  ExactSolution<dim>          exact_solution;
  RightHandSide<dim>          right_hand_side;
  DiffusionCoefficient<dim>   diffusion_coefficient;
  AdvectionField<dim>         advection_field;
  ReactionCoefficient<dim>    reaction_coefficient;

  ADRMatrixSolver<dim> solver(triangulation,
                              fe_degree,
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

  solver.assemble_system();
  const double assembly_time = timer.wall_time();
  timer.restart();

  solver.solve();
  const double solve_time = timer.wall_time();
  std::cout << "Assembly time (s): " << assembly_time << std::endl;
  std::cout << "Solve time (s): " << solve_time << std::endl;
  print_separator();

  double L2_error, H1_error;
  solver.compute_errors(L2_error, H1_error);
  std::cout << std::scientific << std::setprecision(4);
  std::cout << "L2 error: " << L2_error << std::endl;
  std::cout << "H1 error: " << H1_error << std::endl;
  print_separator();

  solver.output_results("solution.vtk");
  std::cout << "Output written to solution.vtk" << std::endl;
}

void run_h_convergence(const unsigned int fe_degree,
                      const unsigned int min_refinements,
                      const unsigned int max_refinements)
{
  print_convergence_header("h-Convergence Test (Matrix-based)");
  std::cout << "FE degree: " << fe_degree << std::endl;
  std::cout << "Refinement levels: " << min_refinements 
            << " to " << max_refinements << std::endl;
  print_separator();

  ConvergenceTest<dim> convergence;

  ExactSolution<dim>          exact_solution;
  RightHandSide<dim>          right_hand_side;
  DiffusionCoefficient<dim>   diffusion_coefficient;
  AdvectionField<dim>         advection_field;
  ReactionCoefficient<dim>    reaction_coefficient;

  for (unsigned int refinement = min_refinements;
       refinement <= max_refinements; 
       ++refinement)
  {
    std::cout << "\nRefinement level " << refinement << ":" << std::endl;

    Triangulation<dim> triangulation;
    create_mesh(triangulation, refinement);

    ADRMatrixSolver<dim> solver(triangulation,
                                fe_degree,
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
    solver.assemble_system();
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

  std::ofstream convergence_file("h_convergence.txt");
  convergence.write_text(convergence_file);
  std::cout << "Convergence data written to h_convergence.txt" << std::endl;
}

void run_p_convergence(const unsigned int n_refinements,
                      const unsigned int min_degree,
                      const unsigned int max_degree)
{
  print_convergence_header("p-Convergence Test (Matrix-based)");
  std::cout << "Refinement level: " << n_refinements << std::endl;
  std::cout << "Polynomial degrees: " << min_degree 
            << " to " << max_degree << std::endl;
  print_separator();

  ConvergenceTest<dim> convergence;

  ExactSolution<dim>          exact_solution;
  RightHandSide<dim>          right_hand_side;
  DiffusionCoefficient<dim>   diffusion_coefficient;
  AdvectionField<dim>         advection_field;
  ReactionCoefficient<dim>    reaction_coefficient;

  for (unsigned int degree = min_degree;
       degree <= max_degree; 
       ++degree)
  {
    std::cout << "\nPolynomial degree " << degree << ":" << std::endl;

    Triangulation<dim> triangulation;
    create_mesh(triangulation, n_refinements);

    ADRMatrixSolver<dim> solver(triangulation,
                                degree,
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
    solver.assemble_system();
    solver.solve();
    const double total_time = timer.wall_time();

    double L2_error, H1_error;
    solver.compute_errors(L2_error, H1_error);
    std::cout << "  L2 error: " << std::scientific << L2_error 
              << ", H1 error: " << H1_error << std::endl;
    std::cout << "  Time: " << std::fixed << total_time << " s" << std::endl;

    convergence.add_value("degree", degree);
    convergence.add_value("dofs", solver.n_dofs());
    convergence.add_value("L2", L2_error);
    convergence.add_value("H1", H1_error);
    convergence.add_value("time", total_time);
  }

  convergence.set_precision("L2", 3);
  convergence.set_precision("H1", 3);
  convergence.set_precision("time", 3);
  convergence.set_scientific("L2", true);
  convergence.set_scientific("H1", true);
  convergence.evaluate_convergence_rates("L2", "dofs", 
                                        ConvergenceTable::reduction_rate_log2);
  convergence.evaluate_convergence_rates("H1", "dofs", 
                                        ConvergenceTable::reduction_rate_log2);

  std::cout << "\n";
  print_separator();
  std::cout << "Convergence Table:\n";
  print_separator();
  convergence.write_text(std::cout);
  std::cout << std::endl;

  std::ofstream convergence_file("p_convergence.txt");
  convergence.write_text(convergence_file);
  std::cout << "Convergence data written to p_convergence.txt" << std::endl;
}

int main(int argc, char **argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    std::cout << "================================================================\n";
    std::cout << "Matrix-Based ADR Solver with Convergence Tests\n";
    std::cout << "================================================================\n";

    if (argc == 1)
    {
      run_h_convergence(2, 3, 7);
    }
    else if (argc == 2 && std::string(argv[1]) == "h")
    {
      run_h_convergence(2, 3, 7);
    }
    else if (argc == 2 && std::string(argv[1]) == "p")
    {
      run_p_convergence(4, 1, 6);
    }
    else if (argc == 3)
    {
      const unsigned int fe_degree   = std::atoi(argv[1]);
      const unsigned int refinements = std::atoi(argv[2]);
      run_single_test(fe_degree, refinements);
    }
    else if (argc == 4 && std::string(argv[1]) == "h")
    {
      const unsigned int fe_degree = std::atoi(argv[2]);
      const unsigned int min_ref   = std::atoi(argv[3]);
      const unsigned int max_ref   = min_ref + 4;
      run_h_convergence(fe_degree, min_ref, max_ref);
    }
    else if (argc == 5 && std::string(argv[1]) == "h")
    {
      const unsigned int fe_degree = std::atoi(argv[2]);
      const unsigned int min_ref   = std::atoi(argv[3]);
      const unsigned int max_ref   = std::atoi(argv[4]);
      run_h_convergence(fe_degree, min_ref, max_ref);
    }
    else if (argc == 4 && std::string(argv[1]) == "p")
    {
      const unsigned int n_ref     = std::atoi(argv[2]);
      const unsigned int min_deg   = std::atoi(argv[3]);
      const unsigned int max_deg   = min_deg + 4;
      run_p_convergence(n_ref, min_deg, max_deg);
    }
    else if (argc == 5 && std::string(argv[1]) == "p")
    {
      const unsigned int n_ref     = std::atoi(argv[2]);
      const unsigned int min_deg   = std::atoi(argv[3]);
      const unsigned int max_deg   = std::atoi(argv[4]);
      run_p_convergence(n_ref, min_deg, max_deg);
    }
    else
    {
      std::cout << "\nUsage:\n";
      std::cout << "  " << argv[0] << "                          "
                << "  # Run default h-convergence test\n";
      std::cout << "  " << argv[0] << " h                        "
                << "  # Run h-convergence test\n";
      std::cout << "  " << argv[0] << " p                        "
                << "  # Run p-convergence test\n";
      std::cout << "  " << argv[0] << " degree refinements       "
                << "  # Single test\n";
      std::cout << "  " << argv[0] << " h degree min_ref max_ref "
                << "  # Custom h-convergence\n";
      std::cout << "  " << argv[0] << " p n_ref min_deg max_deg  "
                << "  # Custom p-convergence\n";
      std::cout << "\nExamples:\n";
      std::cout << "  " << argv[0] << " 2 5           "
                << "  # Single test: Q2 elements, 5 refinements\n";
      std::cout << "  " << argv[0] << " h 2 3 7       "
                << "  # h-convergence: Q2, refinements 3-7\n";
      std::cout << "  " << argv[0] << " p 4 1 6       "
                << "  # p-convergence: 4 refinements, Q1-Q6\n";
      return 0;
    }

    std::cout << "\n================================================================\n";
    std::cout << "All tests completed successfully!\n";
    std::cout << "================================================================\n";
  }
  catch (std::exception &exc)
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
