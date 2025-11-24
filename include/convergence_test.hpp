#pragma once

#include <deal.II/base/convergence_table.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <iostream>
#include <iomanip>
#include <fstream>

using namespace dealii;

template <int dim>
class ConvergenceTest
{
public:
  ConvergenceTest() = default;

  void add_value(const std::string &key, const double value)
  {
    table.add_value(key, value);
  }

  void set_precision(const std::string &key, const unsigned int precision)
  {
    table.set_precision(key, precision);
  }

  void set_scientific(const std::string &key, const bool scientific)
  {
    table.set_scientific(key, scientific);
  }

  void evaluate_convergence_rates(const std::string &data_column_key,
                                  const std::string &reference_column_key,
                                  const ConvergenceTable::RateMode rate_mode,
                                  const unsigned int dim_value = dim)
  {
    table.evaluate_convergence_rates(data_column_key,
                                     reference_column_key,
                                     rate_mode,
                                     dim_value);
  }

  void write_text(std::ostream &out) const
  {
    table.write_text(out);
  }

  const ConvergenceTable& get_table() const
  {
    return table;
  }

private:
  ConvergenceTable table;
};

template <int dim>
void create_mesh(Triangulation<dim> &tria,
                const unsigned int n_refinements,
                const double left = 0.0,
                const double right = 1.0)
{
  GridGenerator::hyper_cube(tria, left, right);
  tria.refine_global(n_refinements);
}
inline void print_convergence_header(const std::string &title)
{
  std::cout << "\n";
  std::cout << "================================================================\n";
  std::cout << title << "\n";
  std::cout << "================================================================\n";
}

inline void print_separator()
{
  std::cout << "----------------------------------------------------------------\n";
}

