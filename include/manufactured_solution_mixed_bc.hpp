#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/numbers.h>

using namespace dealii;

/**
 * Manufactured solution classes for mixed boundary conditions.
 *
 * Domain: [0,1]^2
 * Boundary IDs (standard deal.II hyper_cube with colorize=true):
 *   - 0: x = 0 (left)   -> Dirichlet
 *   - 1: x = 1 (right)  -> Neumann
 *   - 2: y = 0 (bottom) -> Dirichlet
 *   - 3: y = 1 (top)    -> Neumann
 *
 * Exact solution: u(x,y) = sin(pi*x/2) * sin(pi*y/2) + 0.5*x*y + 0.1
 */

// Exact solution
template <int dim>
class ExactSolutionMixed : public Function<dim>
{
public:
  double value(const Point<dim> &p,
               const unsigned int component = 0) const override
  {
    (void)component;
    const double x = p(0);
    const double y = p(1);

    return std::sin(numbers::PI * x / 2.0) * std::sin(numbers::PI * y / 2.0)
           + 0.5 * x * y + 0.1;
  }

  Tensor<1, dim> gradient(const Point<dim> &p,
                          const unsigned int component = 0) const override
  {
    (void)component;
    const double x = p(0);
    const double y = p(1);

    Tensor<1, dim> grad;
    grad[0] = (numbers::PI / 2.0) * std::cos(numbers::PI * x / 2.0)
              * std::sin(numbers::PI * y / 2.0) + 0.5 * y;
    grad[1] = (numbers::PI / 2.0) * std::sin(numbers::PI * x / 2.0)
              * std::cos(numbers::PI * y / 2.0) + 0.5 * x;

    return grad;
  }
};


// Right-hand side (forcing term)
// f = -div(mu * grad(u)) + beta · grad(u) + gamma * u
template <int dim>
class RightHandSideMixed : public Function<dim>
{
public:
  double value(const Point<dim> &p,
               const unsigned int component = 0) const override
  {
    (void)component;
    const double x = p(0);
    const double y = p(1);
    const double pi = numbers::PI;

    const double sin_x = std::sin(pi * x / 2.0);
    const double sin_y = std::sin(pi * y / 2.0);
    const double cos_x = std::cos(pi * x / 2.0);
    const double cos_y = std::cos(pi * y / 2.0);

    // mu = 1 + x^2 + y^2
    // beta = (y, -x)
    // gamma = 2.0
    // u = sin(pi*x/2)*sin(pi*y/2) + 0.5*x*y + 0.1

    // Computed using sympy:
    return 1.0*x*y
           - 0.5*x*(x + pi*sin_x*cos_y)
           - 1.0*x*(y + pi*sin_y*cos_x)
           - 1.0*y*(x + pi*sin_x*cos_y)
           + 0.5*y*(y + pi*sin_y*cos_x)
           + pi*pi*(x*x + y*y + 1)*sin_x*sin_y/2.0
           + 2.0*sin_x*sin_y
           + 0.2;
  }
};


// Neumann boundary condition: g_N = mu * grad(u) · n
// This returns the Neumann data for the right (x=1) and top (y=1) boundaries
template <int dim>
class NeumannBCMixed : public Function<dim>
{
public:
  double value(const Point<dim> &p,
               const unsigned int component = 0) const override
  {
    (void)component;
    const double x = p(0);
    const double y = p(1);

    const double tol = 1e-10;

    // mu = 1 + x^2 + y^2

    // Right boundary (x=1): normal = (1, 0)
    // g_N = mu * du/dx = (1 + 1 + y^2) * (pi/2 * cos(pi/2) * sin(pi*y/2) + 0.5*y)
    //     = (2 + y^2) * (0 + 0.5*y) = 0.5*y*(2 + y^2)
    if (std::abs(x - 1.0) < tol)
    {
      return 0.5 * y * (y * y + 2.0);
    }
    // Top boundary (y=1): normal = (0, 1)
    // g_N = mu * du/dy = (1 + x^2 + 1) * (pi/2 * sin(pi*x/2) * cos(pi/2) + 0.5*x)
    //     = (2 + x^2) * (0 + 0.5*x) = 0.5*x*(2 + x^2)
    else if (std::abs(y - 1.0) < tol)
    {
      return 0.5 * x * (x * x + 2.0);
    }

    // Should not reach here for well-posed problems
    return 0.0;
  }
};


// Diffusion coefficient: mu(x,y) = 1 + x^2 + y^2
template <int dim>
class DiffusionCoefficientMixed : public Function<dim>
{
public:
  double value(const Point<dim> &p,
               const unsigned int component = 0) const override
  {
    (void)component;
    const double x = p(0);
    const double y = p(1);

    return 1.0 + x * x + y * y;
  }
};


// Advection field: beta(x,y) = (y, -x)
template <int dim>
class AdvectionFieldMixed : public TensorFunction<1, dim>
{
public:
  Tensor<1, dim> value(const Point<dim> &p) const override
  {
    const double x = p(0);
    const double y = p(1);

    Tensor<1, dim> beta;
    beta[0] = y;
    beta[1] = -x;

    return beta;
  }
};


// Reaction coefficient: gamma(x,y) = 2.0
template <int dim>
class ReactionCoefficientMixed : public Function<dim>
{
public:
  double value(const Point<dim> &p,
               const unsigned int component = 0) const override
  {
    (void)component;
    (void)p;

    return 2.0;
  }
};

