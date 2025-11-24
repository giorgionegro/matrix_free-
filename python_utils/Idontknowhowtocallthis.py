#!/usr/bin/env python3


import sympy as sp
from sympy import symbols, Function, diff, simplify, sin, cos, exp, pi

def compute_forcing_term(u_exact, mu, beta, gamma, dim=2):
    """Compute the forcing term f from the exact solution."""

    if dim == 2:
        x, y = symbols('x y', real=True)
        coords = [x, y]
    else:
        x, y, z = symbols('x y z', real=True)
        coords = [x, y, z]



    grad_u = [diff(u_exact, coord) for coord in coords]

    mu_grad_u = [mu * grad_u[i] for i in range(dim)]
    div_mu_grad_u = sum([diff(mu_grad_u[i], coords[i]) for i in range(dim)])
    diffusion_term = -div_mu_grad_u


    beta_u = [beta[i] * u_exact for i in range(dim)]
    div_beta_u = sum([diff(beta_u[i], coords[i]) for i in range(dim)])
    advection_term = div_beta_u



    reaction_term = gamma * u_exact


    f = diffusion_term + advection_term + reaction_term
    f_simplified = simplify(f)





    return f_simplified, grad_u, coords

def generate_cpp_code(u_exact, f, grad_u, mu, beta, gamma, coords):
    """Generate C++ code snippets for deal.II implementation."""

    from sympy.utilities.codegen import codegen
    from sympy.printing.cxx import cxxcode





    dim = len(coords)

    cpp_exact = f"""
// Exact solution
template <int dim>
class ExactSolution : public Function<dim>
{{
public:
  double value(const Point<dim> &p,
                      const unsigned int component = 0) const override
  {{
    const double x = p(0);
    const double y = p(1);"""

    if dim == 3:
        cpp_exact += "\n    const double z = p(2);"

    u_cpp = cxxcode(u_exact)
    u_cpp = u_cpp.replace('M_PI', 'numbers::PI')
    u_cpp = u_cpp.replace('std::pow', 'pow')

    cpp_exact += f"""
    
    return {u_cpp};
  }}
  
  Tensor<1, dim> gradient(const Point<dim> &p,
                                  const unsigned int component = 0) const override
  {{
    const double x = p(0);
    const double y = p(1);"""

    if dim == 3:
        cpp_exact += "\n    const double z = p(2);"

    cpp_exact += "\n    \n    Tensor<1, dim> grad;\n"

    for i, g in enumerate(grad_u):
        grad_cpp = cxxcode(g)
        grad_cpp = grad_cpp.replace('M_PI', 'numbers::PI')
        grad_cpp = grad_cpp.replace('std::pow', 'pow')
        cpp_exact += f"    grad[{i}] = {grad_cpp};\n"

    cpp_exact += """    
    return grad;
  }
};"""

    cpp_forcing = f"""
// Right-hand side (forcing term)
template <int dim>
class RightHandSide : public Function<dim>
{{
public:
  double value(const Point<dim> &p,
                      const unsigned int component = 0) const override
  {{
    const double x = p(0);
    const double y = p(1);"""

    if dim == 3:
        cpp_forcing += "\n    const double z = p(2);"

    f_cpp = cxxcode(f)
    f_cpp = f_cpp.replace('M_PI', 'numbers::PI')
    f_cpp = f_cpp.replace('std::pow', 'pow')

    cpp_forcing += f"""
    
    return {f_cpp};
  }}
}};"""

    mu_cpp = cxxcode(mu)
    mu_cpp = mu_cpp.replace('M_PI', 'numbers::PI')

    cpp_diffusion = f"""
// Diffusion coefficient
template <int dim>
class DiffusionCoefficient : public Function<dim>
{{
public:
  double value(const Point<dim> &p,
                      const unsigned int component = 0) const override
  {{
    const double x = p(0);
    const double y = p(1);"""

    if dim == 3:
        cpp_diffusion += "\n    const double z = p(2);"

    cpp_diffusion += f"""
    
    return {mu_cpp};
  }}
}};"""

    cpp_advection = f"""
// Advection field
template <int dim>
class AdvectionField : public TensorFunction<1, dim>
{{
public:
  Tensor<1, dim> value(const Point<dim> &p) const override
  {{
    const double x = p(0);
    const double y = p(1);"""

    if dim == 3:
        cpp_advection += "\n    const double z = p(2);"

    cpp_advection += "\n    \n    Tensor<1, dim> beta;\n"

    for i, b in enumerate(beta):
        beta_cpp = cxxcode(b)
        beta_cpp = beta_cpp.replace('M_PI', 'numbers::PI')
        beta_cpp = beta_cpp.replace('std::pow', 'pow')
        cpp_advection += f"    beta[{i}] = {beta_cpp};\n"

    cpp_advection += """    
    return beta;
  }
};"""

    gamma_cpp = cxxcode(gamma)
    gamma_cpp = gamma_cpp.replace('M_PI', 'numbers::PI')

    cpp_reaction = f"""
// Reaction coefficient
template <int dim>
class ReactionCoefficient : public Function<dim>
{{
public:
  double value(const Point<dim> &p,
                      const unsigned int component = 0) const override
  {{
    const double x = p(0);
    const double y = p(1);"""

    if dim == 3:
        cpp_reaction += "\n    const double z = p(2);"

    cpp_reaction += f"""
    
    return {gamma_cpp};
  }}
}};"""







    with open('../include/manufactured_solution.hpp', 'w') as f:
        f.write("#pragma once\n\n")
        f.write("#include <deal.II/base/function.h>\n")
        f.write("#include <deal.II/base/tensor_function.h>\n")
        f.write("#include <deal.II/base/point.h>\n")
        f.write("#include <deal.II/base/numbers.h>\n\n")
        f.write("using namespace dealii;\n\n")
        f.write(cpp_exact + "\n\n")
        f.write(cpp_forcing + "\n\n")
        f.write(cpp_diffusion + "\n\n")
        f.write(cpp_advection + "\n\n")
        f.write(cpp_reaction + "\n\n")


def main():
    """Main function with example."""


    x, y = symbols('x y', real=True)

 
    u_exact_3 = exp(-((x-0.5)**2 + (y-0.5)**2))
    mu_3 = 1 + x**2 + y**2
    beta_3 = (y, -x)
    gamma_3 = 2.0

    f_3, grad_u_3, coords_3 = compute_forcing_term(u_exact_3, mu_3, beta_3, gamma_3, dim=2)


    generate_cpp_code(u_exact_3, f_3, grad_u_3, mu_3, beta_3, gamma_3, coords_3)




if __name__ == "__main__":
    main()