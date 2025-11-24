#!/usr/bin/env python3


import sympy as sp
from sympy import symbols, Function, diff, simplify, sin, cos, exp, pi, latex

def compute_forcing_term(u_exact, mu, beta, gamma, dim=2):
    """Compute the forcing term f from the exact solution."""

    if dim == 2:
        x, y = symbols('x y', real=True)
        coords = [x, y]
    else:
        x, y, z = symbols('x y z', real=True)
        coords = [x, y, z]

    print(f"Computing forcing term for {dim}D problem...")
    print(f"Exact solution: u = {u_exact}")
    print(f"Diffusion coefficient: μ = {mu}")
    print(f"Advection field: β = {beta}")
    print(f"Reaction coefficient: γ = {gamma}")
    print("\n" + "="*70)

    grad_u = [diff(u_exact, coord) for coord in coords]
    print(f"\n∇u = {grad_u}")

    mu_grad_u = [mu * grad_u[i] for i in range(dim)]
    div_mu_grad_u = sum([diff(mu_grad_u[i], coords[i]) for i in range(dim)])
    diffusion_term = -div_mu_grad_u

    print(f"\nDiffusion term: -∇·(μ∇u) = {simplify(diffusion_term)}")

    beta_u = [beta[i] * u_exact for i in range(dim)]
    div_beta_u = sum([diff(beta_u[i], coords[i]) for i in range(dim)])
    advection_term = div_beta_u

    print(f"Advection term: ∇·(βu) = {simplify(advection_term)}")

    reaction_term = gamma * u_exact
    print(f"Reaction term: γu = {simplify(reaction_term)}")

    f = diffusion_term + advection_term + reaction_term
    f_simplified = simplify(f)

    print("\n" + "="*70)
    print(f"FORCING TERM: f = {f_simplified}")
    print("="*70)

    return f_simplified, grad_u, coords

def generate_weak_formulation(dim=2):
    """Generate the weak formulation in LaTeX."""

    print("\n\n" + "="*70)
    print("WEAK FORMULATION")
    print("="*70)

    weak_form = r"""
Find u ∈ V such that for all v ∈ V₀:

a(u,v) = F(v)

where:

a(u,v) = ∫_Ω μ∇u·∇v dx + ∫_Ω β·∇u v dx + ∫_Ω γuv dx

F(v) = ∫_Ω fv dx + ∫_{Γ_N} hv ds

with:
- V = {v ∈ H¹(Ω) : v = g on Γ_D}
- V₀ = {v ∈ H¹(Ω) : v = 0 on Γ_D}
"""


    latex_weak = r"""
\text{Find } u \in V \text{ such that for all } v \in V_0:

a(u,v) = F(v)

\text{where:}

a(u,v) = \int_{\Omega} \mu \nabla u \cdot \nabla v \, dx 
       + \int_{\Omega} \beta \cdot \nabla u \, v \, dx 
       + \int_{\Omega} \gamma u v \, dx

F(v) = \int_{\Omega} f v \, dx + \int_{\Gamma_N} h v \, ds
"""

    print("\nLaTeX version:")
    print(latex_weak)

    return weak_form

def generate_cpp_code(u_exact, f, grad_u, mu, beta, gamma, coords):
    """Generate C++ code snippets for deal.II implementation."""

    from sympy.utilities.codegen import codegen
    from sympy.printing.cxx import cxxcode

    print("\n\n" + "="*70)
    print("C++ CODE GENERATION")
    print("="*70)

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

    print(cpp_exact)
    print("\n" + cpp_forcing)
    print("\n" + cpp_diffusion)
    print("\n" + cpp_advection)
    print("\n" + cpp_reaction)

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

    print("\n\nCode saved to 'manufactured_solution.hpp'")

def main():
    """Main function with example."""

    print("="*70)
    print("WEAK FORMULATION AND FORCING TERM GENERATOR")
    print("For Advection-Diffusion-Reaction Problems")
    print("="*70)

    x, y = symbols('x y', real=True)

    print("\n\n Polynomial solution")
    print("-" * 70)
    u_exact = x * (1 - x) * y * (1 - y)
    mu = 1.0
    beta = (1.0, 0.5)
    gamma = 1.0

    f, grad_u, coords = compute_forcing_term(u_exact, mu, beta, gamma, dim=2)

    print("\n\n\nTrigonometric solution")
    print("-" * 70)
    u_exact_2 = sin(pi * x) * sin(pi * y)
    mu_2 = 1.0
    beta_2 = (0.0, 0.0)
    gamma_2 = 0.0

    f_2, grad_u_2, coords_2 = compute_forcing_term(u_exact_2, mu_2, beta_2, gamma_2, dim=2)

    print("\n\n\nEXAMPLE 3: Variable coefficients")
    print("-" * 70)
    u_exact_3 = exp(-((x-0.5)**2 + (y-0.5)**2))
    mu_3 = 1 + x**2 + y**2
    beta_3 = (y, -x)
    gamma_3 = 2.0

    f_3, grad_u_3, coords_3 = compute_forcing_term(u_exact_3, mu_3, beta_3, gamma_3, dim=2)

    generate_weak_formulation(dim=2)

    print("\n\nGenerating C++ code for Example 3...")
    generate_cpp_code(u_exact_3, f_3, grad_u_3, mu_3, beta_3, gamma_3, coords_3)

    print("\n\n" + "="*70)
    print("="*70)


if __name__ == "__main__":
    main()