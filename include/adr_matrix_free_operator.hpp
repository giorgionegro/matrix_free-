#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

using namespace dealii;

// Matrix-free operator for ADR equation
// Inherits from Base for automatic constraint handling
template <int dim, int fe_degree, typename number = double>
class ADRMatrixFreeOperator
  : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;
  using value_type = number;

  ADRMatrixFreeOperator();

  void set_coefficients(const Function<dim> &mu,
                       const TensorFunction<1, dim> &beta,
                       const Function<dim> &gamma);

   void clear() override;
   void compute_diagonal() override;

private:
  // Called by Base::vmult() after handling constraints
   void apply_add(VectorType &dst, const VectorType &src) const override;

  // Cell-wise operator application
  void local_apply(const MatrixFree<dim, number> &data,
                   VectorType &dst,
                   const VectorType &src,
                   const std::pair<unsigned int, unsigned int> &cell_range) const;

  ObserverPointer<const Function<dim>> diffusion_coefficient;
  ObserverPointer<const TensorFunction<1, dim>> advection_field;
  ObserverPointer<const Function<dim>> reaction_coefficient;
};


template <int dim, int fe_degree, typename number>
ADRMatrixFreeOperator<dim, fe_degree, number>::ADRMatrixFreeOperator()
  : MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>()
  , diffusion_coefficient(nullptr)
  , advection_field(nullptr)
  , reaction_coefficient(nullptr)
{}

template <int dim, int fe_degree, typename number>
void ADRMatrixFreeOperator<dim, fe_degree, number>::set_coefficients(
  const Function<dim> &mu,
  const TensorFunction<1, dim> &beta,
  const Function<dim> &gamma)
{

  // Store coefficient functions
  diffusion_coefficient = &mu;
  advection_field = &beta;
  reaction_coefficient = &gamma;
}

template <int dim, int fe_degree, typename number>
void ADRMatrixFreeOperator<dim, fe_degree, number>::clear()
{
  diffusion_coefficient = nullptr;
  advection_field = nullptr;
  reaction_coefficient = nullptr;
  MatrixFreeOperators::Base<dim, VectorType>::clear();
}

template <int dim, int fe_degree, typename number>
void ADRMatrixFreeOperator<dim, fe_degree, number>::apply_add(
  VectorType &dst,
  const VectorType &src) const
{
  this->data->cell_loop(&ADRMatrixFreeOperator::local_apply, this, dst, src, false);
}

template <int dim, int fe_degree, typename number>
void ADRMatrixFreeOperator<dim, fe_degree, number>::compute_diagonal()
{
  // TODO: Implement proper diagonal computation for better preconditioning
  // For now, use identity matrix
  this->inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
  VectorType &diagonal = this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(diagonal);

  for (unsigned int i = 0; i < diagonal.locally_owned_size(); ++i)
    diagonal.local_element(i) = 1.0;
}

template <int dim, int fe_degree, typename number>
void ADRMatrixFreeOperator<dim, fe_degree, number>::local_apply(
  const MatrixFree<dim, number> &data,
  VectorType &dst,
  const VectorType &src,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    phi.reinit(cell);
    phi.read_dof_values(src);
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      const auto value = phi.get_value(q);
      const auto gradient = phi.get_gradient(q);


      VectorizedArray<number> mu_value = {};
      VectorizedArray<number> gamma_value = {};
      Tensor<1, dim, VectorizedArray<number>> beta_value;
      for (unsigned int d = 0; d < dim; ++d)
        beta_value[d] = {};
      //TODO: this need to be rewritten in a more standard way, we probably want to avoid ObserverPointer here for better Vectorization performance
      //TODO: I'm also doing probably unvectorizible operation while I could use  already implemented deal.ii routines
      for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
      {
        Point<dim> q_point;
        for (unsigned int d = 0; d < dim; ++d)
          q_point[d] = phi.quadrature_point(q)[d][v];

        mu_value[v] = diffusion_coefficient->value(q_point);
        gamma_value[v] = reaction_coefficient->value(q_point);

        const Tensor<1, dim> beta_point = advection_field->value(q_point);
        for (unsigned int d = 0; d < dim; ++d)
          beta_value[d][v] = beta_point[d];
      }

      phi.submit_gradient(mu_value * gradient, q);//Diffusion term
      phi.submit_value(beta_value * gradient + gamma_value * value, q); //Advection and Reaction terms
    }

    phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    phi.distribute_local_to_global(dst);
  }
}

