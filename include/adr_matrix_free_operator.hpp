#pragma once

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/table.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

using namespace dealii;

// Matrix-free operator for ADR equation with pre-computed coefficient tables
template <int dim, int fe_degree, typename number = double>
class ADRMatrixFreeOperator
  : public MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<number>;
  using value_type = number;

  ADRMatrixFreeOperator();

  void evaluate_coefficients(const Function<dim> &mu,
                            const TensorFunction<1, dim> &beta,
                            const Function<dim> &gamma);

  const Table<2, VectorizedArray<number>>& get_diffusion_coefficient() const
  { return diffusion_coefficient; }

  const Table<2, Tensor<1, dim, VectorizedArray<number>>>& get_advection_field() const
  { return advection_field; }

  const Table<2, VectorizedArray<number>>& get_reaction_coefficient() const
  { return reaction_coefficient; }

  void clear() override;
  void compute_diagonal() override;

private:
  void apply_add(VectorType &dst, const VectorType &src) const override;

  void local_apply(const MatrixFree<dim, number> &data,
                   VectorType &dst,
                   const VectorType &src,
                   const std::pair<unsigned int, unsigned int> &cell_range) const;

  void local_compute_diagonal(const MatrixFree<dim, number> &data,
                             VectorType &dst,
                             const unsigned int &dummy,
                             const std::pair<unsigned int, unsigned int> &cell_range) const;

  Table<2, VectorizedArray<number>> diffusion_coefficient;
  Table<2, Tensor<1, dim, VectorizedArray<number>>> advection_field;
  Table<2, VectorizedArray<number>> reaction_coefficient;
};


template <int dim, int fe_degree, typename number>
ADRMatrixFreeOperator<dim, fe_degree, number>::ADRMatrixFreeOperator()
  : MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number>>()
{}

template <int dim, int fe_degree, typename number>
void ADRMatrixFreeOperator<dim, fe_degree, number>::evaluate_coefficients(
  const Function<dim> &mu,
  const TensorFunction<1, dim> &beta,
  const Function<dim> &gamma)
{
  const unsigned int n_cells = this->data->n_cell_batches();
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(*this->data);

  diffusion_coefficient.reinit(n_cells, phi.n_q_points);
  advection_field.reinit(n_cells, phi.n_q_points);
  reaction_coefficient.reinit(n_cells, phi.n_q_points);

  for (unsigned int cell = 0; cell < n_cells; ++cell)
  {
    phi.reinit(cell);

    for (unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      const auto q_point_vectorized = phi.quadrature_point(q);

      for (unsigned int v = 0; v < this->data->n_active_entries_per_cell_batch(cell); ++v)
      {
        Point<dim> q_point;
        for (unsigned int d = 0; d < dim; ++d)
          q_point[d] = q_point_vectorized[d][v];

        diffusion_coefficient(cell, q)[v] = mu.value(q_point);
        reaction_coefficient(cell, q)[v] = gamma.value(q_point);

        const Tensor<1, dim> beta_value = beta.value(q_point);
        for (unsigned int d = 0; d < dim; ++d)
          advection_field(cell, q)[d][v] = beta_value[d];
      }
    }
  }
}

template <int dim, int fe_degree, typename number>
void ADRMatrixFreeOperator<dim, fe_degree, number>::clear()
{
  diffusion_coefficient.reinit(0, 0);
  advection_field.reinit(0, 0);
  reaction_coefficient.reinit(0, 0);
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
  this->inverse_diagonal_entries.reset(new DiagonalMatrix<VectorType>());
  VectorType &diagonal = this->inverse_diagonal_entries->get_vector();
  this->data->initialize_dof_vector(diagonal);

  unsigned int dummy = 0;
  this->data->cell_loop(&ADRMatrixFreeOperator::local_compute_diagonal,
                        this,
                        diagonal,
                        dummy);

  this->set_constrained_entries_to_one(diagonal);

  for (unsigned int i = 0; i < diagonal.locally_owned_size(); ++i)
  {
    Assert(diagonal.local_element(i) > 0.,
           ExcMessage("Diagonal entry should be positive"));
    diagonal.local_element(i) = number(1.) / diagonal.local_element(i);
  }
}

template <int dim, int fe_degree, typename number>
void ADRMatrixFreeOperator<dim, fe_degree, number>::local_compute_diagonal(
  const MatrixFree<dim, number> &data,
  VectorType &dst,
  const unsigned int &,
  const std::pair<unsigned int, unsigned int> &cell_range) const
{
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(data);

  AlignedVector<VectorizedArray<number>> diagonal(phi.dofs_per_cell);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    phi.reinit(cell);

    for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
        phi.submit_dof_value(VectorizedArray<number>(), j);
      phi.submit_dof_value(make_vectorized_array<number>(1.), i);

      phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

      for (unsigned int q = 0; q < phi.n_q_points; ++q)
      {
        const auto value = phi.get_value(q);
        const auto gradient = phi.get_gradient(q);

        const auto mu = diffusion_coefficient(cell, q);
        const auto beta = advection_field(cell, q);
        const auto gamma = reaction_coefficient(cell, q);

        phi.submit_gradient(mu * gradient, q);
        phi.submit_value(beta * gradient + gamma * value, q);
      }

      phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      diagonal[i] = phi.get_dof_value(i);
    }

    for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
      phi.submit_dof_value(diagonal[i], i);
    phi.distribute_local_to_global(dst);
  }
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
    phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

    for (unsigned int q = 0; q < phi.n_q_points; ++q)
    {
      const auto value = phi.get_value(q);
      const auto gradient = phi.get_gradient(q);

      const auto mu = diffusion_coefficient(cell, q);
      const auto beta = advection_field(cell, q);
      const auto gamma = reaction_coefficient(cell, q);

      phi.submit_gradient(mu * gradient, q);
      phi.submit_value(beta * gradient + gamma * value, q);
    }

    phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
  }
}

