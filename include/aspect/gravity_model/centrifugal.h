#ifndef __aspect__gravity_model_centrifugal_h
#define __aspect__gravity_model_centrifugal_h

#include <aspect/gravity_model/interface.h>
#include <aspect/gravity_model/radial.h>
#include <aspect/simulator.h>

namespace aspect
{
  namespace GravityModel
  {
    using namespace dealii;

    template <int dim>
    class Centrifugal : public virtual GravityModel::Interface<dim>, public virtual SimulatorAccess<dim>
    {
      public:
        virtual Tensor<1,dim> gravity_vector (const Point<dim> &position) const;
        Tensor<1,dim> centrifugal_vector(const Point<dim> &p) const;

        double get_rotational_energy() const;
        Tensor<1,dim> get_spin_axis() const;

        static void declare_parameters (ParameterHandler &prm);

        virtual void parse_parameters (ParameterHandler &prm);
 
        virtual void update();
        virtual void output(const unsigned int) const;

        virtual void initialize(const Simulator<dim> &);
 
      private:

        SymmetricTensor<2,dim> compute_moment_of_inertia(bool include_density);

        void compute_figure_axis(); //compute figure axis of the planet, i.e. constant density
        void compute_rotation_axis(); //compute total rotation axis, including internal density variations

        void solve_eigenvalue_problem(const SymmetricTensor<2,dim> &tensor, Tensor<1,dim> &max_eigenvector, double *eigenvalues);
	
        double Omega;  //angular velocity of the planet
        double rotational_energy;
        double angular_momentum;

        double principal_moments[dim];  //array for storing the eigenvalues of the moment of inertia tensor
        double figure_principal_moments[dim];  //array for storing the eigenvalues of the figure moment of inertia tensor

	SymmetricTensor<2,dim> moment_of_inertia; //Moment of inertia tensor of the planet
	SymmetricTensor<2,dim> figure_moment_of_inertia; //Moment of inertia tensor of the figure of the planet
 
        Tensor<1,dim> figure_axis;  //axis of the figure
	Tensor<1,dim> rotation_axis; //axis of rotation unit vector
        
    };

    template <int dim>
    class RadialWithCentrifugal : public virtual GravityModel::Interface<dim>, public virtual SimulatorAccess<dim>
    {
      public:
        virtual Tensor<1,dim> gravity_vector (const Point<dim> &position) const;

        virtual void update();

        virtual void initialize(const Simulator<dim> &);

        static void declare_parameters (ParameterHandler &prm);

        virtual void parse_parameters (ParameterHandler &prm);

      private:
         Centrifugal<dim> centrifugal;
         RadialConstant<dim> gravity;
    };

  }
}

#endif
