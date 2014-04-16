#include <aspect/gravity_model/centrifugal.h>
#include <aspect/simulator.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <cmath>
#include <fstream>

namespace aspect
{
  template<int dim>
  SymmetricTensor<2,dim> reduce_tensor(const SymmetricTensor<2,dim> &local, const MPI_Comm &mpi_communicator)
  {
    double entries[6] = {0,0,0,0,0,0};
    double global_entries[6] = {0,0,0,0,0,0};
    SymmetricTensor<2,dim> global;

    for(unsigned int i=0; i<dim; ++i)
      for(unsigned int j=0; j<=i; ++j)
        entries[i*dim + j] = local[i][j];

    MPI_Allreduce (&entries, &global_entries, 6, MPI_DOUBLE,
                       MPI_SUM, mpi_communicator);

    for(unsigned int i=0; i<dim; ++i)
      for(unsigned int j=0; j<=i; ++j)
        global[i][j] = global_entries[i*dim + j];

    return global;
  }
  template<int dim>
  Tensor<1,dim> reduce_vector(const Tensor<1,dim> &local, const MPI_Comm &mpi_communicator)
  {
    double entries[3] = {0,0,0};
    double global_entries[3] = {0,0,0};
    Tensor<1,dim> global;

    for(unsigned int i=0; i<dim; ++i)
      entries[i] = local[i];

    MPI_Allreduce (&entries, &global_entries, 3, MPI_DOUBLE,
                       MPI_SUM, mpi_communicator);

    for(unsigned int i=0; i<dim; ++i)
      global[i] = global_entries[i];
 
    return global;
  }
  namespace GravityModel
  {

    //The Centrifugal bit
    template <int dim>
    void Centrifugal<dim>::initialize(const Simulator<dim> &simulation)
    {
       SimulatorAccess<dim>::initialize(simulation);
    }
   
    template <int dim>
    Tensor<1,dim>
    Centrifugal<dim>::gravity_vector (const Point<dim> &p) const
    {
      return centrifugal_vector(p);
    }

    template<int dim>
    Tensor<1,dim>
    Centrifugal<dim>::centrifugal_vector (const Point<dim> &p) const
    {
      //Not the usual form, but works equally well in 2d and 3d
      const Point<dim> r = p - (p*rotation_axis)*rotation_axis;
      return Omega*Omega*r;
    }
 
    template<int dim>
    Tensor<1,dim>
    Centrifugal<dim>::get_spin_axis () const
    {
       return rotation_axis;
    }
    template<int dim>
    double
    Centrifugal<dim>::get_rotational_energy () const
    {
      return rotational_energy;
    }
    
    template<int dim>
    void Centrifugal<dim>::output(const unsigned int) const
    {
      if( Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0 )
      {
        std::cout<<"Spin vector: "<<std::setprecision(15)<<rotation_axis<<"\tTime: "<<this->get_time()<<"\tOmega: "<<Omega<<"\tRotational energy: "<<rotational_energy<<std::endl;
        std::cout<<"Moment of inertia: "<<std::setprecision(15)<<moment_of_inertia<<std::endl;

        const std::string file_name = this->get_output_directory()+"spin_statistics.txt";
        std::ofstream spin_file (file_name.c_str(), std::ofstream::app);

        spin_file<<std::setprecision(15)<<this->get_time()<<"\t"<<Omega<<"\t"<<rotational_energy<<"\t"<<rotation_axis<<"\t"<<figure_axis<<"\t"<<std::acos(rotation_axis*figure_axis)*180.0/M_PI<<"\t";
        for( unsigned int i=0; i<dim; ++i) spin_file<<std::setprecision(15)<<principal_moments[i]<<"\t";
        spin_file<<std::endl;

        spin_file.close();
      }
    }

    template<>
    void Centrifugal<3>::solve_eigenvalue_problem(const SymmetricTensor<2,3> &tensor, Tensor<1,3> &max_eigenvector, double *eigenvalues)
    {
      double m = trace(tensor)/3.0;
      SymmetricTensor<2,3> tmp = (tensor - m*unit_symmetric_tensor<3>());
      double q = determinant(tmp)/2.0;
      if (q == 0.0) //degenerate, just return z hat
      {
        max_eigenvector = 0;
        max_eigenvector[2] = 1.0;
        return;
      }

      double p = tmp.norm()*tmp.norm()/6.0;
      double arg = (p*p*p - q*q);  if(arg < 0.0) arg = 0.0;
      double phi = std::atan(std::sqrt(arg)/q)/3.0;
      if( phi < 0.0) phi = 0.0;
      if( phi > M_PI) phi = M_PI;

      eigenvalues[0] = m + 2.0* std::sqrt(p)*std::cos(phi);
      eigenvalues[1] = m - std::sqrt(p)*(std::cos(phi) + std::sqrt(3.0)*std::sin(phi));
      eigenvalues[2] = m - std::sqrt(p)*(std::cos(phi) - std::sqrt(3.0)*std::sin(phi));
      
      Tensor<2,3> tmp2 = Tensor<2,3>(tensor - eigenvalues[1]*unit_symmetric_tensor<3>()) *
                         Tensor<2,3>(tensor - eigenvalues[2]*unit_symmetric_tensor<3>());
      for(int i = 0; i <3; ++i)
      {
        Tensor<1,3> test, product; test = 0.0;  test[i] = 1.0;
        product = tmp2*test;
        if ( product.norm() > 1e-10)
        {
          max_eigenvector = product/product.norm();
          break;
        }
      }
      return;
    }


    template<>
    void Centrifugal<2>::solve_eigenvalue_problem(const SymmetricTensor<2,2> &tensor, Tensor<1,2> &max_eigenvector, double *eigenvalues)
    {
      double tr = trace(tensor);
      double det = determinant(tensor);
      double disc = tr*tr - 4.0*det;
      if (disc < 0.0) disc = 0.0;   

      //compute the eigenvalues
      eigenvalues[0] = (tr + std::sqrt(disc))/2.0;
      eigenvalues[1] = (tr - std::sqrt(disc))/2.0;
      this->get_pcout()<<"Eigenvalues: "<<std::setprecision(15)<<eigenvalues[0]<<"  "<<eigenvalues[1]<<"  "<<eigenvalues[0]/eigenvalues[1] - 1.0<<std::endl;

      //annihilate the eigenvector for the second eigenvalue
      SymmetricTensor<2,2> tmp = tensor - eigenvalues[1]*unit_symmetric_tensor<2>();
      for(int i = 0; i <2; ++i)
      {
        Tensor<1,2> test, product; test = 0.0;  test[i] = 1.0;
        product = tmp*test;
        if ( product.norm() > 1e-10)
        {
          max_eigenvector = product/product.norm();
          break;
        }
      }
      return;
    }

 
    template<int dim>
    void Centrifugal<dim>::compute_rotation_axis()
    {
      moment_of_inertia = compute_moment_of_inertia(true);
      solve_eigenvalue_problem(moment_of_inertia, rotation_axis, principal_moments);
    }
      
    template<int dim>
    void Centrifugal<dim>::compute_figure_axis()
    {
      figure_moment_of_inertia = compute_moment_of_inertia(false);
      solve_eigenvalue_problem(figure_moment_of_inertia, figure_axis, figure_principal_moments);
    }

    template<int dim>
    void Centrifugal<dim>::update()
    {
      compute_rotation_axis();
      compute_figure_axis();

      if(this->get_timestep() == 0)
        angular_momentum = Tensor<1,dim>( moment_of_inertia * rotation_axis * Omega ).norm();

      Omega = angular_momentum/Tensor<1,dim>(moment_of_inertia * rotation_axis).norm();
      rotational_energy = (rotation_axis * (moment_of_inertia * rotation_axis) )*Omega*Omega;

      output(0);
    }

    template<int dim>
    SymmetricTensor<2,dim> Centrifugal<dim>::compute_moment_of_inertia( bool include_density)
    {
      QGauss<dim> quadrature(this->get_dof_handler().get_fe().base_element(2).degree+1);
      FEValues<dim> fe(this->get_mapping(), this->get_dof_handler().get_fe(), quadrature,
                              UpdateFlags(update_quadrature_points | update_JxW_values | update_values));

      typename DoFHandler<dim>::active_cell_iterator cell;
      std::vector<Point<dim> > q_points(quadrature.size());
      std::vector<Vector<double> > fe_vals(quadrature.size(), Vector<double>(this->get_dof_handler().get_fe().n_components()));

      //allocate the local and global moments
      SymmetricTensor<2,dim> local_moment;
      SymmetricTensor<2,dim> global_moment;

      //loop over all local cells
      for (cell = this->get_dof_handler().begin_active(); cell != this->get_dof_handler().end(); ++cell)
        if (cell->is_locally_owned())
        {
          fe.reinit (cell);
          q_points = fe.get_quadrature_points();
          fe.get_function_values(this->get_solution(), fe_vals);

          //get the density at each quadrature point
          typename MaterialModel::Interface<dim>::MaterialModelInputs in(q_points.size(), this->n_compositional_fields());
          typename MaterialModel::Interface<dim>::MaterialModelOutputs out(q_points.size(), this->n_compositional_fields());
          if( include_density )
          {
            for(int i=0; i< q_points.size(); i++)
            {
               in.pressure[i] = fe_vals[i][dim];
               in.temperature[i] = fe_vals[i][dim+1];
               for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
                 in.composition[i][c] = fe_vals[i][dim+2+c];
               in.position[i] = q_points[i];

            }
            this->get_material_model().evaluate(in, out);
          }
          else out.densities.assign(q_points.size(), 1.0);
  
          //actually compute the moment of inertia
          Point<dim> r_vec;
          for (unsigned int k=0; k< quadrature.size(); ++k)
            {
              r_vec = q_points[k];
              if(dim == 2)
              {
                local_moment[0][0] += fe.JxW(k) *(r_vec.square() - r_vec[0]*r_vec[0])*out.densities[k];
                local_moment[1][1] += fe.JxW(k) *(r_vec.square() - r_vec[1]*r_vec[1])*out.densities[k];
                local_moment[0][1] += -fe.JxW(k) *( r_vec[1]*r_vec[0])*out.densities[k];
              }
              else if(dim==3)
              {
                local_moment[0][0] += fe.JxW(k) *(r_vec.square() - r_vec[0]*r_vec[0])*out.densities[k];
                local_moment[1][1] += fe.JxW(k) *(r_vec.square() - r_vec[1]*r_vec[1])*out.densities[k];
                local_moment[2][2] += fe.JxW(k) *(r_vec.square() - r_vec[2]*r_vec[2])*out.densities[k];
                local_moment[0][1] += -fe.JxW(k) *( r_vec[0]*r_vec[1])*out.densities[k];
                local_moment[0][2] += -fe.JxW(k) *( r_vec[0]*r_vec[2])*out.densities[k];
                local_moment[1][2] += -fe.JxW(k) *( r_vec[1]*r_vec[2])*out.densities[k];
       
              }
            }
        }
      global_moment = reduce_tensor(local_moment, this->get_mpi_communicator());
      return global_moment;
    }

    template <int dim>
    void
    Centrifugal<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Gravity model");
      {
        prm.enter_subsection("Centrifugal");
        {
          prm.declare_entry ("Omega", "7.29e-5",
                             Patterns::Double (0),
                             "Angular velocity of the planet, rad/s.");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }


    template <int dim>
    void
    Centrifugal<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Gravity model");
      {
        prm.enter_subsection("Centrifugal");
        {
          Omega = prm.get_double ("Omega");
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }
    template <int dim>
    void
    RadialWithCentrifugal<dim>::parse_parameters (ParameterHandler &prm)
    {
      gravity.parse_parameters(prm);
      centrifugal.parse_parameters(prm);
    }
    template <int dim>
    void
    RadialWithCentrifugal<dim>::declare_parameters (ParameterHandler &prm)
    {
      RadialConstant<dim>::declare_parameters(prm);
      Centrifugal<dim>::declare_parameters(prm);
    }

    template <int dim>
    Tensor<1,dim>
    RadialWithCentrifugal<dim>::gravity_vector (const Point<dim> &p) const
    {
      return gravity.gravity_vector(p) + centrifugal.gravity_vector(p);
    }
    template <int dim>
    void
    RadialWithCentrifugal<dim>::update()
    {
      centrifugal.update();
    }
    template <int dim>
    void RadialWithCentrifugal<dim>::initialize(const Simulator<dim> &simulation)
    {
       centrifugal.initialize(simulation);
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace GravityModel
  {
    ASPECT_REGISTER_GRAVITY_MODEL(Centrifugal,
                                  "centrifugal",
                                  "what it sounds like.")
    ASPECT_REGISTER_GRAVITY_MODEL(RadialWithCentrifugal,
                                  "radial with centrifugal",
                                  "what it sounds like.")

  }
}
