
#include <aspect/simulator.h>
#include <aspect/global.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/distributed/solution_transfer.h>


namespace aspect
{
  using namespace dealii;
  
  template <int dim>
  void Simulator<dim>::resume_solution_only()
  {
    try
      {
        triangulation.load ((parameters.output_directory + "restart.mesh").c_str());
      }
    catch (...)
      {
        AssertThrow(false, ExcMessage("Cannot open snapshot mesh file."));
      }
    global_volume = GridTools::volume (triangulation, mapping);
    setup_dofs();

    LinearAlgebra::BlockVector
    distributed_system (system_rhs);
    LinearAlgebra::BlockVector
    old_distributed_system (system_rhs);
    LinearAlgebra::BlockVector
    old_old_distributed_system (system_rhs);
    std::vector<LinearAlgebra::BlockVector *> x_system (3);
    x_system[0] = & (distributed_system);
    x_system[1] = & (old_distributed_system);
    x_system[2] = & (old_old_distributed_system);

    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::BlockVector>
    system_trans (dof_handler);

    system_trans.deserialize (x_system);

    solution = distributed_system;
    old_solution = old_distributed_system;
    old_old_solution = old_old_distributed_system;
 
    time                      = 0;
    timestep_number           = 0;
    time_step = old_time_step = 0;

    
    pcout << "*** Resuming just solution from snapshot!" << std::endl << std::endl;
  }
}

