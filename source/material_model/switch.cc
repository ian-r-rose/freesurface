/*
  Copyright (C) 2011, 2012 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file doc/COPYING.  If not see
  <http://www.gnu.org/licenses/>.
*/
/*  $Id: simple.cc 1892 2013-09-13 21:29:19Z heister $  */


#include <aspect/material_model/switch.h>
#include <deal.II/base/parameter_handler.h>
#include <aspect/global.h>

using namespace dealii;

namespace aspect
{
  namespace MaterialModel
  {

    template <int dim>
    double
    Switch<dim>::
    viscosity (const double temperature,
               const double,
               const std::vector<double> &composition,       /*composition*/
               const SymmetricTensor<2,dim> &,
               const Point<dim> &) const
    {
      return eta;
    }


    template <int dim>
    double
    Switch<dim>::
    reference_viscosity () const
    {
      return eta;
    }

    template <int dim>
    double
    Switch<dim>::
    reference_density () const
    {
      return reference_rho;
    }

    template <int dim>
    double
    Switch<dim>::
    reference_thermal_expansion_coefficient () const
    {
      return thermal_alpha;
    }

    template <int dim>
    double
    Switch<dim>::
    specific_heat (const double,
                   const double,
                   const std::vector<double> &, /*composition*/
                   const Point<dim> &) const
    {
      return reference_specific_heat;
    }

    template <int dim>
    double
    Switch<dim>::
    reference_cp () const
    {
      return reference_specific_heat;
    }

    template <int dim>
    double
    Switch<dim>::
    thermal_conductivity (const double,
                          const double,
                          const std::vector<double> &, /*composition*/
                          const Point<dim> &) const
    {
      return k_value;
    }

    template <int dim>
    double
    Switch<dim>::
    reference_thermal_diffusivity () const
    {
      return k_value/(reference_rho*reference_specific_heat);
    }

    template <int dim>
    double
    Switch<dim>::
    density (const double temperature,
             const double,
             const std::vector<double> &compositional_fields, /*composition*/
             const Point<dim> &p) const
    {
      if (this->get_time() < switch_time * ( this->convert_output_to_years() ? year_in_seconds : 1.0 ) || std::isnan(this->get_time()))
      {
	Point<dim> test_p; test_p[0] = 4.5e6*std::sin(switch_angle*M_PI/180.0) ; test_p[1]=4.5e6*std::cos(switch_angle*M_PI/180.0);
        return reference_rho + delta_rho*( p.distance(test_p) < 2.e5 ? 1.0: 0.0);
      }
      else
      {
        Point<dim> test_p; test_p[0] = 0.00e6; test_p[1]=4.5e6;
        return reference_rho + delta_rho * ( p.distance(test_p) < 2.e5 ? 1.0 : 0.0);
      }
    }


    template <int dim>
    double
    Switch<dim>::
    thermal_expansion_coefficient (const double temperature,
                                   const double,
                                   const std::vector<double> &, /*composition*/
                                   const Point<dim> &) const
    {
      return thermal_alpha;
    }


    template <int dim>
    double
    Switch<dim>::
    compressibility (const double,
                     const double,
                     const std::vector<double> &, /*composition*/
                     const Point<dim> &) const
    {
      return 0.0;
    }

    template <int dim>
    bool
    Switch<dim>::
    viscosity_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      return false;
    }


    template <int dim>
    bool
    Switch<dim>::
    density_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      return false;
    }

    template <int dim>
    bool
    Switch<dim>::
    compressibility_depends_on (const NonlinearDependence::Dependence) const
    {
      return false;
    }

    template <int dim>
    bool
    Switch<dim>::
    specific_heat_depends_on (const NonlinearDependence::Dependence) const
    {
      return false;
    }

    template <int dim>
    bool
    Switch<dim>::
    thermal_conductivity_depends_on (const NonlinearDependence::Dependence dependence) const
    {
      return false;
    }


    template <int dim>
    bool
    Switch<dim>::
    is_compressible () const
    {
      return false;
    }



    template <int dim>
    void
    Switch<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Switch");
        {
          prm.declare_entry ("Density differential", "0",
                             Patterns::Double (0),
                             "");
          prm.declare_entry ("Switch time", "1e6",
                             Patterns::Double (0),
                             "");
          prm.declare_entry ("Switch angle", "90",
                              Patterns::Double (0),
                              "");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    Switch<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Switch");
        {
          delta_rho    = prm.get_double ("Density differential");
          switch_time    = prm.get_double ("Switch time");
          switch_angle    = prm.get_double ("Switch angle");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
      reference_rho = 3300.;
      reference_T = 0.0;
      eta = 1.e21;
      k_value = 4.5;
      reference_specific_heat = 1250.0;
      thermal_alpha = 4.0e-5;
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(Switch,
                                   "switch",
                                   "A simple material model that has constant values "
                                   "for all coefficients but the density and viscosity. "
                                   "This model uses the formulation that assumes an incompressible"
                                   " medium despite the fact that the density follows the law "
                                   "$\\rho(T)=\\rho_0(1-\\beta(T-T_{\\text{ref}})$. "
                                   "The temperature dependency of viscosity is "
                                   " switched off by default and follows the formula"
                                   "$\\eta(T)=\\eta_0*e^{\\eta_T*\\Delta T / T_{\\text{ref}})}$."
                                   "The value for the components of this formula and additional "
                                   "parameters are read from the parameter file in subsection "
                                   "'Switch model'.")
  }
}
