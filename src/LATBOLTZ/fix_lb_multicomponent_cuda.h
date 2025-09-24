#ifdef FIX_CLASS
FixStyle(lb/multicomponent_cuda,FixLbMulticomponentCuda)
#else

#ifndef LMP_FIX_LB_MULTICOMPONENT_CUDA_H
#define LMP_FIX_LB_MULTICOMPONENT_CUDA_H

#include "fix.h"
#include "fix_lb_fluid.h"

namespace LAMMPS_NS{
    double *dev_density_lb;
    double *dev_phi_lb;
    double *dev_psi_lb;
    double *dev_pressure_lb;
    double *dev_u_lb;
    double *dev_f_lb;
    double *dev_g_lb;
    double *dev_k_lb;

    double *dev_fnew;
    double *dev_gnew;
    double *dev_knew;
    double *dev_feq;
    double *dev_geq;
    double *dev_keq;

    double *dev_laplace_rho;
    double *dev_laplace_phi;
    double *dev_laplace_psi;
    double *dev_mu_phi;
    double *dev_mu_psi;
    double *dev_density_gradient;
    double *dev_phi_gradient;
    double *dev_psi_gradient;

}