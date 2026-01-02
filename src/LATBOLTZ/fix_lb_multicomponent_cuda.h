#ifdef FIX_CLASS
FixStyle(lb/multicomponent-cuda, FixLbMulticomponentCuda)
#else

#ifndef LMP_FIX_LB_MULTICOMPONENT_CUDA_H
#define LMP_FIX_LB_MULTICOMPONENT_CUDA_H


#include "fix.h"
#include "fix_lb_multicomponent.h"


namespace LAMMPS_NS{
    class FixLbMulticomponentCuda : public FixLbMulticomponent {
    public:
        FixLbMulticomponentCuda(class LAMMPS *, int, char **);
        ~FixLbMulticomponentCuda() override;

        int setmask() override;
        void initial_integrate(int) override;
        void end_of_step() override;

    private:
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

        double *dev_wg19;
        double *dev_e19;
        double *dev_w_lb19;

        void lb_update();
        void cuda_update_cube(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
        void datacpy_cpu_to_gpu();
        void allocateArrays();
        void datacpy_gpu_to_cpu();
        void cuda_read_sites(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
        void cuda_write_sites(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
        void cuda_calc_moments(int x, int y, int z);
        void cuda_calc_equilibrium(int x, int y, int z);
        void cuda_collide_stream(int x, int y, int z);
        void cuda_calc_gradient_laplacian(int x, int y, int z, double *field, double *gradient, double *laplacian);
        void cuda_calc_chemical_potentials(int x, int y, int z);
        void cuda_calc_feq(int x, int y, int z);
        void cuda_calc_geq(int x, int y, int z);
        void cuda_calc_keq(int x, int y, int z);

    };
}
#endif
#endif
