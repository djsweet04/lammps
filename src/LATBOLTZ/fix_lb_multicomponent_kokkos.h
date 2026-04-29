#ifdef FIX_CLASS
FixStyle(lb/multicomponent_kokkos,FixLbMulticomponentKokkos)
#else

#ifndef LMP_FIX_LB_MULTICOMPONENT_KOKKOS_H
#define LMP_FIX_LB_MULTICOMPONENT_KOKKOS_H

#include "fix_lb_multicomponent.h"
#include <Kokkos_Core.hpp>

namespace LAMMPS_NS {

  class FixLbMulticomponentKokkos : public FixLbMulticomponent {

  public:
    FixLbMulticomponentKokkos(class LAMMPS *, int, char **);
    ~FixLbMulticomponentKokkos() override;

    int setmask() override;
    void initial_integrate(int) override;
    void end_of_step() override;   
    void calc_moments_full();
    void init_mixture_d();
    void init_droplet(double radius);
    void init_liquid_lens(double radius);
    void init_double_emulsion(double radius);
    void init_film(double thickness, double C1_film, double C2_film);
    void init_mixed_droplet(double radius, double C1_drop, double C2_drop);
    void init_fluid();
    void halo_comm(int dir);
    void halo_wait();
    void halo_comm();
    void init_halo();
    void destroy_halo();
    void dump_xdmf(const int step);
    void init_output(void);
    void destroy_output(void);
    void init_lattice();
    void destroy_lattice();
    void init_parameters(int argc, char **argv);
     

    Kokkos::View<double***> ViewWg19;
    Kokkos::View<double*> ViewWlb19;
    Kokkos::View<int**> ViewE19;

    Kokkos::View<double****> ViewF;
    Kokkos::View<double****> ViewG;
    Kokkos::View<double****> ViewK;
    Kokkos::View<double****> ViewFNew;
    Kokkos::View<double****> ViewGNew;
    Kokkos::View<double****> ViewKNew;
    Kokkos::View<double****> ViewFeq;
    Kokkos::View<double****> ViewGeq;
    Kokkos::View<double****> ViewKeq;
    Kokkos::View<double***> ViewDensity;
    Kokkos::View<double****> ViewU;
    Kokkos::View<double***> ViewPhi;
    Kokkos::View<double***> ViewPsi;
    Kokkos::View<double***> ViewPressure;
    Kokkos::View<double***> ViewMuRho;
    Kokkos::View<double***> ViewMuPhi;
    Kokkos::View<double***> ViewMuPsi;
    Kokkos::View<double****> ViewDensityGradient;
    Kokkos::View<double****> ViewPhiGradient;
    Kokkos::View<double****> ViewPsiGradient;
    Kokkos::View<double***> ViewLaplaceRho;
    Kokkos::View<double***> ViewLaplacePhi;
    Kokkos::View<double***> ViewLaplacePsi;


    // Failed attempt to use constant memory for constants
    /* using HostWlb19 = Kokkos::View<double*,
                                   Kokkos::LayoutRight,
                                   Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>;
    using HostE19   = Kokkos::View<int**,
                                   Kokkos::LayoutRight,
                                   Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>;
    using HostWg19  = Kokkos::View<double***,
                                   Kokkos::LayoutRight,
                                   Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>;
    
    using DeviceWlb19 = Kokkos::View<double*, Kokkos::LayoutRight>;
    using DeviceE19   = Kokkos::View<int**,    Kokkos::LayoutRight>;
    using DeviceWg19  = Kokkos::View<double***,Kokkos::LayoutRight>;
    
    HostWlb19 h_w_lb19;
    HostE19   h_e19;
    HostWg19  h_wg19;
    
    DeviceWlb19 d_w_lb19;
    DeviceE19   d_e19;
    DeviceWg19  d_wg19; */

    KOKKOS_INLINE_FUNCTION
    double pressure_kokkos(double rho, double phi, double psi, double cs2, double kappa1, double kappa2, double kappa3) {
      const double rho2 = rho*rho;
      const double rho3 = rho2*rho;
      const double rho4 = rho3*rho;
      const double phi2 = phi*phi;
      const double phi3 = phi2*phi;
      const double phi4 = phi3*phi;
      const double psi2 = psi*psi;
      const double psi3 = psi2*psi;
      const double psi4 = psi3*psi;

      double p0 = rho*cs2
        + (kappa1+kappa2)*(3./32.*(rho4+phi4+psi4)
                          - 1./4.*(rho3+rho*psi-psi3)
                          + 1./8.*(rho2+phi2+psi2)
                          - 3./8.*(rho3*psi+psi3*rho)
                          + 9./16.*(rho2*phi2+rho2*psi2+phi2*psi2)
                          + 3./4.*(rho2*psi-rho*phi2-rho*psi2+phi2*psi)
                          - 9./8.*phi2*psi*rho)
        + (kappa1-kappa2)*(3./8.*(rho3*phi+rho*phi3-phi3*psi-phi*psi3)
                          + 1./4.*(rho*phi-phi*psi-phi3)
                          + 9./8.*(phi*psi2*rho-phi*psi*rho2)
                          - 3./4.*(rho2*phi+phi*psi2)
                          + 3./2.*phi*psi*rho)
        + kappa3*(3./2.*psi4 - 2.*psi3 + 1./2.*psi2);

      return p0;
    }



    void lb_update();
    void create_views();
    void copy_from_views();
    void copy_to_views();
    void read_sites(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
    void calc_gradients_laplacians(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
    void calc_chemical_potentials(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
    void calc_feq(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
    void calc_geq(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
    void calc_keq(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
    void collide_stream(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
    void add_one(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
  };
}
#endif
#endif

