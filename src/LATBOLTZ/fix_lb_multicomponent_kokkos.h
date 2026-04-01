#ifdef FIX_CLASS
FixStyle(lb/multicomponent/kokkos,FixLbMulticomponentKokkos)
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

    void initial_integrate(int) override;
    void end_of_step() override;    

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
    
    using HostWlb19 = Kokkos::View<const double*,
                                   Kokkos::LayoutRight,
                                   Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>;
    using HostE19   = Kokkos::View<const int**,
                                   Kokkos::LayoutRight,
                                   Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>;
    using HostWg19  = Kokkos::View<const double***,
                                   Kokkos::LayoutRight,
                                   Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>;
    
    using DeviceWlb19 = Kokkos::View<const double*, Kokkos::LayoutRight>;
    using DeviceE19   = Kokkos::View<const int**,    Kokkos::LayoutRight>;
    using DeviceWg19  = Kokkos::View<const double***,Kokkos::LayoutRight>;
    
    HostWlb19 h_w_lb19;
    HostE19   h_e19;
    HostWg19  h_wg19;
    
    DeviceWlb19 d_w_lb19;
    DeviceE19   d_e19;
    DeviceWg19  d_wg19;

    void create_views();
    void copy_to_views();

  }
}

