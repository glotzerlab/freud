#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/tuple.h> // NOLINT(misc-include-cleaner): used implicitly
#include <utility>

// #include "BondHistogramCompute.h"
// #include "NeighborList.h"
// #include "NeighborQuery.h"
#include "Nematic.h"
#include "VectorMath.h"

namespace freud { namespace order {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void computeNematic(const std::shared_ptr<Nematic>& self, 
                    const nb_array<float, nanobind::shape<-1,3>>& orientations)
{
  unsigned int const num_orientations = orientations.shape(0);
  auto* orientations_data = reinterpret_cast<vec3<float>*>(orientations.data());
  
  self->compute(orientations_data, num_orientations);
}

nanobind::tuple getNematicDirector(const std::shared_ptr<Nematic>& self)
{
    vec3<float> nem_d_cpp = self->getNematicDirector();
    return nanobind::make_tuple(nem_d_cpp.x, nem_d_cpp.y, nem_d_cpp.z);
}
}; // namespace wrap


namespace detail {

// void export_PMFT(nanobind::module_& m)
// {
//     nanobind::class_<PMFT, locality::BondHistogramCompute>(m, "PMFT").def("getPCF", &PMFT::getPCF);
// }

void export_Nematic(nanobind::module_& m)
{
    nanobind::class_<Nematic>(m, "Nematic")
        .def(nanobind::init<>())
        .def("compute", &wrap::computeNematic, nanobind::arg("orientations"))
        .def("getNematicOrderParameter", &Nematic::getNematicOrderParameter)
        .def("getNematicDirector",&wrap::getNematicDirector)
        .def("getParticleTensor",&Nematic::getParticleTensor)
        .def("getNematicTensor",&Nematic::getNematicTensor)
        ;
}

} // namespace detail

}; }; // end namespace freud::pmft
