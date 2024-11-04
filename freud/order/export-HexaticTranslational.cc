#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h> // NOLINT(misc-include-cleaner): used implicitly
#include <nanobind/stl/tuple.h> // NOLINT(misc-include-cleaner): used implicitly
#include <utility>

// #include "BondHistogramCompute.h"
#include "NeighborList.h"
#include "NeighborQuery.h"
#include "HexaticTranslational.h"
#include "VectorMath.h"

namespace freud { namespace order {

template<typename T, typename shape>
using nb_array = nanobind::ndarray<T, shape, nanobind::device::cpu, nanobind::c_contig>;

namespace wrap {

void computeHexaticTranslational(const std::shared_ptr<Hexatic>& self, 
                    std::shared_ptr<locality::NeighborList> nlist,
                    std::shared_ptr<locality::NeighborQuery>& points,
                    //nanobind::shape<-1,3>>& points,
                    const locality::QueryArgs& qargs
                    )
{
  //auto* points_data = reinterpret_cast<vec3<float>*>(points.data());

  self->compute(std::move(nlist), points, qargs);
}

//nanobind::tuple getHexaticTranslationalDirector(const std::shared_ptr<HexaticTranslational>& self)
//{
//    vec3<float> nem_d_cpp = self->getHexaticTranslationalDirector();
//    return nanobind::make_tuple(nem_d_cpp.x, nem_d_cpp.y, nem_d_cpp.z);
//}
}; // namespace wrap


namespace detail {

// void export_PMFT(nanobind::module_& m)
// {
//     nanobind::class_<PMFT, locality::BondHistogramCompute>(m, "PMFT").def("getPCF", &PMFT::getPCF);
// }

void export_HexaticTranslational(nanobind::module_& m)
{
    //nanobind::class_<HexaticTranslational>(m, "HexaticTranslational")
    //nanobind::class_<HexaticTranslational>(m, "Hexatic")
    //nanobind::class_<Hexatic>(m, "HexaticTranslational")
    nanobind::class_<Hexatic>(m, "Hexatic")
        .def(nanobind::init<unsigned int, bool>())
        //.def(nanobind::init<>())
        .def("compute", &wrap::computeHexaticTranslational, nanobind::arg("nlist").none(), nanobind::arg("points"), nanobind::arg("qargs"))
        //.def("getOrder", &Hexatic::HexaticTranslational::getOrder)
        //.def("getHexaticTranslationalDirector",&wrap::getHexaticTranslationalDirector)
        //.def("getParticleTensor",&Hexatic::HexaticTranslational::getParticleTensor)
        //.def("getHexaticTranslationalTensor",&HexaticTranslational::getHexaticTranslationalTensor)
        ;
}

} // namespace detail

}; }; // end namespace freud::pmft
