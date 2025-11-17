#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bilagrid_sample_forward", &bilagrid_sample_forward_tensor);
    m.def("bilagrid_sample_backward", &bilagrid_sample_backward_tensor);
    m.def("bilagrid_uniform_sample_forward", &bilagrid_uniform_sample_forward_tensor);
    m.def("bilagrid_uniform_sample_backward", &bilagrid_uniform_sample_backward_tensor);
    m.def("bilagrid_patched_sample_forward", &bilagrid_patched_sample_forward_tensor);
    m.def("bilagrid_patched_sample_backward", &bilagrid_patched_sample_backward_tensor);
    m.def("tv_loss_forward", &tv_loss_forward_tensor);
    m.def("tv_loss_backward", &tv_loss_backward_tensor);
}
