#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bilagrid_sample_forward", &bilagrid_sample_forward_tensor);
    m.def("bilagrid_sample_backward", &bilagrid_sample_backward_tensor);
    m.def("bilagrid_uniform_sample_forward", &bilagrid_uniform_sample_forward_tensor);
    m.def("bilagrid_uniform_sample_backward", &bilagrid_uniform_sample_backward_tensor);
    m.def("bilagrid_patched_sample_forward", &bilagrid_patched_sample_forward_tensor);
    m.def("bilagrid_patched_sample_backward", &bilagrid_patched_sample_backward_tensor);
    m.def("bilagrid_ppisp_sample_forward", &bilagrid_ppisp_sample_forward_tensor);
    m.def("bilagrid_ppisp_sample_backward", &bilagrid_ppisp_sample_backward_tensor);
    m.def("bilagrid_ppisp_packed_sample_forward", &bilagrid_ppisp_packed_sample_forward_tensor);
    m.def("bilagrid_ppisp_packed_sample_backward", &bilagrid_ppisp_packed_sample_backward_tensor);
    m.def("bilagrid_ppisp_uniform_sample_forward", &bilagrid_ppisp_uniform_sample_forward_tensor);
    m.def("bilagrid_ppisp_uniform_sample_backward", &bilagrid_ppisp_uniform_sample_backward_tensor);
    m.def("bilagrid_ppisp_patched_sample_forward", &bilagrid_ppisp_patched_sample_forward_tensor);
    m.def("bilagrid_ppisp_patched_sample_backward", &bilagrid_ppisp_patched_sample_backward_tensor);
    m.def("bilagrid_loglinear_uniform_sample_forward", &bilagrid_loglinear_uniform_sample_forward_tensor);
    m.def("bilagrid_loglinear_uniform_sample_backward", &bilagrid_loglinear_uniform_sample_backward_tensor);
    m.def("bilagrid_loglinear_patched_sample_forward", &bilagrid_loglinear_patched_sample_forward_tensor);
    m.def("bilagrid_loglinear_patched_sample_backward", &bilagrid_loglinear_patched_sample_backward_tensor);
    m.def("compute_depth_scalars", &compute_depth_scalars_tensor);
    m.def("bilagrid_depth_uniform_sample_forward", &bilagrid_depth_uniform_sample_forward_tensor);
    m.def("bilagrid_depth_uniform_sample_backward", &bilagrid_depth_uniform_sample_backward_tensor);
    m.def("bilagrid_depth_patched_sample_forward", &bilagrid_depth_patched_sample_forward_tensor);
    m.def("bilagrid_depth_patched_sample_backward", &bilagrid_depth_patched_sample_backward_tensor);
    m.def("bilagrid_normal_uniform_sample_forward", &bilagrid_normal_uniform_sample_forward_tensor);
    m.def("bilagrid_normal_uniform_sample_backward", &bilagrid_normal_uniform_sample_backward_tensor);
    m.def("bilagrid_normal_patched_sample_forward", &bilagrid_normal_patched_sample_forward_tensor);
    m.def("bilagrid_normal_patched_sample_backward", &bilagrid_normal_patched_sample_backward_tensor);
    m.def("tv_loss_forward", &tv_loss_forward_tensor);
    m.def("tv_loss_backward", &tv_loss_backward_tensor);
    m.def("tv_loss_backward_inplace", &tv_loss_backward_inplace_tensor);
    m.def("channel_mean_forward", &channel_mean_forward_tensor);
    m.def("channel_mean_backward", &channel_mean_backward_tensor);
    m.def("channel_mean_backward_inplace", &channel_mean_backward_inplace_tensor);
}
