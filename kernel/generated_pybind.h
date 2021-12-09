//#include <pybind11/pybind11.h>
//#include "dlpack.h"
//#include "kernel.h"
//#include "csr.h"

//namespace py = pybind11;


// expects python capsule for output
// convert to simple 1D or 2D array
// once inside, calls invoke_gspmm
inline void export_kernel(py::module &m) { 
    m.def("gspmm",
        [](graph_t& graph, py::capsule& input, py::capsule& output, bool reverse, bool norm) {
            array2d_t<float> input_array = capsule_to_array2d(input);
            array2d_t<float> output_array = capsule_to_array2d(output);
            return invoke_gspmm(graph, input_array, output_array, reverse, norm);
        }
    );

    m.def("forward_edge_attn",
        [](graph_t& graph, py::capsule& input1, py::capsule& input2, py::capsule& output) {
            array2d_t<float> input_array1 = capsule_to_array2d(input1);
            array2d_t<float> input_array2 = capsule_to_array2d(input2);
            array2d_t<float> output_array = capsule_to_array2d(output);
            return invoke_forward_edge_attention(graph, input_array1, input_array2, output_array);
        }
    );

    m.def("backprop_attn",
        [](graph_t& graph, py::capsule& input, py::capsule& output1, py::capsule& output2) {
            array2d_t<float> input_array = capsule_to_array2d(input);
            array2d_t<float> output_array1 = capsule_to_array2d(output1);
            array2d_t<float> output_array2 = capsule_to_array2d(output2);
            return invoke_forward_edge_attention(graph, input_array, output_array1, output_array2);
        }
    );


}
