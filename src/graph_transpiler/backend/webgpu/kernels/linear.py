from typing import List

from graph_transpiler.backend.webgpu.allocator import MemoryLayout
from graph_transpiler.backend.webgpu.kernel import Kernel, GPUSize
from graph_transpiler.backend.webgpu.kernels import util
from graph_transpiler.backend.webgpu.meta_buffer_injector import MetaBufferInjector
from graph_transpiler.graph.axis import Axis
from graph_transpiler.graph.operators.linear import Linear
from graph_transpiler.graph.variables.attributes.order import OrderNC, OrderNHWC, OrderCN, OrderHWCN

template = """
kernel void %%FUNC_NAME%%(device float *data_buffer[[buffer(0)]],
                          const device int * %%META_NAME%% [[buffer(1)]],
                          uint index[[thread_position_in_grid]],
                          uint num_threads[[threads_per_grid]])
{
    const device float *X = data_buffer + %%META_LOAD(linear_X_offset)%%;
    const device float *W = data_buffer + %%META_LOAD(linear_W_offset)%%;
    device float *Y = data_buffer + %%META_LOAD(linear_Y_offset)%%;
    const int M = %%META_LOAD(linear_M)%%;
    const int N = %%META_LOAD(linear_N)%%;
    const int K = %%META_LOAD(linear_K)%%;
    
    for (int gid = index; gid < M * N; gid += num_threads) {
        int n = gid % N;
        int m = gid / N;

        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += X[m * K + k] * W[k * N + n];
        }

        Y[gid] = sum;
    }
}
"""


def linear(op: Linear,
           memory_layout: MemoryLayout,
           metabuffer_injector: MetaBufferInjector = None) -> List[Kernel]:
    x = memory_layout[op.inputs["x"]]
    w = memory_layout[op.inputs["w"]]
    y = memory_layout[op.outputs["y"]]

    assert x.variable.order == OrderNC or x.variable.order == OrderNHWC
    assert w.variable.order == OrderCN or w.variable.order == OrderHWCN
    assert y.variable.order == OrderNC or y.variable.order == OrderNHWC
    assert w.variable.ndim == x.variable.ndim

    if metabuffer_injector is None:
        metabuffer_injector = MetaBufferInjector()
    metabuffer_injector.register({
        "linear_X_offset": x.offset,
        "linear_Y_offset": y.offset,
        "linear_W_offset": w.offset,
        "linear_M": y.variable.shape_dict[Axis.N],
        "linear_N": y.variable.size // y.variable.shape_dict[Axis.N],
        "linear_K": x.variable.size // x.variable.shape_dict[Axis.N],
    })

    source = metabuffer_injector.inject(template)
    func_name = util.add_canonical_suffix("linear", source)
    source = source.replace("%%FUNC_NAME%%", func_name)

    kernel = Kernel(
        {func_name: source},
        func_name,
        GPUSize(8, 1, 1),
        GPUSize(1024, 1, 1),
        metabuffer_injector.generate_buffer()
    )

    return [kernel]
