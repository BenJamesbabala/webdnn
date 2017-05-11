from typing import List

from graph_transpiler.backend.webgpu.allocator import MemoryLayout
from graph_transpiler.backend.webgpu.kernel import Kernel, GPUSize
from graph_transpiler.backend.webgpu.kernels import util
from graph_transpiler.backend.webgpu.meta_buffer_injector import MetaBufferInjector
from graph_transpiler.graph.axis import Axis
from graph_transpiler.graph.operators.axiswise_scale import AxiswiseScale
from graph_transpiler.graph.variables.attributes.order import OrderNHWC, OrderNC, OrderHWNC

template = """
kernel void %%FUNC_NAME%%(device float *data_buffer[[buffer(0)]],
                          const device int * %%META_NAME%% [[buffer(1)]],
                          uint index[[thread_position_in_grid]],
                          uint num_threads[[threads_per_grid]])
{
    const device float *X = data_buffer + %%META_LOAD(axiswise_scale_X_offset)%%;
    const device float *S = data_buffer + %%META_LOAD(axiswise_scale_S_offset)%%;
    device float *Y = data_buffer + %%META_LOAD(axiswise_scale_Y_offset)%%;
    const int N = %%META_LOAD(axiswise_scale_N)%%;
    const int C = %%META_LOAD(axiswise_scale_C)%%;
  
    for (int gid = index; gid < N; gid += num_threads) {
        int c = gid % C;
        float result = X[gid] * S[c];

        Y[gid] = result;
    }
}
"""


def axiswise_scale(op: AxiswiseScale,
                   memory_layout: MemoryLayout,
                   metabuffer_injector: MetaBufferInjector = None) -> List[Kernel]:
    x = memory_layout[op.inputs["x"]]
    s = memory_layout[op.inputs["s"]]
    y = memory_layout[op.outputs["y"]]

    if metabuffer_injector is None:
        metabuffer_injector = MetaBufferInjector()

    assert x.variable.order == OrderNC or x.variable.order == OrderNHWC or x.variable.order == OrderHWNC
    assert y.variable.order == OrderNC or y.variable.order == OrderNHWC or y.variable.order == OrderHWNC
    assert op.parameters["axis"] == Axis.C, "[WebGPU] AxiswiseScale supports only channelwise bias."

    metabuffer_injector.register({
        "axiswise_scale_X_offset": x.offset,
        "axiswise_scale_Y_offset": y.offset,
        "axiswise_scale_S_offset": s.offset,
        "axiswise_scale_N": y.variable.size,
        "axiswise_scale_C": y.variable.shape_dict[Axis.C],
    })

    source = metabuffer_injector.inject(template)
    func_name = util.add_canonical_suffix("axiswise_scale", source)
    source = source.replace("%%FUNC_NAME%%", func_name)

    kernel = Kernel(
        {func_name: source},
        func_name,
        GPUSize(8, 1, 1),
        GPUSize(1024, 1, 1),
        metabuffer_injector.generate_buffer()
    )

    return [kernel]
