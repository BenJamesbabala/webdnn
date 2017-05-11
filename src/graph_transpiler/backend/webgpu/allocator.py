from typing import Dict, List, Set

import numpy as np

import graph_transpiler.util.flags.optimize
from graph_transpiler.backend.interface.memory_layout import IMemoryLayout, IAllocation
from graph_transpiler.graph import traverse
from graph_transpiler.graph.graph import Graph
from graph_transpiler.graph.operator import Operator
from graph_transpiler.graph.operators.attributes.inplace import Inplace
from graph_transpiler.graph.variable import Variable
from graph_transpiler.graph.variables.attributes.constant import Constant
from graph_transpiler.graph.variables.constant_variable import ConstantVariable
from graph_transpiler.util import json, flags


class AllocationAnalysisData:
    variable: Variable
    start: int
    end: int
    offset: int
    size: int

    def __init__(self, variable, start, end=-1, offset=-1):
        self.variable = variable
        self.size = variable.size
        self.start = start
        self.end = end
        self.offset = offset


class Allocation(json.SerializableMixin, IAllocation):
    variable: Variable
    offset: int

    def __init__(self,
                 variable: Variable,
                 offset: int):
        self.variable = variable
        self.offset = offset

    @property
    def size(self) -> int:
        return self.variable.size

    def _to_serializable_(self):
        return {
            "name": self.variable.name,
            "offset": self.offset,
            "size": self.size
        }


class MemoryLayout(json.SerializableMixin, IMemoryLayout):
    size: int
    allocations: Dict[str, Allocation]
    data: np.array

    def __init__(self):
        self.allocations = {}

    def _to_serializable_(self):
        return {
            "total_size": self.size,
            "allocation": {a.variable.name: a for _, a in self.allocations.items()}
        }

    def __len__(self):
        return len(self.allocations)

    def __getitem__(self, var: Variable):
        return self.allocations[var.name]

    def __contains__(self, var: Variable):
        return var.name in self.allocations

    def append(self, var: Variable, offset: int = -1):
        if offset == -1:
            offset = self.size

        self.allocations[var.name] = Allocation(var, offset)

    @property
    def size(self) -> int:
        size = 0
        for a in self.allocations.values():
            size = max(a.offset + a.size, size)

        return size


class Allocator:
    layout: MemoryLayout

    @classmethod
    def allocate(cls, graph: Graph) -> MemoryLayout:
        variables = set(traverse.listup_variables(graph))
        for i, v in enumerate(variables):
            v.name = f"v{i}"

        return cls.allocate_variables(graph, list(variables))

    @classmethod
    def allocate_variables(cls, graph: Graph, variables: List[Variable]):
        ops = traverse.listup_operators(graph)
        layout = MemoryLayout()

        if graph_transpiler.util.flags.optimize.OPTIMIZE_MEMORY_ALLOCATION:
            analysis_list = _analyse_variable_lifetime(graph, ops, variables)

            _optimize_allocation_offset(analysis_list)

            allocation_dict = {item.variable: item.offset for item in analysis_list}
            for var in variables:
                original_var = var
                while "inplace_src" in var.parameters:
                    var = var.parameters["inplace_src"]

                layout.append(original_var, allocation_dict[var])

        else:
            for variable in variables:
                layout.append(variable)

        data = np.zeros(layout.size, dtype=np.float32)
        for var in variables:
            if not isinstance(var, ConstantVariable):
                continue
            allocation = layout[var]
            data[allocation.offset:allocation.offset + allocation.size] = var.data.flatten()

        layout.data = data

        if graph_transpiler.util.flags.VISUALIZE_MEMORY_ALLOCATION:
            _visualize_allocation(layout, graph, variables, ops)

        return layout


def _analyse_variable_lifetime(graph: Graph, ops: List[Operator], variables: List[Variable]):
    # 計算グラフを辿りながら、retain回数をカウントし、生存期間のテーブルを作成する

    LIFETIME_FOREVER = len(ops) + 1
    analysis_table: Dict[Variable, AllocationAnalysisData] = {}

    retain_count: Dict[Variable, int] = {v: 0 for v in variables}
    allocated: Set[Variable] = set()

    for var in variables:
        if isinstance(var, ConstantVariable):
            analysis_table[var] = AllocationAnalysisData(var, 0, LIFETIME_FOREVER)
            allocated.add(var)

    for var in graph.inputs:
        analysis_table[var] = AllocationAnalysisData(var, 0, LIFETIME_FOREVER)
        allocated.add(var)

    for t, op in enumerate(ops):
        for var in op.outputs.values():
            if var not in allocated:
                # 新規に確保する
                flag_allocated = False

                if graph_transpiler.util.flags.optimize.OPTIMIZE_INPLACE_OPERATION \
                    and not flag_allocated \
                    and traverse.check_attribute_match(op, Inplace):

                    # Inplace処理
                    input_variables = traverse.filter_nodes(op.inputs.values(), Constant, mode_not=True)
                    if len(input_variables) > 1:
                        if flags.DEBUG:
                            print(f"[WebGPUAllocator] Inplace operator with ambiguous memory location is detected. "
                                  + f"Allocator skipped inplace allocation and allocate other memory. "
                                  + f"Operator: {op}")

                    else:
                        # 入力のメモリをそのまま使う
                        v_in = input_variables[0]
                        while "inplace_src" in v_in.parameters:
                            v_in = v_in.parameters["inplace_src"]
                        var.parameters["inplace_src"] = v_in
                        retain_count[v_in] += len(var.input_to)

                        allocated.add(var)
                        flag_allocated = True

                if not flag_allocated:
                    # 新しくメモリを確保
                    analysis_table[var] = AllocationAnalysisData(var, t, LIFETIME_FOREVER)
                    retain_count[var] = len(var.input_to)

                    allocated.add(var)
                    flag_allocated = True

                if not flag_allocated:
                    raise ValueError("[WebGPUAllocator] Memory Allocation Failed.")

        for var in op.inputs.values():
            while "inplace_src" in var.parameters:
                var = var.parameters["inplace_src"]
            retain_count[var] -= 1

            if retain_count[var] == 0:
                # 解放はこの層が終わってから
                analysis_table[var].end = t + 1

    return [x for x in analysis_table.values()]


def _optimize_allocation_offset(analysis_list: List[AllocationAnalysisData]):
    analysis_list = sorted(analysis_list, key=lambda x: x.variable.size, reverse=True)
    analysis_list = sorted(analysis_list, key=lambda x: x.end)
    analysis_list = sorted(analysis_list, key=lambda x: x.end - x.start, reverse=True)
    analysis_list = sorted(analysis_list, key=lambda x: isinstance(x.variable, ConstantVariable), reverse=True)
    analysis_list = list(analysis_list)
    memory_offset_table = {}
    queue = list(analysis_list)

    while len(queue) > 0:
        for item1 in queue:
            offset = 0

            flag_retry = True
            while flag_retry:
                flag_retry = False

                for t in range(item1.start, item1.end):
                    if t not in memory_offset_table:
                        continue

                    for item2 in memory_offset_table[t]:
                        if item2.offset + item2.size <= offset or offset + item1.size <= item2.offset:
                            continue

                        else:
                            offset = item2.offset + item2.size
                            flag_retry = True
                            break

                    if flag_retry:
                        break

            item1.offset = offset

        queue = list(sorted(queue, key=lambda x: x.offset))
        item1 = queue.pop(0)
        for t in range(item1.start, item1.end):
            if t not in memory_offset_table:
                memory_offset_table[t] = []

            memory_offset_table[t].append(item1)


def _visualize_allocation(layout: MemoryLayout, graph: Graph, variables: List[Variable], ops: List[Operator]):
    UNIT_HEIGHT = 14
    analysis_list = _analyse_variable_lifetime(graph, ops, variables)
    total_size = layout.size

    class RenderingInfo:
        names: List[str]
        data: AllocationAnalysisData

        def __init__(self, data: AllocationAnalysisData):
            self.names = []
            self.data = data

        @property
        def offset(self):
            return layout[self.data.variable].offset

        @property
        def size(self):
            return layout[self.data.variable].size

        @property
        def top(self):
            return f"{self.data.start * UNIT_HEIGHT}px"

        @property
        def height(self):
            return f"{(self.data.end - self.data.start) * UNIT_HEIGHT + 1}px"

        @property
        def left(self):
            return f"{self.offset * 100 / total_size}%"

        @property
        def width(self):
            return f"calc({self.size * 100 / total_size}% + 1px)"

        # noinspection PyMethodMayBeStatic
        def generate_html(self):
            return f"""<div class="Allocation {"Constant" if isinstance(self.data.variable, ConstantVariable) else ""}"
style="top: {self.top}; height: {self.height}; left: {self.left}; width: {self.width}" title="{", ".join(self.names)}
size: {self.size}
offset: {self.offset}
lifetime: {self.data.start} - {self.data.end} 
">
    <p>{", ".join(self.names)}</p>
</div>"""

    allocation_dict = {item.variable: item for item in analysis_list}
    rendering_dict: Dict[Variable, RenderingInfo] = {}

    html = """<html>
<head>
    <style>
        html, body {
            margin: 0;
        }

        body {
            padding: 32px;
            box-sizing: border-box;
        }

        .MemoryLayout {
            position: relative;
            background: #888;
            font-size: 8px;
        }

        .Allocation {
            position: absolute;
            border: 1px solid #000;
            display: block;
            padding: 0;
            box-sizing: border-box;
            overflow: hidden;
            background: #0f0;
        }
        .Constant {
            background: #ff0;
        }
        p {
            margin: 0;
            white-space: nowrap;
        }
    </style>
</head>
<body>
<header style="margin-bottom: 32px;">
    <h1>Memory Allocation Visualization</h1>
    <div style="margin: 32px 0">
        <p>Total allocation size: """ + str(layout.size * 4) + """[byte]</p>
        <p># of allocated variables: """ + str(len(layout)) + """</p>
    </div>
    <div style="margin: 32px 0">
        <p>縦軸：時間経過（上から下へ）</p>
        <p>横軸：メモリアドレス</p>
        <p>各要素はカーソルホバーで詳細が見られます。</p>
    </div>
</header>
    <div class="MemoryLayout" style="height: """ + str(UNIT_HEIGHT * (len(ops) + 1) + 1) + """px;">
"""

    for variable in variables:
        original_var = variable
        while "inplace_src" in variable.parameters:
            variable = variable.parameters["inplace_src"]

        if variable not in rendering_dict:
            rendering_dict[variable] = RenderingInfo(allocation_dict[variable])

        rendering_dict[variable].names.append(original_var.name)

    for item in rendering_dict.values():
        html += item.generate_html()

    html += """
    </div>
</body>
</html>
"""

    with open('memory_visualize.html', "w+") as f:
        f.write(html)
