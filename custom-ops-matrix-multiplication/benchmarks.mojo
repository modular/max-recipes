# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from benchmark import ThroughputMeasure, BenchId, BenchMetric, Bench, Bencher
from bit import log2_floor
from buffer.dimlist import DimList
from gpu.host import DeviceContext, DeviceBuffer
from math import iota
from max.driver import cpu
from max.tensor import (
    ManagedTensorSlice,
    InputTensor,
    OutputTensor,
    StaticTensorSpec,
    IOSpec,
    Input,
    Output,
    MutableInput,
)
from memory import AddressSpace
from memory import UnsafePointer
from operations.matrix_multiplication import MatrixMultiplication
from random import rand
from runtime.asyncrt import DeviceContextPtr
from sys import sizeof, has_nvidia_gpu_accelerator
from utils import IndexList


# Wrap a ManagedTensorSlice with a DeviceBuffer which has a lifetime to use
# Mojo's memory management, and sidestep the Python initialized garbage
# collected version.
@value
struct _BenchTensor[
    dtype: DType,
    rank: Int, //,
    io_spec: IOSpec,
    static_spec: StaticTensorSpec[dtype, rank],
]:
    alias tensor_type = ManagedTensorSlice[
        io_spec=io_spec, static_spec=static_spec
    ]
    alias buffer_type = DeviceBuffer[dtype]
    alias ptr_type = UnsafePointer[Scalar[dtype]]
    alias size = Int(static_spec.shape.product())

    var tensor: Self.tensor_type
    var buffer: Self.buffer_type

    fn __init__(out self, ctx: DeviceContext) raises:
        self.buffer = ctx.enqueue_create_buffer[dtype](Self.size)

        self.tensor = ManagedTensorSlice[
            io_spec=io_spec, static_spec=static_spec
        ](
            self.buffer.unsafe_ptr(),
            Self.static_spec.shape.into_index_list[rank](),
            Self.static_spec.strides.into_index_list[rank](),
        )

    fn unsafe_ptr(self) -> Self.ptr_type:
        return self.buffer.unsafe_ptr()

    fn rand(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            rand(host_buffer.unsafe_ptr(), Self.size)
            return self

    fn iota(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            iota(host_buffer.unsafe_ptr(), Self.size)
            return self


# TODO: Change StaticTensorSpec to use `IndexList` instead of `DimList` in order
# to determine strides from shape at compile time, and align with
# RuntimeTensorSpec.
fn _static_spec[
    dtype: DType, rank: Int
](shape: DimList, strides: DimList, out spec: StaticTensorSpec[dtype, rank]):
    spec = __type_of(spec)(
        shape=shape,
        strides=strides,
        alignment=sizeof[dtype](),
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
    )


def matmul():
    alias M = 1028
    alias K = 1028
    alias N = 1028

    alias rank = 2
    alias dtype = DType.float32

    alias FLOPS = M * N * (2 * K - 1)

    alias a_spec = _static_spec[dtype, rank](shape=(M, K), strides=(K, 1))
    alias b_spec = _static_spec[dtype, rank](shape=(K, N), strides=(N, 1))
    alias c_spec = _static_spec[dtype, rank](shape=(M, N), strides=(N, 1))

    var cpu_ctx = DeviceContext(api="cpu")

    var a = _BenchTensor[Input, a_spec](cpu_ctx).rand()
    var b = _BenchTensor[Input, b_spec](cpu_ctx).rand()
    var c = _BenchTensor[Output, c_spec](cpu_ctx).rand()

    var bench = Bench()
    var flops = ThroughputMeasure(BenchMetric.flops, FLOPS)
    var elements = ThroughputMeasure(BenchMetric.elements, M * N)

    @parameter
    @always_inline
    fn bench_cpu(mut bencher: Bencher) raises:
        @parameter
        @always_inline
        fn run_bench() raises:
            MatrixMultiplication["naive"].execute[target="cpu"](
                c.tensor, a.tensor, b.tensor, cpu_ctx
            )

        bencher.iter[run_bench]()

    bench.bench_function[bench_cpu](BenchId("cpu", "naive"), flops, elements)

    @parameter
    if has_nvidia_gpu_accelerator():
        var gpu_ctx = DeviceContext()
        var a_dev = _BenchTensor[Input, a_spec](gpu_ctx).rand()
        var b_dev = _BenchTensor[Input, b_spec](gpu_ctx).rand()
        var c_dev = _BenchTensor[Output, c_spec](gpu_ctx).rand()

        @parameter
        def bench_matmul_kernel[impl: StringLiteral]():
            @parameter
            @always_inline
            fn bench_gpu(mut bench: Bencher) raises:
                @parameter
                @always_inline
                fn kernel_launch(gpu_ctx: DeviceContext) raises:
                    MatrixMultiplication[impl].execute[target="gpu"](
                        c_dev.tensor, a_dev.tensor, b_dev.tensor, gpu_ctx
                    )

                bench.iter_custom[kernel_launch](gpu_ctx)

            bench.bench_function[bench_gpu](
                BenchId("gpu", impl), flops, elements
            )

        bench_matmul_kernel["naive"]()
        bench_matmul_kernel["coalescing"]()
        bench_matmul_kernel["tiled"]()
        bench_matmul_kernel["tiled_register"]()
        bench_matmul_kernel["block_tiled"]()
        bench_matmul_kernel["block_tiled_vectorized"]()
        bench_matmul_kernel["tensor_core"]()

    bench.config.verbose_metric_names = False
    print(bench)


def main():
    matmul()
