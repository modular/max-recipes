# Custom Operations: An Introduction to Programming GPUs and CPUs with Mojo

In this recipe, we will cover:

* How to extend a MAX Graph using custom operations.
* Using Mojo to write high-performance calculations that run on GPUs and CPUs.
* The basics of GPU programming in MAX.

We'll walk through running three examples that show

* adding one to every number in an input tensor
* performing hardware-specific addition of two vectors
* and calculating the Mandelbrot set on CPU and GPU.

Let's get started.

## Requirements

Please make sure your system meets our
[system requirements](https://docs.modular.com/max/get-started).

To proceed, ensure you have the `magic` CLI installed:

```bash
curl -ssL https://magic.modular.com/ | bash
```

or update it via:

```bash
magic self-update
```

### GPU requirements

These examples can all be run on either a CPU or GPU. To run them on a GPU,
ensure your system meets
[these GPU requirements](https://docs.modular.com/max/faq/#gpu-requirements):

* Officially supported GPUs: NVIDIA Ampere A-series (A100/A10), or Ada
  L4-series (L4/L40) data center GPUs. Unofficially, RTX 30XX and 40XX series
  GPUs have been reported to work well with MAX.
* NVIDIA GPU driver version 555 or higher. [Installation guide here](https://www.nvidia.com/download/index.aspx).

## Quick start

1. Download the code for this recipe using git:

```bash
git clone https://github.com/modular/max-recipes.git
cd max-recipes/custom-ops-introduction
```

2. Run each of the examples:

```bash
magic run add_one
magic run vector_addition
magic run mandelbrot
```

3. Browse through the commented source code to see how they work.

## Custom operation examples

Graphs in MAX can be extended to use custom operations written in Mojo. The
following examples are shown here:

* **add_one**: Adding 1 to every element of an input tensor.
* **vector_addition**: Performing vector addition using a manual GPU function.
* **mandelbrot**: Calculating the Mandelbrot set.

Custom operations have been written in Mojo to carry out these calculations. For
each example, a simple graph containing a single operation is constructed
in Python. This graph is compiled and dispatched onto a supported GPU if one is
available, or the CPU if not. Input tensors, if there are any, are moved from
the host to the device on which the graph is running. The graph then runs and
the results are copied back to the host for display.

One thing to note is that this same Mojo code runs on CPU as well as GPU. In
the construction of the graph, it runs on a supported accelerator if one is
available or falls back to the CPU if not. No code changes for either path.
The `vector_addition` example shows how this works under the hood for common
MAX abstractions, where compile-time specialization lets MAX choose the optimal
code path for a given hardware architecture.

The `operations/` directory contains the custom kernel implementations, and the
graph construction occurs in the Python files in the base directory. These
examples are designed to stand on their own, so that they can be used as
templates for experimentation.

The execution has two phases: first an `operations.mojopkg` is compiled from the
custom Mojo kernel, and then the graph is constructed and run in Python. The
inference session is pointed to the `operations.mojopkg` in order to load the
custom operations.

## Conclusion

In this recipe, we've introduced the basics of how to write custom MAX Graph
operations using Mojo, place them in a one-operation graph in Python, and run
them on an available CPU or GPU.

## Next Steps

* Follow [our tutorial for building a custom operation from scratch](https://docs.modular.com/max/tutorials/build-custom-ops).

* Explore MAX's [documentation](https://docs.modular.com/max/) for additional
  features. The [`gpu`](https://docs.modular.com/mojo/stdlib/gpu/) module has
  detail on Mojo's GPU programming functions and types, and the documentation
  on [`@compiler.register`](https://docs.modular.com/max/api/mojo-decorators/compiler-register/)
  shows how to register custom graph operations.

* Join our [Modular Forum](https://forum.modular.com/) and [Discord community](https://discord.gg/modular) to share your experiences and get support.

We're excited to see what you'll build with MAX! Share your projects and experiences with us using `#ModularAI` on social media.
