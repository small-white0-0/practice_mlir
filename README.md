# practice_mlir

**A device-aware MLIR compiler prototype with compile-time data-parallel partitioning, device region fusion, interface-driven transformation, and end-to-end LLVM lowering.**

`practice_mlir` 是一个基于 **MLIR/LLVM** 构建的编译器原型。项目通过自定义 `My` Dialect，探索设备感知 IR、编译期数据并行、Region Fusion、Dialect Conversion 以及多阶段 LLVM Lowering 的完整实现流程。

项目核心包括：

* Device-aware Tensor Type
* Data Parallelism Attribute
* Interface-driven Transformation
* Compile-time Data Parallel Partitioning
* Buffer Scatter / Gather
* Device Region Fusion
* Custom Dialect Conversion
* Type Conversion
* Tensor / Linalg / Bufferization / MemRef / SCF / LLVM 多阶段 Lowering
* Compiler Runtime Support
* MLIR → LLVM IR → Native Executable

> **Project Scope**
>
> 当前项目主要实现**编译期设备感知 IR 与数据并行编译原型**。
>
> `DeviceKernelOp` 用于表示设备相关计算 Region，并在 Lowering 阶段转换为 `func.func` / `func.call`。
>
> 当前尚未实现实际硬件设备调度、设备选择以及 Runtime Device Dispatch。

---

# 1. Architecture

整体编译架构：

```text
                         High-Level My IR
                                │
                ┌───────────────┴───────────────┐
                │                               │
        Device-aware Types              Custom Operations
          MyTensorType                  Constant / Softmax
                │                       Exp / Add / ...
                │                       Print / BufferCast
                │                       DeviceRegion / Return
                └───────────────┬───────────────┘
                                │
                                ▼
                    Data Parallelism Attribute
                                │
                                ▼
                  Interface-driven Transformation
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
           Device 0          Device 1        Device N
         compile-time      compile-time     compile-time
          partition         partition        partition
                │               │               │
                └───────────────┼───────────────┘
                                │
                                ▼
                     Device Region Fusion
                                │
                                ▼
                       DeviceKernelOp
                       / device_region
                                │
                                ▼
                   Device-aware Function Lowering
                         func.func / func.call
                                │
                                ▼
                          My → Builtin
                                │
                                ▼
                       Tensor / Linalg Lowering
                                │
                                ▼
                          Bufferization
                                │
                                ▼
                             MemRef
                                │
                                ▼
                            SCF / CF
                                │
                                ▼
                              LLVM
                                │
                                ▼
                            LLVM IR
                                │
                                ▼
                           clang++
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
             Generated Code          Runtime Support
                                        libRuntime
                    │                       │
                    └───────────┬───────────┘
                                ▼
                         Native Executable
```

其中：

```text
Device 0 ... Device N
```

表示**编译期 IR 中的数据划分与设备标识**，并不代表当前项目已经能够将计算实际提交到对应物理设备。

---

# 2. Device-aware IR

项目扩展 MLIR Type System，定义带有设备信息的 Tensor Type：

```text
MyTensorType
├── Shape
├── Element Type
└── Device ID
```

其中 `Device ID` 用于在编译期 IR 中标识 Tensor 所属的数据分区。

示例：

```mlir
!my.my_tensor<4x128xf32, device = 0>
!my.my_tensor<4x128xf32, device = 1>
```

`MyTensorType` 实现 `ShapedTypeInterface`，使 Shape、Element Type 和 Device ID 能够参与后续编译期 Transformation。

Device information 主要用于：

* IR Verification
* Data Partitioning
* Scatter / Gather
* Device Region Formation
* Region Fusion
* Function Lowering

进入标准 MLIR Tensor Type 后，Device ID 不再作为标准 Tensor Type 的组成部分，而由此前形成的 Device Region 保留设备相关计算边界。

---

# 3. Data Parallelism

项目定义自定义 `DataParallelism` Attribute，用于描述编译期数据并行配置：

```text
DataParallelism
├── dpNum
└── deviceIds
```

例如：

```text
DP = 3
Device IDs = [0, 1, 2]
```

对应的 Attribute Interface：

```text
DataParallelAttrInterface
├── getDP()
└── getDeviceIds()
```

整体 Transformation Pipeline：

```text
MarkDistributeParallelParametersPass
                │
                ▼
        DataParallelism Attribute
                │
                ▼
ApplyDistributeTransformPass
                │
                ▼
   DistributeParallelOpInterface
                │
                ▼
      Operation-specific
       Distribution Logic
```

`MarkDistributeParallelParametersPass` 为相关函数参数建立数据并行配置。

随后 `ApplyDistributeTransformPass` 通过 Interface 判断 Operation 是否支持数据并行，并调用：

```cpp
supportsDistribution(...)
applyDistribution(...)
```

将具体 Transformation 逻辑交给 Operation 自身实现。

这种设计降低了 Pass 与具体 Operation 之间的耦合。

---

# 4. Compile-time Data Parallel Partitioning

以 `SoftmaxOp` 为例，原始计算：

```text
Input
  │
  ▼
Softmax
  │
  ▼
Output
```

经过 Data Parallel Transformation：

```text
                         Input
                           │
                           ▼
                         Scatter
                    ┌──────┼──────┐
                    ▼      ▼      ▼
                Device 0 Device 1 Device 2
                    │      │      │
                    ▼      ▼      ▼
                 Softmax Softmax Softmax
                    │      │      │
                    └──────┼──────┘
                           ▼
                         Gather
                           │
                           ▼
                         Output
```

对应实现位于：

```cpp
SoftmaxOp::applyDistribution()
```

主要步骤：

1. 根据 `DPNum` 计算 Batch Partition；
2. 为每个 Device 创建对应的 `MyTensorType`；
3. 创建 Scatter `BufferCastOp`；
4. 在各 Device 上生成独立的 `SoftmaxOp`；
5. 创建 Gather `BufferCastOp`；
6. 替换原始 Operation 的 SSA Uses。

## Remainder-aware Partitioning

当前实现支持 Batch 无法被设备数量整除的情况。

例如：

```text
Batch = 10
DP = 3
```

得到：

```text
Device 0 → 4
Device 1 → 3
Device 2 → 3
```

即：

```text
[4, 3, 3]
```

因此 Partitioning 不仅支持固定大小切分，还处理了 Batch Partition 中的 remainder。

---

# 5. Interface-driven Transformation

项目使用 MLIR Interface 将 Transformation 能力与具体 Operation 解耦。

## 5.1 Distribution Interface

核心 Operation Interface：

```text
DistributeParallelOpInterface
```

包含：

```cpp
supportsDistribution(...)
applyDistribution(...)
```

分别用于：

* 判断 Operation 是否支持指定并行策略；
* 执行对应的数据并行 Transformation。

整体关系：

```text
ApplyDistributeTransformPass
          │
          ▼
DistributeParallelOpInterface
          │
      ┌───┴────┐
      ▼        ▼
 SoftmaxOp   Other Ops
      │
      ▼
applyDistribution()
```

Pass 不需要直接依赖具体 Operation 的 Transformation 实现。

---

## 5.2 Attribute Interface

项目同时定义：

```text
DistributeParallelAttrInterface
```

以及继承它的：

```text
DataParallelAttrInterface
```

后者提供：

```cpp
int64_t getDP();

ArrayRef<int64_t> getDeviceIds();
```

因此 Transformation 可以通过 Interface 获取并行配置，而不是依赖具体 Attribute 的内部实现。

---

## 5.3 Fusion Interface

项目定义：

```text
FusionRegionOpInterface
```

用于描述 Region Fusion 能力：

```cpp
FusionOps(
    RewriterBase &rewriter,
    ArrayRef<Operation *> ops,
    Location loc
)
```

该 Interface 为后续 Device Region Fusion 提供统一的 Operation 扩展机制。

---

# 6. Device Region Fusion

数据划分后，同一设备上的计算可能仍然以多个独立 Operation 存在。

项目通过：

```text
DeviceRegionFusionPass
```

以及：

```text
BufferCastOpDeviceRegionFusion
```

将相关计算组织成明确的 Device Region。

主要流程：

1. 找到 Scatter 后的计算数据流；
2. 沿 Use-Def Chain 收集相关 Operation；
3. 判断 Operation 是否满足对应 Interface；
4. 根据 Device ID 确定计算归属；
5. 将属于同一设备的计算区域进行融合；
6. 创建 `DeviceKernelOp`；
7. 将原始 Operation Clone 到 Kernel Region；
8. 重建 Block Arguments；
9. 重建 SSA Use-Def；
10. 创建 `ReturnOp`；
11. 删除原始计算 Operation。

最终形成：

```text
my.device_region
┌─────────────────────────────────┐
│                                 │
│      Block Argument             │
│            │                    │
│            ▼                    │
│        SoftmaxOp                │
│            │                    │
│            ▼                    │
│          ExpOp                  │
│            │                    │
│            ▼                    │
│         ReturnOp                │
│                                 │
└─────────────────────────────────┘
```

通过 Region Fusion，原本分散的数据并行计算被组织为具有明确设备边界的计算 Region。

---

# 7. Device-aware Function Lowering

`DeviceKernelOp` 对应的 MLIR Operation 名称为：

```text
my.device_region
```

它用于表达：

```text
Device ID
+
Device-specific Computation Region
```

当前并不负责实际设备 Runtime Launch。

Lowering 后：

```text
DeviceKernelOp
      │
      ▼
  func.func
      │
      ▼
  func.call
```

因此当前实现完成的是：

> **将设备相关计算 Region 从自定义 Dialect Lowering 为标准 MLIR Function / Call。**

这建立了：

```text
Device-aware IR
       │
       ▼
Device Region
       │
       ▼
Function Boundary
       │
       ▼
LLVM Codegen
```

之间的连接。

实际的：

```text
Hardware Scheduling
Device Selection
Runtime Device Dispatch
GPU Launch
Remote Execution
```

不属于当前项目实现范围。

---

# 8. Custom My Dialect

项目通过 TableGen 定义：

```text
MyDialect
```

完整 IR Components：

```text
MyDialect
├── Type
│   └── MyTensorType
│
├── Attribute
│   └── DataParallelism
│
├── Attribute Interfaces
│   ├── DistributeParallelAttrInterface
│   └── DataParallelAttrInterface
│
├── Operation Interfaces
│   ├── DistributeParallelOpInterface
│   └── FusionRegionOpInterface
│
└── Operations
    ├── ConstantOp
    ├── SoftmaxOp
    ├── ExpOp
    ├── AddOp
    ├── SubOp
    ├── MulOp
    ├── DivOp
    ├── PrintOp
    ├── DeviceKernelOp
    ├── BufferCastOp
    └── ReturnOp
```

## Operation Categories

### Tensor Operations

```text
ConstantOp
SoftmaxOp
ExpOp
AddOp
SubOp
MulOp
DivOp
```

用于表达基础 Tensor Computation。

### Data Movement

```text
BufferCastOp
```

用于表达编译期 Data Parallel Transformation 中的 Scatter / Gather。

### Device Region

```text
DeviceKernelOp
```

用于表示设备相关计算 Region。

### Runtime-related Operation

```text
PrintOp
```

用于输出生成代码中的 Tensor / MemRef 数据。

### Region Terminator

```text
ReturnOp
```

用于表示 Device Region 内部的计算返回值。

---

# 9. IR Verification

项目针对自定义 IR 实现 Verification。

当前 `SoftmaxOp` 对 `axis` 进行合法性检查：

```text
axis ∈ [-rank, rank)
```

整体流程：

```text
Custom IR
    │
    ▼
Verification
    │
    ▼
Transformation
    │
    ▼
Dialect Conversion
```

通过在 Transformation 前进行基本 IR Correctness 检查，避免非法 IR 进入后续编译阶段。

---

# 10. BufferCast

`BufferCastOp` 是 Data Parallel Transformation 中的数据划分与结果合并 Operation。

## Scatter

```text
Tensor
  │
  ▼
BufferCast
  │
  ├── Tensor<Device 0>
  ├── Tensor<Device 1>
  └── Tensor<Device N>
```

用于根据 Data Parallel Partition 将逻辑 Tensor 划分为多个 Device-specific Tensor。

## Gather

```text
Tensor<Device 0>
Tensor<Device 1>
Tensor<Device N>
        │
        ▼
    BufferCast
        │
        ▼
      Tensor
```

用于将多个 Device-specific Tensor 重新组装为逻辑 Tensor。

## Lowering

`BufferCastOp` Lowering 为标准 Tensor Operations：

```text
my.buffer_cast
      │
      ├── tensor.extract_slice
      ├── tensor.insert_slice
      └── tensor.empty
```

分别用于：

* Tensor Slice Extraction
* Tensor Slice Insertion
* Tensor Creation

## Canonicalization

`BufferCastOp` 提供自定义 Canonicalization，用于：

* 删除 Dead BufferCast；
* 消除等价的连续 BufferCast；
* 简化 Transformation 后产生的冗余 IR。

---

# 11. Dialect Conversion

My Dialect 通过 MLIR **Dialect Conversion Framework** 转换到标准 MLIR Dialect。

核心组件：

```text
ConversionTarget
TypeConverter
OpConversionPattern
applyPartialConversion
```

主要 Conversion：

```text
MyTensorType
      │
      ▼
RankedTensorType
```

```text
My::ConstantOp
      │
      ▼
arith.constant
```

```text
My::SoftmaxOp
      │
      ▼
linalg.softmax
      │
      ▼
linalg.generic
```

```text
My::BufferCastOp
      │
      ├── tensor.extract_slice
      └── tensor.insert_slice
```

```text
My::DeviceKernelOp
      │
      ▼
func.func / func.call
```

```text
My::ReturnOp
      │
      ▼
func.return
```

该阶段完成自定义高层 IR 到标准 MLIR IR 的结构化转换，为后续标准 Lowering Pipeline 提供输入。

---

# 12. Type Conversion

自定义 Tensor Type：

```text
!my.my_tensor<...>
```

转换为标准：

```text
tensor<...>
```

即：

```text
MyTensorType
      │
      ▼
RankedTensorType
```

转换过程中：

```text
Shape        → 保留
Element Type → 保留
Device ID    → 不再属于标准 Tensor Type
```

Device ID 在进入标准 Tensor Dialect 前，已经通过：

```text
MyTensorType
     ↓
Data Parallel Transformation
     ↓
Device Region
     ↓
DeviceKernelOp
```

完成设备相关信息的显式建模。

因此标准 Tensor / Linalg / Bufferization Pipeline 可以继续处理 Tensor 数据，而设备计算边界由 `DeviceKernelOp` 保留。

---

# 13. End-to-End Lowering Pipeline

项目实现从自定义 Dialect 到 LLVM IR 的多阶段 Lowering：

```text
My Dialect
    ↓
Builtin / Tensor / Linalg
    ↓
Bufferization
    ↓
MemRef
    ↓
Loops
    ↓
SCF
    ↓
Control Flow
    ↓
LLVM Dialect
    ↓
LLVM IR
```

主要 Pipeline：

```text
My → Builtin
      ↓
Reconcile Unrealized Casts
      ↓
Canonicalizer
      ↓
CSE
      ↓
Tensor → Linalg
      ↓
Linalg Generalize Named Ops
      ↓
Canonicalizer
      ↓
CSE
      ↓
One-Shot Bufferize
      ↓
Canonicalizer
      ↓
CSE
      ↓
Linalg → Loops
      ↓
Canonicalizer
      ↓
CSE
      ↓
SCF → Control Flow
      ↓
Canonicalizer
      ↓
CSE
      ↓
Control Flow → LLVM
      ↓
Arith → LLVM
      ↓
Index → LLVM
      ↓
Math → LLVM
      ↓
Func → LLVM
      ↓
Finalize MemRef → LLVM
      ↓
Reconcile Unrealized Casts
      ↓
Canonicalizer
      ↓
CSE
```

最终生成：

```text
build/output.ll
```

---

# 14. Runtime Support

项目提供独立的 **Runtime Support Library**，用于支持 Lowering 后生成代码中的 `my.print` Operation。

调用链：

```text
my.print
   ↓
LLVM IR
   ↓
_mlir_ciface_my_print_*
   ↓
libRuntime
   ↓
MemRef data
   ↓
printf
```

当前 Runtime 提供：

```text
_mlir_ciface_my_print_f32
_mlir_ciface_my_print_f64
_mlir_ciface_my_print_i32
_mlir_ciface_my_print_i64
```

最终链接时：

```bash
clang++ ./build/output.ll \
    -L./build/src \
    -lruntime \
    -o ./build/a.out
```

> `libRuntime` 属于 **Compiler Runtime Support**，用于支持生成代码运行；当前不负责设备调度、设备选择或 Runtime Device Dispatch。

---

# 15. End-to-End Example

一个简单的 My Dialect 计算：

```mlir
func.func @main() {
  ...
  %result = my.softmax %input
  ...
}
```

经过编译期 Data Parallel Transformation：

```text
                     Input
                       │
                       ▼
                    Scatter
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
          Device 0  Device 1  Device 2
              │        │        │
              ▼        ▼        ▼
           Softmax   Softmax   Softmax
              │        │        │
              └────────┼────────┘
                       ▼
                     Gather
```

然后进行 Device Region Fusion：

```text
Data Parallel Operations
        │
        ▼
Device Region Fusion
        │
        ▼
DeviceKernelOp
        │
        ▼
func.func / func.call
```

继续进行标准 Lowering：

```text
My Dialect
    ↓
Tensor / Linalg
    ↓
Bufferization
    ↓
MemRef
    ↓
SCF / Control Flow
    ↓
LLVM
    ↓
LLVM IR
```

最终：

```text
LLVM IR
   │
   ▼
clang++
   │
   ├───────────────┐
   │               │
   ▼               ▼
Generated Code   libRuntime
   │               │
   └───────┬───────┘
           ▼
        Native Binary
```

---

# 16. Project Structure

```text
practice_mlir/
├── src/
│   ├── IR/
│   │   ├── MyDialect.cpp
│   │   ├── MyTypes.cpp
│   │   ├── MyAttrs.cpp
│   │   ├── MyOps.cpp
│   │   ├── MyOpInterfaces.cpp
│   │   └── MyAttrInterface.cpp
│   │
│   ├── Transforms/
│   │   └── MyPasses.cpp
│   │
│   ├── Conversion/
│   │   ├── MyConversionPass.cpp
│   │   └── MyToBuiltin.cpp
│   │
│   ├── tools/
│   │   └── my-opt.cpp
│   │
│   ├── runtime.cpp
│   └── main.cpp
│
├── third_party/
│   └── MLIR
│
├── CMakeLists.txt
├── build_and_run.sh
├── code.mlir
├── code1.mlir
└── Notes.md
```

---

# 17. Build & Run

## Requirements

* C++20 compatible compiler
* CMake
* LLVM / MLIR
* Clang

Clone repository:

```bash
git clone https://github.com/small-white0-0/practice_mlir.git
cd practice_mlir
git submodule update --init --recursive
```

Run:

```bash
./build_and_run.sh
```

脚本负责：

1. Build and install MLIR；
2. Configure and build `practice_mlir`；
3. Build Runtime Support Library；
4. Execute built-in tests；
5. Run end-to-end MLIR → LLVM pipeline；
6. Generate `build/output.ll`；
7. Link Runtime Support Library；
8. Generate Native Executable；
9. Execute generated binary。

---

## Manual Execution

运行完整 Pipeline：

```bash
./build/src/practice_mlir 4 ${CODE_MLIR} ./build/output.ll
```

生成：

```text
build/output.ll
```

链接 Runtime：

```bash
clang++ ./build/output.ll \
    -L./build/src \
    -lruntime \
    -o ./build/a.out
```

运行：

```bash
./build/a.out
```

---

# 18. End-to-End Demo

## Demo 1 — IR Construction

```bash
practice_mlir 1
```

展示：

* Custom Type
* Custom Attribute
* Custom Operation
* MLIR IR Construction

---

## Demo 2 — Transformation

```bash
practice_mlir 2
```

展示：

* Data Parallel Attribute
* Compile-time Data Partitioning
* Buffer Scatter / Gather
* Device Region Fusion

---

## Demo 3 — Dialect Conversion

```bash
practice_mlir 3
```

展示：

* Data Parallel Transformation
* Device Region Fusion
* My Dialect → Standard MLIR Conversion

---

## Demo 4 — Complete Pipeline

```bash
practice_mlir 4
```

展示：

```text
MLIR Input
    ↓
Data Parallel Distribution
    ↓
Device Region Fusion
    ↓
DeviceKernelOp
    ↓
My → Builtin
    ↓
Tensor / Linalg
    ↓
Bufferization
    ↓
MemRef / SCF / CF
    ↓
LLVM
    ↓
LLVM IR
    ↓
clang++
    ↓
libRuntime
    ↓
Native Executable
```

---

# 19. Technical Highlights

## IR Design

* Custom MLIR Dialect
* Custom Type / Attribute / Operation
* Device-aware Tensor Type
* Device ID in Tensor Type
* Data Parallelism Attribute
* Operation Interface
* Attribute Interface
* Region Fusion Interface

## Transformation

* Compile-time Data Parallel Partitioning
* Remainder-aware Batch Partitioning
* Scatter / Gather
* Interface-driven Transformation
* Greedy Rewrite Pattern
* Device Region Fusion
* Block Argument Reconstruction
* SSA Use-Def Reconstruction

## Dialect Conversion

* MLIR Dialect Conversion Framework
* `ConversionTarget`
* `TypeConverter`
* `OpConversionPattern`
* `applyPartialConversion`
* Unrealized Cast Materialization
* Custom Type Conversion

## Lowering

* My Dialect → Builtin
* Tensor → Linalg
* Linalg Named Op Generalization
* One-Shot Bufferization
* Linalg → Loops
* SCF → Control Flow
* Control Flow → LLVM
* Arith → LLVM
* Index → LLVM
* Math → LLVM
* Func → LLVM
* MemRef → LLVM
* LLVM IR Generation

## Runtime Support

* Custom Compiler Runtime Support
* C ABI Runtime Entry Points
* MLIR MemRef ABI
* LLVM IR / Runtime Library Linking

---

# 20. Resume Summary

> **基于 MLIR/LLVM 实现设备感知编译器原型，自定义 Tensor Type、Data Parallelism Attribute 及相关 Interface，基于 Interface 实现编译期数据并行划分、Scatter/Gather 与 Device Region Fusion，并将设备计算 Region Lowering 为标准函数调用；进一步通过 Dialect Conversion、Bufferization 和多阶段 Lowering Pipeline 将自定义 Dialect 编译至 LLVM IR，并结合 Runtime Support 完成 Native Executable 生成。**
