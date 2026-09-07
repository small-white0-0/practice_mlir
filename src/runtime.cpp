#include <cstdio>
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#include <type_traits>

/// 给my dialect一些 op 下降之后提供的运行时实现，编译my dialect输出的llvm ir时需要链接该静态库
namespace my_runtime {
    template<typename T>
    inline void printElement(const T &value) {
        if constexpr (std::is_same_v<T, float>) {
            std::printf("%f", value);
        } else if constexpr (std::is_same_v<T, double>) {
            std::printf("%lf", value);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            std::printf("%d", static_cast<int>(value));
        } else if constexpr (std::is_same_v<T, int16_t>) {
            std::printf("%d", static_cast<int>(value));
        } else if constexpr (std::is_same_v<T, int32_t>) {
            std::printf("%d", value);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            std::printf("%ld", static_cast<long>(value));
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            std::printf("%u", static_cast<unsigned>(value));
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            std::printf("%u", static_cast<unsigned>(value));
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            std::printf("%u", value);
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            std::printf("%lu", static_cast<unsigned long>(value));
        } else {
            static_assert(
                std::is_same_v<T, void>,
                "unsupported element type");
        }
    }

    template<typename T>
    void printMemRefRecursive(
        const DynamicMemRefType<T> &memref,
        int64_t dim,
        int64_t offset) {
        // 最后一维：打印元素
        if (dim == memref.rank - 1) {
            std::printf("[");

            for (int64_t i = 0; i < memref.sizes[dim]; ++i) {
                if (i != 0)
                    std::printf(", ");

                const int64_t elementOffset =
                        offset + i * memref.strides[dim];

                printElement(memref.data[elementOffset]);
            }

            std::printf("]");
            return;
        }

        std::printf("[");

        for (int64_t i = 0; i < memref.sizes[dim]; ++i) {
            if (i != 0)
                std::printf(",\n");

            const int64_t nextOffset =
                    offset + i * memref.strides[dim];

            printMemRefRecursive(
                memref,
                dim + 1,
                nextOffset);
        }

        std::printf("]");
    }

    template<typename T>
    void printMemRef(UnrankedMemRefType<T> *ptr) {
        if (!ptr) {
            std::printf("<null memref>\n");
            return;
        }

        DynamicMemRefType<T> memref(*ptr);

        if (memref.rank == 0) {
            printElement(memref.data[memref.offset]);
            std::printf("\n");
            return;
        }
        std::printf("rank = %ld\n",
                    static_cast<long>(memref.rank));

        std::printf("shape = [");

        for (int64_t i = 0; i < memref.rank; ++i) {
            if (i != 0)
                std::printf(", ");

            std::printf("%ld",
                        static_cast<long>(memref.sizes[i]));
        }

        std::printf("]\n");

        printMemRefRecursive(
            memref,
            0,
            memref.offset);

        std::printf("\n");
    }
} // namespace my_runtime
extern "C"
void _mlir_ciface_my_print_f32(
    UnrankedMemRefType<float> *ptr) {
    my_runtime::printMemRef(ptr);
}

extern "C"
void _mlir_ciface_my_print_f64(
    UnrankedMemRefType<double> *ptr) {
    my_runtime::printMemRef(ptr);
}

extern "C"
void _mlir_ciface_my_print_i32(
    UnrankedMemRefType<int32_t> *ptr) {
    my_runtime::printMemRef(ptr);
}

extern "C"
void _mlir_ciface_my_print_i64(
    UnrankedMemRefType<int64_t> *ptr) {
    my_runtime::printMemRef(ptr);
}
