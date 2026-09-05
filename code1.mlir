module @My {
  func.func @main() -> i32 {
    %0 = "my.constant"() <{value = dense<1.000000e+00> : tensor<128x128x24xf32>}> : () -> !my.my_tensor<128x128x24xf32,0>
    %1 = "my.softmax"(%0) <{axis = 1 : i64}> : (!my.my_tensor<128x128x24xf32,0>) -> !my.my_tensor<128x128x24xf32,0>
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }
}