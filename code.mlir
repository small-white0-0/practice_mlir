module @My {
  func.func @main(%arg0: !my.my_tensor<128x128x24xf32,0>) -> !my.my_tensor<128x128x24xf32,0> {
    %0 = "my.softmax"(%arg0) <{axis = 1 : i64}> : (!my.my_tensor<128x128x24xf32,0>) -> !my.my_tensor<128x128x24xf32,0>
    %1 = "my.softmax"(%0) <{axis = 1 : i64}> : (!my.my_tensor<128x128x24xf32,0>) -> !my.my_tensor<128x128x24xf32,0>
    return %1 : !my.my_tensor<128x128x24xf32,0>
  }
}