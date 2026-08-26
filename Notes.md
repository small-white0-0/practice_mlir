## 遇到的问题

1. 将LLVM库和项目一起编译时，会出现一些名称冲突或其他很奇怪的编译错误。
   > 解决方法是把LLVM库单独编译，然后在项目中链接已经编译好的LLVM库。

2. 编译时mlir-tablegen报错，`belonging to more than one dialect. Must select one via '--(attr|type)defs-dialect'`
   > 原因是因为include导入了其他的dialect定义，导致mlir-tablegen无法确定该生成哪个dialect的Types定义。
   > 解决方法就是移除多余的include,或者在命令行中指定`--attrdefs-dialect`或`--typedefs-dialect`参数来明确指定使用哪个dialect。

3. 在`TypeBuilder`中错误的使用`$`符号，导致生成的代码中参数使用存在问题，导致编译错误。
   > 因为TypeBuilder的第二个参数是一个字符串模板，使用`$`符号会被误认为是模板变量，导致生成的代码中参数使用不正确。
   > 正确的方式就是直接使用`dag`定义的参数名称去除`$`符号即可。
   > 注：
   > 在`Builder`中写的代码也可能会被直接展开复制到`get`和`getChecked`方法中，
   > 所以如果要使用使用`Base::get`方法应该使用 `$_get` 来调用。
   > ```
   > AttrOrTypeBuilder<(ins "int":$integerArg, CArg<"float", "3.0f">:$floatArg), [{
   >   return $_get($_ctxt, integerArg, floatArg);
   > }]>
   > ```