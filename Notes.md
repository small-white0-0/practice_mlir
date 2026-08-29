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

4. 在tablegen文件中突然很奇怪的报错，原因好像是mlir-tablegen的词法解析存在bug。
   > 报错：
   > ```
   > error: Unexpected token at top level
   > #endif
   > ^
   > ```
   > 实测，在`#endif`后面加上空白字符或换行符，就解决了。
   > 目前估计是mlir-tablegen的词法解析在文件的EOF附近的边界情况判断存在问题。
   
5. 补记: dialect中的`useDefaultTypePrinterParser`和`useDefaultAttributePrinterParser`没有特殊原因一定要开启。
   > 原因：因为type和attribute的`assemblyFormat`字段自动生成的方法会依赖默认的printerParser,
   > 如果不开启，后面的type和attribute的该字段就相当于用不了了。

6. mlir生成的Pass默认是没有命名空间的，所以在倒入生成代码时，要加上命名空间。
7. mlir的整数类型注意`iN`表示符号无关的`N`位整数，`siN`和`uiN` 才是分别表示有符号`N`位整数和无符号`N`位整数。
8. 突然发现自己因为跟着写，命名风格完全是混乱的。下次记得一开始记一下社区的命名风格。
9. `mlir::isa<mlir::BlockArgument>(operand)` 这个判断一定要记得做，因为如果是块参数，某些方法不是返回空，而是会导致段错误的。
10. 规范化的方法执行需要Pass进行，同时Pass添加的时候需要注意添加顺序。
    > 因为最初看教程说是规范化方法会自动执行，结果写完发现不执行。搜索了才知道还是要依靠pass执行，需要进行pass添加。
    > 下意识的根据根op添加了，如下:
    > ```
    > pm.addPass(my::createMarkDistributeParallelParametersPass({.DPNums = 3, .TPNums = 1}));
    > pm.addPass(mlir::createCanonicalizerPass());
    > pm.addNestedPass<mlir::func::FuncOp>(my::createApplyDistributeTransformPass());
    > ```
    > 结果还是不执行规范化方法。
    > 试了半天才注意到`buffer_cast`是`ApplyDistributeTransformPass`创建的，
    > 那么pass按照顺序执行，`buffer_cast`的规范化方法肯定不能执行了。
11. 多个op删除时，一定要使用倒序的删除顺序，不然死代码之间的use会导致前面的op无法删除。没有依赖关系的也建议这么做，形成良好习惯。
