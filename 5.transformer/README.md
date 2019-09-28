### 任务五：建立一个transformer

这里主要分析如何建立一个transformer，为了避免代码过于复杂，这里先不用transformer解决一个实际问题，而且考虑到transformer是一个十分重要的模型，也有一定难度，仅仅从概念公式上理解是不足够的，所以这里也详细给出了transformer的各个构成部分的代码供学习。

1. 知识点：

   1. [self attention](<https://jesseyule.github.io/naturallanguage/selfAttention/content.html>)
   2. [transformer](<https://jesseyule.github.io/naturallanguage/transformer/content.html>)

### 代码说明

​	简单介绍一下代码结构，也可以把以下的介绍顺序作为分析代码的顺序。首先是encoderdecoder，主要建立编码解码模型，基于transformer，然后逐步实现编码器和解码器的各个构成部分，首先是注意力机制attention，基于注意力机制进一步构成multiHeadAttention，multiHeadAttention的输出会输入到positionwiseFeedForward中进行处理，而这两个结构就构成了解码器或者编码器中的一个子层，为了让模型能够识别出序列的位置，我们还需要positionEncoding，最后，transformer会把结果进行embedding，并交由generator进行处理。

​	说实话，我觉得transformer的模型概念理解起来不难，但是代码的实现对我来说还是有点复杂，可能上述的叙述有点偏差，欢迎指出，我也在进一步研究如何结合transformer进行实际的应用。







