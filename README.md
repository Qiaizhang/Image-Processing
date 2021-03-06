# OpenCV

![Image text](https://github.com/Qiaizhang/OpenCV/blob/master/%E6%BB%A4%E9%95%9C/processing.png)

主要针对数字图像进行常规处理：

（1）几何处理（Geometrical Image Processing）

	主要包括坐标变换，图像的放大、缩小、旋转、移动，多个图像配准，全景畸变校正，扭曲校正，周长、面积、体积计算等。
	

（2）算术处理（Arithmetic Processing）

	主要对图像施以 ＋、－、×、÷ 等运算，虽然该处理主要针对像素点的处理，但非常有用，如医学图像的减影处理就有显著的效果。
	

 (3）图像增强（Image Enhancement）

	就是突出图像中感兴趣的信息，而减弱或去除不需要的信息，从而使有用信息得到加强。

		○ 改善图像的视觉效果，提高图像成分的清晰度；

		○ 使图像变得更有利于计算机处理，便于进一步进行区分或解释。
		

（4）图像复原（或恢复）（Image Restoration）

	就是尽可能地减少或者去除图像在获取过程中的降质（干扰和模糊），恢复被退化图像的本来面貌，从而改善图像质量。

	关键是对每种退化（图像品质下降）建立一个合理的模型。
	

  (5）图像重建（Image Reconstruction）

	是从数据到图像的处理。即输入的是某种数据,而处理结果得到的是图像。典型应用有CT技术和三维重建技术。
	

（6）图像编码（Image Encoding）

	主要是利用图像信号的统计特性及人类视觉的生理学及心理学特征对图像信号进行高效编码，其目的是压缩数据量，以解决数据量大的矛盾。
	

 （7）图像识别（Image Recognition）

	利用计算机识别出图像中的目标并分类、用机器的智能代替人的智能。它所研究的领域十分广泛，如，机械加工中零部件的识别、分类；从遥感图片中分辨农作物、森林、湖泊和军事设施；从气象观测数据或气象卫星照片准确预报天气；从X光照片判断是否发生肿瘤；从心电图的波形判断被检查者是否患有心脏病；在交通中心实现交通管制、识别违章行驶的汽车及司机，等等。
	
  常见的算法主要有（陆续更新.....）：
  
  （1）简单相机滤镜：
  
        - 图像灰度化
        - 二值化实现黑白滤镜
        - 反向滤镜
        - 去色滤镜
        - 单色滤镜
        - 怀旧滤镜
        - 熔铸滤镜
        - 冰冻滤镜
        - 连环画滤镜
        - 浮雕雕刻特效
        - 素描
        - 羽化
        - PS扩散特效
        - 实现晕影vignetting效果
        
  （2）直方图：
  
  	    - 灰度直方图：
        - 颜色直方图：
	      - 掩模直方图：
	      - 灰度直方图均衡化：
	      - 颜色直方图均衡化：
        - 掩模直方图均衡化：
  
  （3）滤波器（图像平滑）：
  
  	       线性滤波：均值滤波、高斯滤波、盒子滤波、拉普拉斯滤波等等，通常线性滤波器之间只是模版系数不同。
           非线性滤波：利用原始图像跟模版之间的一种逻辑关系得到结果，如最值滤波器，中值滤波器和双边滤波器等。
           
  （4）阈值处理：	
  
          - 二值化阈值处理：
	        - 截断阈值处理：
	        - 超阈值零处理：
	        - 低阈值零处理：
          - 自适应阈值处理：
          
   （5）边缘检测：
   
          - roberts算子
          - Prewitt算子
          - Sobel算子
          - Scharr算子
          - Canny算子
          - Laplacian算子
          - 高斯拉普拉斯（LoG）
          - 高斯差分（DoG）
	        - Marr-Hildreth
  
