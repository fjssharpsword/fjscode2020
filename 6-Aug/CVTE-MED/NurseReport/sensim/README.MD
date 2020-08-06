# Sentence similarity#

1.dutyModel.py
输入：文件，一条句子一行，放在data文件夹下
输出：保存模型
执行：python dutyModel.py --fileName med.csv

2.dutySim.py
输入：一条句子
输出：相似度最高的k条
执行：python dutySim.py --text '患者安静卧床休息'  --topK 10