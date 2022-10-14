# datas = []
#
# with open("testdata.txt", encoding="utf-8") as f:
#     for line in f.readlines():
#         line = line.split()
#         line_num = []
#         for i in line:
#             line_num.append(i)
#         datas.append(line_num)
#
# print(datas)
#
# with open("testdata.txt", "w+") as f:
#     for line in datas:
#         j = 0
#         for i in line:
#             f.write(i)
#             if j < 4:
#                 f.write(", ")
#             j = j + 1
#         f.write("\n")

# import pandas as pd
#
# datas = pd.read_csv("testdata.txt")
# print(type(datas.loc[0][1]))
