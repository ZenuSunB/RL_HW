hp_deficit=[]

# 打开文件，'r'表示以只读模式打开
with open('/home/zenu/code/rl_hw/mindspore_ac/mindspore_ac/eval_log.txt', 'r') as file:
    # 使用for循环逐行读取文件内容
    for line in file:
        # 打印每一行
        if "At the end," in line:
            line_list = line.strip().split()
            print(line_list)
            hp_deficit.append(int(line_list[4][0:-1]) - int(line_list[6][0:-1]))
            
print(hp_deficit)
print(len(hp_deficit))
with open('/home/zenu/code/rl_hw/ac_hp_deficit.txt', 'w') as file:
    # 写入字符串
    for hp in hp_deficit:
        file.write(str(hp)+"\n")