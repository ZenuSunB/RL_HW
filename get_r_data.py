r_list=[]
loss_list=[]
with open('/home/zenu/code/rl_hw/mindspore_ddpg2/train_log.txt', 'r') as file:
    # 使用for循环逐行读取文件内容
    for line in file:
        # 打印每一行
        if "loss is" in line and  "rewards is" in line:
            line_list = line.strip().split()
            print(line_list)
            loss_list.append(float(line_list[4][0:-1]))
            r_list.append(float(line_list[7]))
            
print(len(loss_list))
print(len(r_list))
with open('/home/zenu/code/rl_hw/ddpg_reward.txt', 'w') as file:
    # 写入字符串
    for r in r_list:
        file.write(str(r)+"\n")
with open('/home/zenu/code/rl_hw/ddpg_loss.txt', 'w') as file:
    # 写入字符串
    for r in loss_list:
        file.write(str(r)+"\n")