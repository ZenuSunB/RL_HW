hp_deficit=[]
rewards=[]
# 打开文件，'r'表示以只读模式打开
with open('/home/zenu/code/rl_hw/mindspore_ddpg2/eval_log.txt', 'r') as file:
    # 使用for循环逐行读取文件内容
    is_line_next_reward = False
    for line in file:
        # 打印每一行
        if "At the end," in line:
            line_list = line.strip().split()
            # print(line_list)
            hp_deficit.append(int(line_list[4][0:-1]) - int(line_list[6][0:-1]))
        if "can't receive signals" in line or "At the end," in line:
            is_line_next_reward= True
        elif is_line_next_reward :
            is_line_next_reward =False
            rewards.append(float(line))
            
print(hp_deficit)
print(len(hp_deficit))
with open('/home/zenu/code/rl_hw/ddpg_hp_deficit.txt', 'w') as file:
    # 写入字符串
    for hp in hp_deficit:
        file.write(str(hp)+"\n")
print(rewards)
print(len(rewards))
with open('/home/zenu/code/rl_hw/ddpg_eval_reward.txt', 'w') as file:
    # 写入字符串
    for reward in rewards:
        file.write(str(reward)+"\n")