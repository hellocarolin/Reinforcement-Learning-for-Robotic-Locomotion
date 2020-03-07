import matplotlib.pyplot as plt
import numpy as np

r = np.load('/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/quadruped/src/eps_rewards_base.npy')
r2 = np.load('/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/quadruped/src/eps_rewards_base_continued.npy')
r3 = np.load('/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/quadruped/src/eps_rewards_base_continued_friction1000.npy') 

r_super_random = np.load('/Users/caro/Uni/Master_Thesis_2019/gitMaster/MasterThesis2019/quadruped/src/eps_rewards_random_every_episode_1151eps.npy')
r_spa = np.concatenate((r,r2),axis=0)
r_friction = np.concatenate((r,r3),axis=0)

N = 50 
cumsum, cumsum2, moving_aves, moving_aves2 = [0], [0], [], []
cumsum4, moving_aves4 = [0], []

for i, x in enumerate(r_spa, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        moving_aves.append(moving_ave)

for i, x in enumerate(r_friction, 1):
    cumsum2.append(cumsum2[i-1] + x)
    if i>=N:
        moving_ave2 = (cumsum2[i] - cumsum2[i-N])/N
        moving_aves2.append(moving_ave2)

for i, x in enumerate(r_super_random, 1):
    cumsum4.append(cumsum4[i-1] + x)
    if i>=N:
        moving_ave4 = (cumsum4[i] - cumsum4[i-N])/N
        moving_aves4.append(moving_ave4)


r_fil_spa = np.array(moving_aves)
r_fil_friction = np.array(moving_aves2)
r_fil_super_random = np.array(moving_aves4)

fig = plt.figure(figsize=(10,7))
#plt.title('Episode Rewards')

plt.axis([0, 10000, -100, 40])

plt.tick_params(axis='x', labelsize=11)
plt.tick_params(axis='y', labelsize=11)

#plt.plot(r_spa,alpha=.5,label='episode rewards')
#plt.plot(r_friction,alpha=.5,label='episode rewards friction change')
#plt.plot(r_super_random,label='average episode rewards random')

plt.plot(r_fil_friction, label='random process every 10 episodes until episode 1000 + friction change')
plt.plot(r_fil_spa,label='random process every 10 episodes until episode 1000')
plt.plot(r_fil_super_random,label='random process every episode')

plt.legend(loc="lower right", title='average total rewards', fontsize=10, title_fontsize=11)

plt.ylabel('total rewards', fontsize=12)
plt.xlabel('episodes', fontsize=12)

plt.show()
