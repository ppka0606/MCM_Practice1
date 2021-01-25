import numpy as np
import seaborn as sns
import random
from matplotlib import pyplot as plt

grass_growing_rate = 0.02
sheep_growing_rate = 0.03
sheep_eating_rate = 0.01

max_grass = 10000

grass_init_num = 2000
sheep_init_num = 30
grass_left_rate = 0.8
sim_days = 2000

grass_num_total_init = 7319.034852546864
sheep_num_total_init = 3903.4852546916477
grass_range_width = 10
grass_range_height = 10

sim_days_dragon = 2000
sheep_total_num = np.zeros(sim_days_dragon)
grass_total_num = np.zeros(sim_days_dragon)
sheep_total_num[0] = sheep_num_total_init
grass_total_num[0] = grass_num_total_init

sns.set()
grass_num = np.random.rand(grass_range_width, grass_range_height)
sheep_num = grass_num.copy()

grass_num /= np.sum(grass_num)
grass_num *= grass_num_total_init

sheep_num /= np.sum(sheep_num)
sheep_num *= sheep_num_total_init

pos = np.unravel_index(np.argmax(sheep_num),sheep_num.shape)
posx = pos[0]
posy = pos[1]

starving_days = 0 
starving_limit = 5 
starving_num = 0
starving_food_num = 34 

already_leave=False

for i in range(1, sim_days_dragon):
    max_food = grass_left_rate * grass_total_num[i-1]
    hold_num = max_food / sheep_eating_rate /150
    eat = min(max_food, sheep_eating_rate * sheep_total_num[i-1])
    sheep_num_temp = (1 + sheep_growing_rate*(hold_num-sheep_total_num[i-1])/hold_num) 
    * sheep_total_num[i-1]
    grass_num_temp = (1 + grass_growing_rate*(max_grass-grass_total_num[i-1])/max_grass) 
    * (grass_total_num[i-1] - eat)

    grass_num /= grass_total_num[i-1]
    grass_num *= grass_num_temp

    sheep_num = grass_num.copy()
    sheep_num /= np.sum(sheep_num)
    sheep_num *= sheep_num_temp

    max_food_posx = posx
    max_food_posy = posy
    
    if posx > 0 and sheep_num[posx-1][posy] > sheep_num[max_food_posx][max_food_posy]:
        max_food_posx = posx - 1
        max_food_posy = posy
    if posx < grass_range_height-1 and sheep_num[posx+1][posy] > 
    sheep_num[max_food_posx][max_food_posy]:
        max_food_posx = posx + 1
        max_food_posy = posy
    if posy > 0 and sheep_num[posx][posy-1] > sheep_num[max_food_posx][max_food_posy]:
        max_food_posx = posx
        max_food_posy = posy - 1
    if posy < grass_range_width-1 and sheep_num[posx][posy+1] > 
    sheep_num[max_food_posx][max_food_posy]:
        max_food_posx = posx
        max_food_posy = posy + 1
    
    posx = max_food_posx
    posy = max_food_posy

    if sheep_num[posx][posy] < starving_food_num:
        starving_num += (starving_food_num - sheep_num[posx][posy])
        sheep_num[posx][posy] = 0
    else:
        sheep_num[posx][posy] -= starving_food_num
        
        if sheep_num[posx][posy] >= starving_num:
            starving_days = 0
            starving_num = 0
            sheep_num[posx][posy] -= starving_num
        else:
            starving_num -= sheep_num[posx][posy]
            sheep_num[posx][posy] = 0
            
    if starving_num > 0:
        starving_days += 1
    if not already_leave:
        if starving_days >= starving_limit:
            print("Too hungry!Bye!")
            print("Days:", i)
            starving_food_num = 0
            starving_days = 0
            starving_num=0
            already_leave=True
    sheep_total_num[i] = np.sum(sheep_num)
    grass_total_num[i] = grass_num_temp



print(grass_num)
sns.heatmap(data=grass_num, cmap=sns.light_palette("#2ecc71", as_cmap=True),
                 xticklabels=[],yticklabels=[])
plt.show()

day = np.linspace(0, sim_days, sim_days)


fig = plt.figure()
ax = fig.add_subplot(111)
lns1 = ax.plot(day, grass_total_num[:sim_days],color="g",label="plants")
ax2 = ax.twinx()
lns2 = ax2.plot(day, sheep_total_num[:sim_days],color="b",label="animals")
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc="right",fontsize=14)

ax.grid()
ax.set_xlabel("time/d",fontsize=16)
ax.set_ylabel("relative quantity",fontsize=16)
plt.show()