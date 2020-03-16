import numpy as np
import matplotlib.pyplot as plt

#slide 4
# the main things you need
# x = np.linspace(0, 5, 10)
# y = x ** 2

# plt.plot(x,y)
# plt.show()




#slide 5 title and label
x = np.linspace(0, 5, 10)
y = x ** 2

# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('title')
# plt.plot(x,y)
# plt.show()


#slide 5: try with y = x+1

# x = np.linspace(0,5,10)
# plt.plot(x,x+1)
# plt.show()

#further in slide 5: add x-1 and x+2
# x = np.linspace(0,5,10)
# plt.plot(x,x+1)
# plt.plot(x,x+2)
# plt.plot(x,x-1)
# plt.savefig('first.png',dpi=200) # for slide8
# plt.show()


# cont. modify color
# ax.plot(x, x+1, color="red", alpha=0.5) # half-transparant red
# ax.plot(x, x+2, color="chocolate")        # RGB hex code for a bluish color
# ax.plot(x, x+3, color="silver")        # RGB hex code for a greenish color
# plt.annotate('nastaran',xy=(1,3),xytext=(3,3),arrowprops={'color':'red'})

# plt.show()

#slide 7
# fig, axes = plt.subplots(figsize=(12,3))
# x = np.linspace(0, 5, 10)
# y = x ** 2
# axes.plot(x, y, 'r')
# axes.set_xlabel('x')
# axes.set_ylabel('y')
# axes.set_title('title');
# plt.show()



#slide 9 first start with simple double diagram
# x = np.linspace(0, 5, 10)
# y = x ** 2
# plt.subplot(1,2,1)
# plt.plot(x, y, 'r--')
# plt.subplot(1,2,2)
# plt.plot(y, x, 'g*-');
# plt.show()

# cont. another example
# more how to modify the lines
# MATLAB style line color and style 
# fig, ax = plt.subplots()
# ax.plot(x, x**2, 'b.-') # blue line with dots
# ax.plot(x, x**3, 'g--') # green dashed line
# plt.show()

#slide 10 line and marker style
# fig, ax = plt.subplots(figsize=(12,6))

# ax.plot(x, x+1, color="blue", linewidth=0.25)
# ax.plot(x, x+2, color="blue", linewidth=0.50)
# ax.plot(x, x+3, color="blue", linewidth=1.00)
# ax.plot(x, x+4, color="blue", linewidth=2.00)

# ax.plot(x, x+5, color="red", lw=2, linestyle='-')
# ax.plot(x, x+6, color="red", lw=2, ls='-.')
# ax.plot(x, x+7, color="red", lw=2, ls=':')

# # custom dash
# line, = ax.plot(x, x+8, color="black", lw=1.50)
# line.set_dashes([5, 10, 15, 10]) # format: line length, space length, ...

# # possible marker symbols: marker = '+', 'o', '*', 's', ',', '.', '1', '2', '3', '4', ...
# ax.plot(x, x+ 9, color="green", lw=2, ls='--', marker='+')
# ax.plot(x, x+10, color="green", lw=2, ls='--', marker='o')
# ax.plot(x, x+11, color="green", lw=2, ls='--', marker='s')
# ax.plot(x, x+12, color="green", lw=2, ls='--', marker='1')

# # marker size and color
# ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)
# ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)
# ax.plot(x, x+15, color="purple", lw=1, ls='-', marker='o', markersize=8, markerfacecolor="red")
# ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8, 
#         markerfacecolor="yellow", markeredgewidth=2, markeredgecolor="blue");
# plt.show()


# before slide 11 moving axis
# fig = plt.figure()
# axes = fig.add_axes([0.2, 0.2, 0.7, 0.7]) # left, bottom, width, height (range 0 to 1)
# axes.plot(x, y, 'r')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('title')
# plt.show()






#slide 11
# fig = plt.figure()

# axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
# axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes



# # main figure
# axes1.plot(x, y, 'r')
# axes1.set_xlabel('x')
# axes1.set_ylabel('y')
# axes1.set_title('title')

# # inset
# axes2.plot(y, x, 'g')
# axes2.set_xlabel('y')
# axes2.set_ylabel('x')
# axes2.set_title('inset title');
# plt.show()


# slide 11 legends: write a program to make the figure
# fig,axes = plt.subplots()
# x = np.linspace(0,5,10)
# axes.plot(x, x**2, label="y = x**2")
# axes.plot(x, x**3, label="y = x**3")
# axes.legend(loc=2); # upper left corner
# axes.set_xlabel('x')
# axes.set_ylabel('y')
# axes.set_title('title');
# plt.show()

#slide 13 latex
# fig, ax = plt.subplots()


import matplotlib


# matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'}) #later
# better one
# Update the matplotlib configuration parameters:
# matplotlib.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'}) #later

# ax.plot(x, x**2, label=r"$y = \alpha^2$")
# ax.plot(x, x**3, label=r"$y = \alpha^3$")
# ax.legend(loc=2) # upper left corner
# ax.set_xlabel(r'$\alpha$', fontsize=18)
# ax.set_ylabel(r'$y$', fontsize=18)
# ax.set_title('title');
# plt.show()

# slide 15 plot range
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# axes[0].plot(x, x**2, x, x**3)
# axes[0].set_title("default axes ranges")

# axes[1].plot(x, x**2, x, x**3)
# axes[1].axis('tight')
# axes[1].set_title("tight axes")

# axes[2].plot(x, x**2, x, x**3)
# axes[2].set_ylim([0, 60])
# axes[2].set_xlim([2, 5])
# axes[2].set_title("custom axes range");
# plt.show()


#slide 16 ticks

# fig, ax = plt.subplots(figsize=(10, 4))

# ax.plot(x, x**2, x, x**3, lw=2)

# ax.set_xticks([1, 2, 3, 4, 5])
# ax.set_xticklabels([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$'], fontsize=18)

# yticks = [0, 50, 100, 150]
# ax.set_yticks(yticks)
# ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=18); # use LaTeX formatted labels

# plt.show()


#slide 16 axis number and label spacing
# distance between x and y axis and the numbers on the axes
# matplotlib.rcParams['xtick.major.pad'] = 5
# matplotlib.rcParams['ytick.major.pad'] = 5

# fig, ax = plt.subplots(1, 1)
      
# ax.plot(x, x**2, x, np.exp(x))
# ax.set_yticks([0, 50, 100, 150])

# ax.set_title("label and axis spacing")

# # padding between axis label and axis numbers
# ax.xaxis.labelpad = 5
# ax.yaxis.labelpad = 5

# ax.set_xlabel("x")
# ax.set_ylabel("y")
# plt.show()


#slide 18 axis grid
# fig, axes = plt.subplots(1, 2, figsize=(10,3))

# # default grid appearance
# axes[0].plot(x, x**2, x, x**3, lw=2)
# axes[0].grid(True)

# # custom grid appearance
# axes[1].plot(x, x**2, x, x**3, lw=2)
# axes[1].grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
# plt.show()

#slide 18
# fig, ax = plt.subplots(figsize=(6,2))

# ax.spines['bottom'].set_color('blue')
# ax.spines['top'].set_color('blue')

# ax.spines['left'].set_color('red')
# ax.spines['left'].set_linewidth(2)

# # turn off axis spine to the right
# ax.spines['right'].set_color("none")
# ax.yaxis.tick_left() # only ticks on the left side
# plt.show()

#slide 20
# fig, ax = plt.subplots()

# xx = np.linspace(-0.75, 1., 100)
# ax.plot(xx, xx**2, xx, xx**3)

# ax.text(0.15, 0.2, r"$y=x^2$", fontsize=20, color="blue")
# ax.text(0.65, 0.1, r"$y=x^3$", fontsize=20, color="green");
# plt.show()


#slide 21

# fig, ax = plt.subplots()

# xx = np.linspace(-0.75, 1., 100)
# ax.plot(xx, xx**2, xx, xx**3)
# plt.annotate(r"$y=x^2$",fontsize=20,xy=(0,0),xytext=(0.2,0.2),arrowprops={'color':'blue'})
# plt.savefig('annotate.png',dpi=200)
# plt.show()


#slide 21 subplots
#first
# fig, ax = plt.subplots(2, 3)
# plt.show()

#second
# fig = plt.figure()
# ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
# ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
# ax3 = plt.subplot2grid((3,3), (1,2), rowspan=2)
# ax4 = plt.subplot2grid((3,3), (2,0))
# ax5 = plt.subplot2grid((3,3), (2,1))
# plt.show()

# #slide 26 other 2d plots
# n = np.array([0,1,2,3,4,5])
# xx = np.linspace(-0.75, 1., 100)

# fig, axes = plt.subplots(1, 4, figsize=(12,3))

# axes[0].scatter(xx, xx + 0.25*np.random.randn(len(xx)))
# axes[0].set_title("scatter")

# axes[1].step(n, n**2, lw=2)
# axes[1].set_title("step")

# axes[2].bar(n, n**2, align="center", width=0.5, alpha=0.5)
# axes[2].set_title("bar")

# axes[3].fill_between(x, x**2, x**3, color="green", alpha=0.5);
# axes[3].set_title("fill_between");

# plt.show()




#slide 29 exercise


# A histogram
print plt.style.available   #for slide 30
# plt.style.use('fivethirtyeight') #for slide 30
plt.style.use('seaborn-dark')
n = np.random.randn(100000)
fig, axes = plt.subplots(1, 2, figsize=(12,4))

axes[0].hist(n)
axes[0].set_title("Default histogram")
axes[0].set_xlim((min(n), max(n)))

axes[1].hist(n, cumulative=True, bins=50)
axes[1].set_title("Cumulative detailed histogram")
axes[1].set_xlim((min(n), max(n)));
plt.show()

# # slide 30
# from mpl_toolkits import mplot3d


# # fig = plt.figure()
# # ax = plt.axes(projection="3d")

# # plt.show()

# # 3d more
# fig = plt.figure()
# ax = plt.axes(projection="3d")

# z_line = np.linspace(0, 15, 1000)
# x_line = np.cos(z_line)
# y_line = np.sin(z_line)
# ax.plot3D(x_line, y_line, z_line, 'gray')

# plt.show()
