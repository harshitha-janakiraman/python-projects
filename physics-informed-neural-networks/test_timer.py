from project.timer import Timer

def xy():
    x = 12
    y = 21
    return x, y

ti = Timer(timer_repeat_times= 100000)
x, y = ti.time_average(lambda :  xy())
print(ti.str_average())
print(x)
print(y)
