import matplotlib.pyplot as plt
import math
def find_min_n_sin(t_max, delta=1e-3, safety_factor = 0.1):
    """Для sin(t): M = 1 всегда"""
    n = 0
    while True:
        term = (t_max ** (n+1)) / math.factorial(n+1)
        if term <= delta * safety_factor:
            return n
        n += 1
N = 50
list_t_max = [i for i in range(1, N + 1)]
error_list = [find_min_n_sin(value, delta=1e-3) for value in list_t_max]
plt.scatter(list_t_max, error_list, color='black', marker='o')
plt.xlabel('t_max') 
plt.ylabel('number') #номер начиная с которого можно отбрасывать 
plt.title('error(t_max)')
plt.savefig(r'D:\5sem\computational_mathematics\practical work\1_8_19\number(t_max).png')
plt.show()