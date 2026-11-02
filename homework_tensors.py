import torch
import time

# 1.1 Создание тензоров
rand_tensor = torch.randint(0, 2, (3,4))
zeros_tensor = torch.zeros((2,3,4))
ones_tensor = torch.ones(5,5)
range_tensor = torch.arange(0, 16).reshape(4,4)

# 1.2 Операции с тензорами
A = torch.randint(0,11,(3,4))
B = torch.randint(0,11,(4,3))

A_t = A.T
matmul_AB = A@B
mul_ABt = A*B.T
sum_A = A.sum()

# 1.3 Индексация и срезы
slicing_tensor = torch.arange(125).reshape(5,5,5)

first_row = slicing_tensor[0, 0, :]
last_column = slicing_tensor[:, :, -1]
center = slicing_tensor[2, 2:4, 2:4]
even_indices = slicing_tensor[::2, ::2, ::2]

# 1.4 Работа с формами
reshape_tensor = torch.arange(24)

x_2x12 = reshape_tensor.reshape(2, 12)
x_3x8 = reshape_tensor.reshape(3, 8)
x_4x6 = reshape_tensor.reshape(4, 6)
x_2x3x4 = reshape_tensor.reshape(2, 3, 4)
x_2x2x2x3 = reshape_tensor.reshape(2, 2, 2, 3)

# 2.1 Простые вычисления с градиентами
x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)
z = torch.tensor([5.0], requires_grad=True)

f = x**2 + y**2 + 2*x*y*z
f.backward()
print(f"x: {x.grad}") # 32
print(f"y: {y.grad}") # 16
print(f"z: {z.grad}") # 6
print()

# 2.2 Градиент функции потерь
w = torch.tensor([0.5, 0.4, 0.1], requires_grad=True)
b =  torch.tensor([3.0], requires_grad=True)

x = torch.tensor([10.0, 11.0, 2.0])
y_true = torch.tensor([12.0])

y_pred = w@x + b

MSE = (y_pred-y_true)**2

MSE.backward()

print(f"w: {w.grad}")
print(f"b: {b.grad}")
print()

# 2.3 Цепное правило

x = torch.tensor([3.0], requires_grad=True)

y = torch.sin(x**2 + 1)

grad_analytic = torch.cos(x ** 2 + 1) * 2 * x
grad_autograd = torch.autograd.grad(y, x)

print(f"grad_analytic: {grad_analytic}")
print(f"auto_grad: {grad_autograd}")

# 3.1 Подготовка данных
big_matrix_1 = torch.rand((64,1024,1024))
big_matrix_2 = torch.rand((128,512,512))
big_matrix_3 = torch.rand((256,256,256))

# 3.2 Функция измерения времени
def measure_time(delegate, is_cuda: bool, repeats: int = 20):
    """Принимает делигат замеряемой функции,
    флаг is_cuda отвечающий за использовние ядер CUDA
    и количество повторных замеров"""

    if is_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            delegate()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats
    delegate()
    start = time.perf_counter_ns()
    for i in range(repeats):
        delegate()
    return (time.perf_counter_ns() - start) / repeats / 10**6


# 3.3 Сравнение операций
def matmul_op(a, b):
    """умножение матриц"""
    return torch.matmul(a, b)

def elem_add(a, b):
    """поэлементное сложение"""
    return a+b

def elem_mul(a, b):
    return torch.mul(a, b)

def transpose_op(m):
    """транспонирование"""
    return torch.transpose(m, -2, -1)

def sum_all_op(m):
    """сумма всех элементов"""
    return m.sum()

def print_stat():
    """Выполняет операции, замеряя время и выводит результаты
    в формате табличек """

    matrices = [big_matrix_1, big_matrix_2, big_matrix_3]
    use_cuda = torch.cuda.is_available()

    for m in matrices:
        shape_str = str(list(m.shape))
        print(f"Матрица размером {shape_str}")
        ops = [
            ("Матричное умножение", matmul_op, 2,
             lambda m=m: matmul_op(m, m)),
            ("Поэлементное сложение", elem_add, 2,
             lambda m=m: elem_add(m, m)),
            ("Поэлементное умножение", elem_mul, 2,
             lambda m=m: elem_mul(m, m)),
            ("Транспонирование", transpose_op, 1,
             lambda m=m: transpose_op(m)),
            ("Сумма всех элементов", sum_all_op, 1,
             lambda m=m: sum_all_op(m)),
        ]
        print(f"{'Операция':<25} | {'CPU (мс)':>10} | {'GPU (мс)':>10} | {'Ускорение':>10}")
        for name, func, n_args, cpu_fn in ops:
            if n_args == 2:
                if m.dim() < 2 or m.shape[-1] != m.shape[-2]:
                    continue

            cpu_time = measure_time(cpu_fn, is_cuda=False)
            gpu_time = None

            if use_cuda:
                mat_cuda = m.cuda()
                if n_args == 2:
                    gpu_fn = lambda m=mat_cuda: func(m, m)
                else:
                    gpu_fn = lambda m=mat_cuda: func(m)
                gpu_time = measure_time(gpu_fn, is_cuda=True)
                del mat_cuda

            if gpu_time is not None:
                speedup = cpu_time / gpu_time
                print(f"{name:<25} | {cpu_time:10.1f} | {gpu_time:10.1f} | {speedup:9.1f}x")
            else:
                print(f"{name:<25} | {cpu_time:10.1f} | {'-':>10} | {'-':>10}")

print_stat()
