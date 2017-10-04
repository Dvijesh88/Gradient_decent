from numpy import *

def compute_error_for_given_points(b,m,points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - ((m * x) + b)) **2
    return totalError / float(len(points))

def step_gradient(b_current,m_current,points,learningRate):
    #gredient descenr
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points,starting_b, starting_m,learning_rate, num_iteration):
    b = starting_b
    m = starting_m
    for i in range(num_iteration):
        b, m = step_gradient(b, m, array(points),learning_rate)
        print(" new b = {0} and new m = {1} and Iteration = {2} Error = {3} ".format(b,m,i,compute_error_for_given_points(b,m,points)))
    return [b,m]

def run():
    points = genfromtxt('data1.csv', delimiter=',',)
    #hyper parameters
    learning_rate=0.0001
    #y =mx+b (Slop formulas)
    initial_b = 0
    initial_m = 0
    num_iteration = 1000000
    print("Starting Gradient at b = {0} ,  m = {1}, error = {2}".format(initial_b, initial_m,
                                                                        compute_error_for_given_points(initial_b,
                                                                                                       initial_m,
                                                                                                       points)))
    [b,m] = gradient_descent_runner(points,initial_b,initial_m,learning_rate,num_iteration)
    print("Starting Gradient at b = {0} ,  m = {1}, error = {2}".format(b, m,
                                                                        compute_error_for_given_points(b,
                                                                                                       m,
                                                                                                       points)))
   # print(b)
   # print(m)


run()
