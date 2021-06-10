#y = wx+b
def compute_error_for_line_given_points(b,w,points):
    totalError =0
    for i in range(0,len(points)):
        x = points[i][0]
        y = points[i][1]
        #computer mean-squared-error
        totalError += (y-(w*x+b))**2
    #average loss for each point
    return totalError/float(len(points))

#Compute Gradinet and update
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0,len(points)):
        x = points[i][0]
        y = points[i][1]
        #grad_b = 2(wx+b-y)
        b_gradient += (2/N)*((w_current*x+b_current)-y)
        #grad_w = 2(wx+b-y)x
        w_gradient += (2/N)*((w_current*x+b_current)-y)*x
    new_b = b_current-(learningRate*b_gradient)
    new_w = w_current-(learningRate*w_gradient)
    return [new_b,new_w]
a
