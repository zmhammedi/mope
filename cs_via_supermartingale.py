import numpy as np
import cvxpy as cp
import time

def rho(x):
    return 1/(2*x) * (np.sqrt((x+1)**2 - 4 * np.exp(1/(2*x)-1/2)*x**(3/2)) + x - 1) 


def bisection_algorithm(f, a, b, y, margin=.00001,direction="right"):
    count = 0
    while count <= 100:
        c = (a + b) / 2
        y_c = f(c)
        if abs(y_c - y) < margin:
            return c
        if direction=="right":
            if y < y_c:
                b = c
            else:
                a = c
        else:
            if y < y_c:
                a = c
            else:
                b = c
        count+=1
    return float(direction=="right")


def linear_search(f, a, b, y, step=.01):
    s=a
    found=False
    while s<= b and f(s) > y:
        s+=step
    if f(min(s,b))<=y:
        found=True
    return min(s,b), found


def supermartingale_value(v,wmax,b0,b1,A0,A1,A2):
    # This function is an efficient version of martingal_value. We avoid calling a solver, and consider cases instead.
    mu = 1
    alpha = 2 # also a free parameter that need be no less than 1. TODO change the name
    
    # Build the relevant matrices
    I = np.identity(2)
    J = np.array([1,1])
    
    gammasols = [np.zeros(3)] # Adding the all zeros feasable solution
    
    # Building S and Q matrices from the sufficient statistics
    S = b0 + b1 * v
    Q = A0 + A1 * v + A2*v**2 
    Sigmainv = pow(mu,-2) * alpha * I + alpha*Q
    Sigma = np.linalg.inv(Sigmainv) # TODO consider Sherman Morison
    Den = np.sqrt(np.linalg.det(I + pow(mu,2)*Q))

    ## Considering cases
    # Case 1: gamma_1 = 0
    gamma = np.zeros(3)
    Cv = np.array([[wmax-1,wmax-1],
                   [-v,wmax-v]])
    if np.linalg.det(Cv)!=0:
        tmp=np.linalg.inv(Cv) @ (-rho(alpha) * Sigmainv @ np.linalg.inv(Cv.T) @ J - S)
        gamma[1]=tmp[0]
        gamma[2]=tmp[1]
        # Check feasability
        if gamma[0]>=0 and gamma[1]>=0 and gamma[2]>=0: 
            gammasols.append(gamma)
    
    # Case 2: gamma_2 = 0
    gamma = np.zeros(3)
    Cv = np.array([[-1,wmax-1],
                   [-v,wmax-v]])
    if np.linalg.det(Cv)!=0:
        tmp=np.linalg.inv(Cv) @ (-rho(alpha) * Sigmainv @ np.linalg.inv(Cv.T) @ J - S)
        gamma[0]=tmp[0]
        gamma[2]=tmp[1]
        if gamma[0]>=0 and gamma[1]>=0 and gamma[2]>=0: 
            gammasols.append(gamma)
    
    # Case 3: gamma_3 = 0
    gamma = np.zeros(3)
    Cv = np.array([[-1,wmax-1],
                   [-v,-v]])
    if np.linalg.det(Cv)!=0:
        tmp=np.linalg.inv(Cv) @ (-rho(alpha) * Sigmainv @ np.linalg.inv(Cv.T) @ J - S)
        gamma[0]=tmp[0]
        gamma[1]=tmp[1]
        if gamma[0]>=0 and gamma[1]>=0 and gamma[2]>=0: 
            gammasols.append(gamma)
    
    # Case 4: (gamma_1,gamma_2) = (0,0)
    gamma = np.zeros(3)
    Cv = np.array([[wmax-1],
                   [wmax-v]])
    
    adding=False
    den = Cv.T @ (Sigma @ Cv)
    if den != 0:
        adding=True
    gamma[2]=(-rho(alpha) - Cv.T @ (Sigma @ S))/den
    if adding and gamma[2]>=0: 
        gammasols.append(gamma)
        
    
    # Case 5: (gamma_1,gamma_3) = (0,0)
    gamma = np.zeros(3)
    Cv = np.array([[wmax-1],
                   [-v]])
    
    adding=False
    den = Cv.T @ (Sigma @ Cv)
    if den != 0:
        adding=True
    gamma[1]=(-rho(alpha) - Cv.T @ (Sigma @ S))/den
    if adding and gamma[1]>=0: 
        gammasols.append(gamma)
        
    # Case 6: (gamma_2,gamma_3) = (0,0)
    gamma = np.zeros(3)
    Cv = np.array([[-1],
                   [-v]])
    
    adding=False
    den = Cv.T @ (Sigma @ Cv)
    if den != 0:
        adding=True
    gamma[0]=(-rho(alpha) - Cv.T @ (Sigma @ S))/den
    if adding and gamma[0]>=0: 
        gammasols.append(gamma)
        
    
    ## Checking the best solution
    J = np.array([1,1,1])
    Cv = np.array([[-1,wmax-1,wmax-1], 
                   [-v,-v,wmax-v]])

    optval = -1
    for gamma in gammasols:
        Stilde = S + Cv @ gamma
        arg = min(20,Stilde.T @ (Sigma @ Stilde)/2 + rho(alpha)*gamma.T @ J)
        val =  np.exp(arg)/Den
        if optval<0 or optval>val:
            optval=val
        
    return optval


def martingale_value_lowerbound(v,wmax,b0,b1,A0,A1,A2,t):  
    eps = 0.001
    # Build the relevant matrices
    I = np.identity(2)
    J = np.array([1,1])
    
    gammasols = [np.zeros(3)] # Adding the all zeros feasable solution
    # Building S and Q matrices from the sufficient statistics
    S = b0 + b1 * v
    Q = A0 + A1 * v + A2*v**2 
    Sigmainv = eps * I + Q
    Sigma = np.linalg.inv(Sigmainv) 
    Den = 1 #(np.exp(1)*t+np.exp(1))

    ## Considering cases
    # Case 1: gamma_1 = 0
    gamma = np.zeros(3)
    Cv = np.array([[wmax-1,wmax-1],
                   [-v,wmax-v]])
    if np.linalg.det(Cv)!=0:
        tmp=np.linalg.inv(Cv) @ (-Sigmainv @ np.linalg.inv(Cv.T) @ J - S)
        gamma[1]=tmp[0]
        gamma[2]=tmp[1]
        # Check feasability
        if gamma[0]>=0 and gamma[1]>=0 and gamma[2]>=0: 
            gammasols.append(gamma)
    
    # Case 2: gamma_2 = 0
    gamma = np.zeros(3)
    Cv = np.array([[-1,wmax-1],
                   [-v,wmax-v]])
    if np.linalg.det(Cv)!=0:
        tmp=np.linalg.inv(Cv) @ (- Sigmainv @ np.linalg.inv(Cv.T) @ J - S)
        gamma[0]=tmp[0]
        gamma[2]=tmp[1]
        if gamma[0]>=0 and gamma[1]>=0 and gamma[2]>=0: 
            gammasols.append(gamma)
    
    # Case 3: gamma_3 = 0
    gamma = np.zeros(3)
    Cv = np.array([[-1,wmax-1],
                   [-v,-v]])
    if np.linalg.det(Cv)!=0:
        tmp=np.linalg.inv(Cv) @ (-Sigmainv @ np.linalg.inv(Cv.T) @ J - S)
        gamma[0]=tmp[0]
        gamma[1]=tmp[1]
        if gamma[0]>=0 and gamma[1]>=0 and gamma[2]>=0: 
            gammasols.append(gamma)
    
    # Case 4: (gamma_1,gamma_2) = (0,0)
    gamma = np.zeros(3)
    Cv = np.array([[wmax-1],
                   [wmax-v]])
    
    adding=False
    den = Cv.T @ (Sigma @ Cv)
    if den != 0:
        adding=True
    gamma[2]=(-1 - Cv.T @ (Sigma @ S))/den
    if adding and gamma[2]>=0: 
        gammasols.append(gamma)
        
    # Case 5: (gamma_1,gamma_3) = (0,0)
    gamma = np.zeros(3)
    Cv = np.array([[wmax-1],
                   [-v]])
    
    den = Cv.T @ (Sigma @ Cv)
    if den != 0:
        adding=True
    gamma[1]=(-1 - Cv.T @ (Sigma @ S))/den
    if adding and gamma[1]>=0: 
        gammasols.append(gamma)
        
    # Case 6: (gamma_2,gamma_3) = (0,0)
    gamma = np.zeros(3)
    Cv = np.array([[-1],
                   [-v]])
    
    adding=False
    den = Cv.T @ (Sigma @ Cv)
    if den != 0:
        adding=True
    gamma[0]=(-1 - Cv.T @ (Sigma @ S))/den
    if adding and gamma[0]>=0: 
        gammasols.append(gamma)
        
    ## Checking the best solution
    J = np.array([1,1,1])
    Cv = np.array([[-1,wmax-1,wmax-1], 
                   [-v,-v,wmax-v]])

    optval = -1
    for gamma in gammasols:
        Stilde = S + Cv @ gamma
        arg = min(20, Stilde.T @ (Sigma @ Stilde)/(4 * (4*np.log(2)-2)) + (1/2) * gamma.T @ J)
        val =  np.exp(arg)/Den
        if optval<0 or optval>val:
            optval=val
        
    return optval


# def martingale_value_lowerbound_slow(v,wmax,b0,b1,A0,A1,A2,t):
#     # the argument of the exponential
#     eps = 0.001
    
#     #Build the relevant matrices
#     I = np.identity(2)
#     J = np.array([1,1,1])
#     #TODO check if we need to use wmin below
#     Cv = np.array([[-1,wmax-1,wmax-1],[-v,-v,wmax-v]])
    
#     S = b0 + b1 * v
#     Q = A0 + A1 * v + A2 * v**2 
#     gamma = cp.Variable(3)
#     Stilde = S + Cv @ gamma
#     Sigma = np.linalg.inv(eps * I + Q) #TODO Make this more efficient via Sherman Morison
#     cost = cp.quad_form(Stilde, Sigma)/(4*(4*np.log(2)-2)) + gamma.T @ J/2
#     prob = cp.Problem(cp.Minimize(cost),[0<=gamma])
#     prob.solve()
    
#     return np.exp(prob.value) #Denominator equal to 1

# def supermartingale_value_slow(v,wmax,b0,b1,A0,A1,A2):
#     # the argument of the exponential
#     mu = 1
#     alpha = 2 # also a free parameter that need be no less than 1 
    
#     # Build the relevant matrices
#     I = np.identity(2)
#     J = np.array([1,1,1])
#     # TODO check if we need to use wmin below
#     Cv = np.array([[-1,wmax-1,wmax-1],[-v,-v,wmax-v]])
    
#     S = b0 + b1 * v
#     Q = A0 + A1 * v + A2 * v**2 
#     gamma = cp.Variable(3)
#     Stilde = S + Cv @ gamma
#     Sigma = np.linalg.inv(pow(mu,-2) * alpha * I + alpha*Q) # TODO Make this more efficient via Sherman Morison
#     cost = cp.quad_form(Stilde, Sigma)/2 + rho(alpha)*gamma.T @ J  
#     prob = cp.Problem(cp.Minimize(cost),[0<=gamma])
#     prob.solve()
    
#     return np.exp(prob.value)/np.sqrt(np.linalg.det(I + pow(mu,2)*Q)) #TODO use the matrix determinant lemma for efficiency


def cs_via_supermartingale(data, wmin, wmax, alpha):
    # Assume data is of type np.array((t,2)), where (w,r)=(wr[:,0],wr[:,1]).
    # TODO we want to allow for different policies eventually
    T = len(data)
    
    # Initialize
    b0 = np.zeros(2)
    b1 = np.zeros(2)
    A0 = np.zeros((2,2))
    A1 = np.zeros((2,2))
    A2 = np.zeros((2,2))

    lb = np.zeros(T)
    ub = np.zeros(T)
    for t in range(T):
        wt = data[t,0] 
        rt = data[t,1]
        b0 += [wt-1,wt*rt]
        b1 += [0,-1]
        A0 += [[(wt-1)**2, (wt-1)*wt*rt],
               [(wt-1) * wt * rt,(wt * rt)**2]]
        A1 += [[0, -(wt-1)],
               [-(wt-1),-2 * wt * rt]]
        A2 += [[0,0],[0,1]]
        
        martval = lambda v: supermartingale_value(v,wmax,b0,b1,A0,A1,A2)        
        # Root finding
        tmpv, found = linear_search(martval,0,1,1/(2 * alpha),step=0.001)
        #tmpv = bisection_algorithm(martval,0,1,1/(2 * alpha),margin=0.000001)
        if not found:
            lb[t]=0.
            ub[t]=1.
            continue
        
        lb[t] = bisection_algorithm(martval,0,tmpv,1/alpha,margin=0.000001,direction="left")
        ub[t] = bisection_algorithm(martval,tmpv,1,1/alpha,margin=0.000001)
            
    return lb, ub


def cs_via_supermartingale_debug(data, wmin, wmax, alpha):
    # Assume data is of type np.array((t,2)), where (w,r)=(wr[:,0],wr[:,1]).
    # TODO we want to allow for different policies eventually
    T = len(data)
    
    # Initialize
    b0 = np.zeros(2)
    b1 = np.zeros(2)
    A0 = np.zeros((2,2))
    A1 = np.zeros((2,2))
    A2 = np.zeros((2,2))

    w = data[:,0] 
    r = data[:,1]
    b0 = np.array([np.sum(w)-T,np.dot(w,r)])
    b1 = np.array([0,-T])
    A0 = np.array([[np.dot(w-1, w-1), np.dot(w-1, w*r)],
          [np.dot(w-1, w*r), np.dot(w*r, w*r)]])
    A1 = np.array([[0, -np.sum(w)+T],
                   [-np.sum(w)+T,-2 * np.dot(w,r)]])
    A2 = np.array([[0,0],[0,T]])
    #print(f'b0={b0},\nb1={b1},\nA0={A0},\nA1={A1},\nA2={A2}\n\n')
    
    martval = lambda v: supermartingale_value(v,wmax,b0,b1,A0,A1,A2)        
    # Root finding
    tmpv, found = linear_search(martval,0,1,1/(2 * alpha),step=0.001)
    #print(tmpv)
    #tmpv = bisection_algorithm(martval,0,1,1/(2 * alpha),margin=0.000001)
    if not found:
        #print(f'b0={b0},\nb1={b1},\nA0={A0},\nA1={A1},\nA2={A2}\n\n')
        #print('tmpv=0')
        return np.array([0.] * T), np.array([1.] * T)

    lb = bisection_algorithm(martval,0,tmpv,1/alpha,margin=0.000001,direction="left")
    ub = bisection_algorithm(martval,tmpv,1,1/alpha,margin=0.000001)
    #print(f'tmpv not eq 1: lb={lb}, up={ub}\n')    
    return np.array([float(lb)] * T), np.array([float(ub)] * T)

def cs_via_EWA(data, wmin, wmax, alpha):
    # Assume data is of type np.array((t,2)), where (w,r)=(wr[:,0],wr[:,1]).
    # TODO we want to allow for different policies eventually
    T = len(data)
    
    # Initialize
    b0 = np.zeros(2)
    b1 = np.zeros(2)
    A0 = np.zeros((2,2))
    A1 = np.zeros((2,2))
    A2 = np.zeros((2,2))

    lb = np.zeros(T)
    ub = np.zeros(T)
    for t in range(T):
        wt = data[t,0] 
        rt = data[t,1]
        b0 += [wt-1,wt*rt]
        b1 += [0,-1]
        A0 += [[(wt-1)**2, (wt-1)*wt*rt],
               [(wt-1) * wt * rt,(wt * rt)**2]]
        A1 += [[0, -(wt-1)],
               [-(wt-1),-2 * wt * rt]]
        A2 += [[0,0],[0,1]]
        
        martval = lambda v: martingale_value_lowerbound(v,wmax,b0,b1,A0,A1,A2,t)
       
        # Root finding
        tmpv, found = linear_search(martval,0,1,1/(2 * alpha),step=0.001)
        #tmpv = bisection_algorithm(martval,0,1,1/(2 * alpha),margin=0.000001)
        if not found:
            #print(f'b0={b0},\nb1={b1},\nA0={A0},\nA1={A1},\nA2={A2}\n\n')
            #print(f'tmpv={tmpv}')
            lb[t]=0.
            ub[t]=1.
            continue
            
        lb[t] = bisection_algorithm(martval,0,tmpv,1/alpha,margin=0.000001,direction="left")
        ub[t] = bisection_algorithm(martval,tmpv,1,1/alpha,margin=0.000001)
            
    return lb, ub


def cs_via_EWA_debug(data, wmin, wmax, alpha):
    # Assume data is of type np.array((t,2)), where (w,r)=(wr[:,0],wr[:,1]).
    # TODO we want to allow for different policies eventually
    T = len(data)
    
    # Initialize
    b0 = np.zeros(2)
    b1 = np.zeros(2)
    A0 = np.zeros((2,2))
    A1 = np.zeros((2,2))
    A2 = np.zeros((2,2))

    #t0 = time.time()
    w = data[:,0] 
    r = data[:,1]
    b0 = np.array([np.sum(w)-T,np.dot(w,r)])
    b1 = np.array([0,-T])
    A0 = np.array([[np.dot(w-1, w-1), np.dot(w-1, w*r)],
          [np.dot(w-1, w*r), np.dot(w*r, w*r)]])
    A1 = np.array([[0, -np.sum(w)+T],
                   [-np.sum(w)+T,-2 * np.dot(w,r)]])
    A2 = np.array([[0,0],[0,T]])
    #print(f'Time to construct matrices is {time.time()-t0} seconds')
    #print(f'b0={b0},\nb1={b1},\nA0={A0},\nA1={A1},\nA2={A2}\n\n')
    
    martval = lambda v: martingale_value_lowerbound(v,wmax,b0,b1,A0,A1,A2,T)
        
    # Root finding
    t0 = time.time()
    tmpv, found = linear_search(martval,0,1,1/(2 * alpha),step=0.001)
    #tmpv = bisection_algorithm(martval,0,1,1/(2 * alpha),margin=0.000001)
    if not found:
        #print(f'b0={b0},\nb1={b1},\nA0={A0},\nA1={A1},\nA2={A2}\n\n')
        #print(f'tmpv={tmpv}')
        return np.array([0.] * T), np.array([1.] * T)
        
    lb = bisection_algorithm(martval,0,tmpv,1/alpha,margin=0.000001,direction="left")
    ub = bisection_algorithm(martval,tmpv,1,1/alpha,margin=0.000001)
    #print(f'tmpv not eq 1: lb={lb}, up={ub}\n')
            
    return np.array([float(lb)] * T), np.array([float(ub)] * T)


