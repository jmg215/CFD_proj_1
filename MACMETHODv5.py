# -*- coding: utf-8 -*-
"""
Created on March  2 20:39:33 2022

@author: J.Michael
"""

import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba import jit




def MAC():
    #start time
    tic = time.time()
    
    #set Length of in X direction
    Lx = 1
    
    #set Length in Y direction
    Ly=1
    """
    !!!!!!!!!!!!!!!!
    make sure to set NX and NY as down below. must be equal. try 21x21 or 129x129, or whatever square grid your heart desires
    """
    #set Number of points in X direction
    NX=21
    
    #set Number of points in Y direction
    NY=21
    
    #set X mesh grid spacing
    dx=Lx/(NX-1)
    
    #set Y mesh grid spacing
    dy=Ly/(NY-1)
    
    # #set time jump
    # dt=.04
    
    """
    !!!!!!!!!!!!!!!!!!!!!!!!
    Reynolds number parameter
    """
    Re=100
    
    #time step requirements (restrictions)
    req1 = 1/(Re*.25)
    req2 = .25*(Re*dx**2)
    
    #set the time step based on the minimum of the time step requirements
    dt = min(req1,req2)
    
    #set the tolerance for the Poisson solver (interpreted as max error for G-S method)
    tol = .000001
    
    #initialilze the p_error
    p_error = 1.0
    
    #set the max allowed error between iterations for u and v
    max_err = .00001
    
    #initialize the u & v errors
    u_error = 1.0
    v_error = 1.0
    
    #number of iterations for Poisson solver
    p_itr = 0
    
    #total iterations of the flow solver
    total_itr = 0
    
    through_p = 0
    
    sigma = (math.cos(math.pi/NX)*2)/2
    wbot = math.sqrt(1-sigma**2)
    
    #optimal overrelaxation parameter for SOR scheme
    w = 2/(1+wbot)
    
    #print(w)
    
    
    
    
    
    """
    TIME STEP RESTRICTIONS:
        (.25(|u|+|v|)**2)*dt*Re <= 1
        dt/(Re*dx**2) <= .25      -> this is assuming dx=dy
    
    """
    
    #initialilze the grid for u and v velocities, p pressure, plus F and G
    u = np.zeros((NX,NY+1),dtype=np.float)
    v = np.zeros((NX+1,NY),dtype=np.float)
    p = np.zeros((NX+1,NY+1),dtype=np.float)
    F = np.zeros((NX,NY+1),dtype=np.float)
    G = np.zeros((NX+1,NY),dtype=np.float)
    p_copy = np.copy(p)
    u_copy = np.copy(u)
    v_copy = np.copy(v) 
    
    """
    INITIAL CONDITIONS
    """
    #SET THE BC ACROSS THE WALL (WHICH IS IMAGINARY NOW. only the ghost remain)
    
    # u[0,:]= 2 - u[1,:] #lid u velocity
    # v[0,:]=-v[1,:] #lid v velocity
    
    # u[1:-1,0]=-u[1:-1,1] # left wall u velocity
    # v[:,0]=-v[:,1] # left wall v velocity
    
    # u[1:-1,-1]=-u[1:-1,-2] # right wall u velocity
    # v[:,-1]=-v[:,-2] # right wall v velocity
    
    # u[-1,:]=-u[-2,:] # bottom wall u velocity
    # v[-1,:]=-v[-2,:] # bottom wall v velocity
    
    
    u[:,0]= 2 - u[:,1] #lid u velocity
    # v[0,:]=-v[1,:] #lid v velocity
    v[:,0]=0 #lid v velocity
    
    # u[1:-1,0]=-u[1:-1,1] # left wall u velocity
    u[-1,:]=0 # left wall u velocity 
    # u[:,0]=0
    v[-1,:]=-v[-2,:] # left wall v velocity
    
    #u[1:-1,-1]=-u[1:-1,-2] # right wall u velocity
    u[0,:]=0 #right wall u velocity #set to 77 
    # u[:,-1]=0
    v[0,:]=-v[1,:] # right wall v velocity
    
    u[:,-1]=-u[:,-2] # bottom wall u velocity
    # u[:,-1]=-99 # bottom wall u velocity
    # v[-1,:]=-v[-2,:] # bottom wall v velocity
    v[:,-1]=0 # bottom wall v velocity
    
    
    
    F[:,0]= 2 - F[:,1] #lid u velocity
    # v[0,:]=-v[1,:] #lid v velocity
    G[:,0]=0 #lid v velocity
    
    # u[1:-1,0]=-u[1:-1,1] # left wall u velocity
    F[-1,:]=0 # left wall u velocity 
    # u[:,0]=0
    G[-1,:]=-G[-2,:] # left wall v velocity
    
    #u[1:-1,-1]=-u[1:-1,-2] # right wall u velocity
    F[0,:]=0 #right wall u velocity #set to 77 
    # u[:,-1]=0
    G[0,:]=-G[1,:] # right wall v velocity
    
    F[:,-1]=-F[:,-2] # bottom wall u velocity
    # u[:,-1]=-99 # bottom wall u velocity
    # v[-1,:]=-v[-2,:] # bottom wall v velocity
    G[:,-1]=0 # bottom wall v velocity
    
    
    
    
    
    
    # F[0,:]=2-F[1,:] #lid u velocity
    # # v[0,:]=-v[1,:] #lid v velocity
    # G[0,:]=0 #lid v velocity
    
    # # u[1:-1,0]=-u[1:-1,1] # left wall u velocity
    # F[1:-1,0]=0 # left wall u velocity - change to all rows
    # # u[:,0]=0
    # G[:,0]=-G[:,1] # left wall v velocity
    
    # #u[1:-1,-1]=-u[1:-1,-2] # right wall u velocity
    # F[1:-1,-1]=0 #non-ghost cell treatment -> change to all rows
    # # u[:,-1]=0
    # G[:,-1]=-G[:,-2] # right wall v velocity
    
    # F[-1,:]=-F[-2,:] # bottom wall u velocity
    # # v[-1,:]=-v[-2,:] # bottom wall v velocity
    # G[-1,:]=0 # bottom wall v velocity
    
    
    
    
    
    
    """
    
    **********************************************
    MAIN LOOP. solver runs through this
    """
    while v_error > max_err or u_error > max_err:
        
            #A = 0
            #B=0
            #C=0
            #D=0
            
            
            
            
            
            """
            CREATE THE F MATRIX
            """
            for i in range(1,NX-1):
                for j in range(1,NY):
                    #F[i,j]:
                    A = (u[i+1,j]-2*u[i,j]+u[i-1,j])/(Re*dx*dx)
                    B = (u[i,j+1]-2*u[i,j]+u[i,j-1])/(Re*dy*dy)
                    C = (((u[i,j]+u[i+1,j])/2)**2 - ((u[i-1,j]+u[i,j])/2)**2)/dx
                    
                    
                    
                    """
                    TYPE-0
                    """
                 
                    #CHANGES MADE REGARDING DIVIDING BY 2.
                    # d1 = (u[i,j]+u[i,j+1])*(v[i,j]+v[i+1,j])/2
                    # d2 = (u[i,j-1]+u[i,j])*(v[i,j-1]+v[i+1,j-1])/2
                    
                    d1 = ((u[i,j]+u[i,j+1])/2)*((v[i,j]+v[i+1,j])/2)
                    d2 = ((u[i,j-1]+u[i,j])/2)*((v[i,j-1]+v[i+1,j-1])/2)
                    
                    
                    """
                    TYPE-I
                    """
                    #d1 = ((u[i,j]+u[i,j+1])/2)*((v[i,j+1]+v[i+1,j+1])/2) #changes to indices for u and v. about j+.5. FROM V5
                    #d2 = ((u[i,j]+u[i,j-1])/2)*((v[i,j]+v[i+1,j])/2) #similar changes. should be about j-.5 FROM V5
                    
                    """
                    TYPE-II
                    """
                    #d1 = ((u[i,j]+u[i,j+1])/2)*((v[i-1,j+1]+v[i+1,j+1])/2) #changes to indices for u and v. about j+.5. original written form. see papers
                    #d2 = ((u[i,j]+u[i,j-1])/2)*((v[i-1,j-1]+v[i+1,j-1])/2) #similar changes. should be about j-.5: made changes back to original written form. see papers
                    
                    """
                    TYPE-III
                    """
                    D = (d1-d2)/dy
                    F[i,j] = u[i,j] + dt*(A+B-C-D) 

            # #weird
            # F[0,:]=-F[1,:]
            # F[-1,:]=-F[-2,:]                      
                    
            """
            CREATE THE G matrix
            """
            for e in range(1,NX):
                for f in range(1,NY-1):
                    
      
                    
                    #G[i,j]:
                    W = (v[e+1,f]-2*v[e,f]+v[e-1,f])/(Re*dx*dx)
                    X = (v[e,f+1]-2*v[e,f]+v[e,f-1])/(Re*dy*dy)
                    Y = (((v[e,f]+v[e,f+1])/2)**2 - ((v[e,f-1]+v[e,f])/2)**2)/dy
                    
                    """
                    TYPE-0
                    """
                    #CHANGES MADE REGARDING DIVIDING BY 2.
            
                    # z1 = (u[e,f]+u[e,f+1])*(v[e,f]+v[e+1,f])/2
                    # z2 = (u[e-1,f]+u[e-1,f+1])*(v[e-1,f]+v[e,f])/2
                    z1 = ((u[e,f]+u[e,f+1])/2)*((v[e,f]+v[e+1,f])/2)
                    z2 = ((u[e-1,f]+u[e-1,f+1])/2)*((v[e-1,f]+v[e,f])/2)
                    
                    """
                    TYPE - I
                    """
                    #z1 = ((u[i+1,j]+u[i+1,j+1])/2)*((v[i,j]+v[i+1,j])/2) #from V5
                    #z2 = ((u[i,j+1]+u[i,j])/2)*((v[i,j]+v[i-1,j])/2) #from V5
                    
                    
                    """
                    TYPE-II
                    """
                    #z1 = ((u[i+1,j-1]+u[i+1,j+1])/2)*((v[i,j]+v[i+1,j])/2) #dame
                    #z2 = ((u[i-1,j+1]+u[i-1,j-1])/2)*((v[i,j]+v[i-1,j])/2) #time
                    
                    Z = (z1-z2)/dx
                    G[e,f] = v[e,f] + dt*(W+X-Y-Z)
                    
            # #why not
            # G[:,0]=-G[:,1]
            # G[:,-1]=-G[:,-2]
                    
                    
            """
            set the boundary conditions of the F and G arrays to be like their
            u and v counterparts. ignore first block
            """
            # F[0,:]=2-F[1,:] #lid u velocity
            # # v[0,:]=-v[1,:] #lid v velocity
            # G[0,:]=0 #lid v velocity
            
            # # u[1:-1,0]=-u[1:-1,1] # left wall u velocity
            # F[1:-1,0]=0 # left wall u velocity - change to all rows
            # # u[:,0]=0
            # G[:,0]=-G[:,1] # left wall v velocity
            
            # #u[1:-1,-1]=-u[1:-1,-2] # right wall u velocity
            # F[1:-1,-1]=0 #non-ghost cell treatment -> change to all rows
            # # u[:,-1]=0
            # G[:,-1]=-G[:,-2] # right wall v velocity
            
            # F[-1,:]=-F[-2,:] # bottom wall u velocity
            # # v[-1,:]=-v[-2,:] # bottom wall v velocity
            # G[-1,:]=0 # bottom wall v velocity
            
            
            F[:,0]= 2 - F[:,1] #lid u velocity
            # v[0,:]=-v[1,:] #lid v velocity
            G[:,0]=0 #lid v velocity
            
            # u[1:-1,0]=-u[1:-1,1] # left wall u velocity
            F[-1,:]=0 # left wall u velocity 
            # u[:,0]=0
            G[-1,:]=-G[-2,:] # left wall v velocity
            
            #u[1:-1,-1]=-u[1:-1,-2] # right wall u velocity
            F[0,:]=0 #right wall u velocity #set to 77 
            # u[:,-1]=0
            G[0,:]=-G[1,:] # right wall v velocity
            
            F[:,-1]=-F[:,-2] # bottom wall u velocity
            # u[:,-1]=-99 # bottom wall u velocity
            # v[-1,:]=-v[-2,:] # bottom wall v velocity
            G[:,-1]=0 # bottom wall v velocity
            
            
            """
            while loop that solves the pressure poisson equation using gauss-seidel
            check BC
            
            """
            """set/reset pressure bounary conditions"""
            p[0,:] = p[1,:]#dp/dy = 0 at the top ##FIX THIS TO BE NEUMANN
            p[:,0] = p[:,1] #dp/dx = 0 at the left wall
            p[:,-1] = p[:,-2] #dp/dx = 0 at the right wall
            p[-1,:] = p[-2,:] #dp/dy = 0 at the bottom
             
            
            p,p_copy = Pressure_Poisson(p,p_copy,NX,NY,tol,w,F,G,dx,dy,dt)
            #Pressure_Poisson(press,press_copy,nx,ny,toll,omega,eff,gee,d_ex,d_why,d_tee)
            
            
            # p_error = 1.0
            # while p_error > tol:        
            #     for m in range(1,NX):
            #         for n in range(1,NY):
            #             #changed the negative to a pos
            #             #huge change. multiply the FG term by dx**2. divide FG term by 2(1+Beta)
            #             #CLOSER TO GOAL THAN BEFORE
                        
            #             """
            #             V-5 PRESSURE
            #             """
            #             #p[m,n] = (1-1.5)*p_copy[m,n] + 1.5*.25*(p_copy[m+1,n]+p[m-1,n]+p_copy[m,n+1]+p[m,n-1]) - (((F[m,n]/dx)+(G[m,n]/dy))/dt)*(dx**2/(4))  
                        
            #             """
            #             V-6 PRESSURE
            #             """
            #             p[m,n] = (1-w)*p_copy[m,n] + w*.25*(p_copy[m+1,n]+p[m-1,n]+p_copy[m,n+1]+p[m,n-1]) -  w*((((F[m,n]-F[m-1,n])/dx) + ((G[m,n]-G[m,n-1])/dy))/dt)*((dx**2)/(4))  
            #             #CHECK ABOVE EQN FOR ACCURACY
            #             #print(F[m,n]/dx)
                    
                
    
                
            #     p_error = max(abs(p.flatten(order='C')-p_copy.flatten(order='C')))
                
            #     """check for divergence by checking if any elements in a or b evaluate to NaN. if true, break"""
            #     if ((np.any(np.isnan(p)) or np.any(np.isnan(p_copy))) == True):
            #         print('iterations diverge')
            #         break
            #     if p_itr % 40 == 0:
            #         print("p_error: ", p_error)
                
            #     """
            #     LOCATION OF P COPY: bottom after reset
            #     """
            #     #p_copy = np.copy(p)
                
            #     p_itr +=1
            #     print("p_itr: ", p_itr)
                
            #     # """set/reset pressure bounary conditions"""
            #     # p[:,0]=p[:,1] #dp/dx = 0 at the left wall
            #     # p[:,-1]=p[:,-2] #dp/dx = 0 at the right wall
            #     # p[-1,:]=p[-2,:] #dp/dy = 0 at the bottom
            #     # p[0,:] = p[1,:] #p = 0 at the top ##FIX THIS TO BE NEUMANN
            #     p_copy = np.copy(p)
            #     # if p_itr == 1800:
            #     #     continue

            #print('p_RRRR$$$$: ', p_error)
            # p[:,0] = p[:,1] #dp/dx = 0 at the left wall
            # p[:,-1] = p[:,-2] #dp/dx = 0 at the right wall
            # p[-1,:] = p[-2,:] #dp/dy = 0 at the bottom
            # p[0,:] = p[1,:] #dp/y = 0 at the top ##FIX THIS TO BE NEUMANN
            through_p +=1             
                
                
            """
            U (n+1 iteration level)
            """
            
            for r in range(1,NX-1):
                for s in range(1,NY):    
                    u[r,s] = F[r,s] - (dt/dx)*(p[r+1,s] - p[r,s])
                    
                    
            """
            V (iteration level n+1)
            """        
            for y in range(1,NX):
                for z in range(1,NY-1):
                    v[y,z] = G[y,z] - (dt/dy)*(p[y,z+1] - p[y,z])
                    
                    
                    
                
            u_error = max(abs(u.flatten(order='C')-u_copy.flatten(order='C')))
            v_error = max(abs(v.flatten(order='C')-v_copy.flatten(order='C')))
            
            print('u_error: ', u_error)
            print('v_error: ', v_error)
            
            """check for divergence by checking if any elements in a or b evaluate to NaN. if true, break"""
            if ((np.any(np.isnan(u)) or np.any(np.isnan(u_copy))) == True):
                print('iterations diverge: U')
                break   
            if ((np.any(np.isnan(v)) or np.any(np.isnan(v_copy))) == True):
                print('iterations diverge: V')
                break         
            
            
            """
            LOCATION OF U AND V COPY: bottom
            """
            #u_copy = np.copy(u)
            #v_vopy = np.copy(v)
            
            # u[0,:]=1 #lid u velocity
            # v[0,:]=0 #lid v velocity
            
            # u[1:-1,0]=0 # left wall u velocity
            # v[:,0]=0 # left wall v velocity
            
            # u[1:-1,-1]=0 # right wall u velocity
            # v[:,-1]=0 # right wall v velocity
            
            # u[-1,:]=0 # bottom wall u velocity
            # v[-1,:]=0 # bottom wall v velocity
            
            """
            issues with ghost cell treatment vs actual boundary treatment
            """
            # u[0,:]= 2 - u[1,:] #lid u velocity
            # # v[0,:]=-v[1,:] #lid v velocity
            # v[0,:]=0 #lid v velocity
            
            # # u[1:-1,0]=-u[1:-1,1] # left wall u velocity
            # u[1:-1,0]=0 # left wall u velocity - change to all rows
            # # u[:,0]=0
            # v[:,0]=-v[:,1] # left wall v velocity
            
            # #u[1:-1,-1]=-u[1:-1,-2] # right wall u velocity
            # u[1:-1,-1]=0 #non-ghost cell treatment -> change to all rows
            # # u[:,-1]=0
            # v[:,-1]=-v[:,-2] # right wall v velocity
            
            # u[-1,:]=-u[-2,:] # bottom wall u velocity
            # # v[-1,:]=-v[-2,:] # bottom wall v velocity
            # v[-1,:]=0 # bottom wall v velocity
            
            
            u[:,0]= 2 - u[:,1] #lid u velocity
            # v[0,:]=-v[1,:] #lid v velocity
            v[:,0]=0 #lid v velocity
            
            # u[1:-1,0]=-u[1:-1,1] # left wall u velocity
            u[-1,:]=0 # left wall u velocity 
            # u[:,0]=0
            v[-1,:]=-v[-2,:] # left wall v velocity
            
            #u[1:-1,-1]=-u[1:-1,-2] # right wall u velocity
            u[0,:]=0 #right wall u velocity #set to 77 
            # u[:,-1]=0
            v[0,:]=-v[1,:] # right wall v velocity
            
            u[:,-1]=-u[:,-2] # bottom wall u velocity
            # u[:,-1]=-99 # bottom wall u velocity
            # v[-1,:]=-v[-2,:] # bottom wall v velocity
            v[:,-1]=0 # bottom wall v velocity
            
            u_copy = np.copy(u)
            v_copy = np.copy(v)
            
            
            
            total_itr += 1
            
            
            #print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            print('#####################################################')
            #print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            #print('#####################################################')
            print('total_itr: ', total_itr)
            if total_itr == 1:
                print('what anime is this again?')
            if total_itr == 20:
                print('20X through u and v. continuing,,,,')
            if total_itr == 25:
                print('i remember when i was 25...')
            if total_itr == 30:
                print('As Bo burnham said, now Im turning 30')
            if total_itr == 40:
                print('THIS IS 40?!!')
            if total_itr == 80:
                continue
            if total_itr == 140:
                continue
            if total_itr == 200:
                continue
            #     return u,v,p
            if total_itr == 300:
                continue
            
            if u_error < max_err:
                 continue
            if v_error < max_err:
                continue
        
        
                
            #p_error = 1
                
        
        
    
    tok = time.time()
    clock = tok-tic
    return u, v, p, clock


@jit(nopython=True, cache=True)
def Pressure_Poisson(press,press_copy,nx,ny,toll,omega,eff,gee,d_ex,d_why,d_tee):
    p_error = 1.0
    while p_error > toll:        
        for m in range(1,nx):
            for n in range(1,ny):
                #changed the negative to a pos
                #huge change. multiply the FG term by dx**2. divide FG term by 2(1+Beta)
                #CLOSER TO GOAL THAN BEFORE
                
                """
                V-5 PRESSURE
                """
                #p[m,n] = (1-1.5)*p_copy[m,n] + 1.5*.25*(p_copy[m+1,n]+p[m-1,n]+p_copy[m,n+1]+p[m,n-1]) - (((F[m,n]/dx)+(G[m,n]/dy))/dt)*(dx**2/(4))  
                
                """
                V-6 PRESSURE
                using SOR. omega is set at top of code
                """
                press[m,n] = (1-omega)*press_copy[m,n] + omega*.25*(press_copy[m+1,n]+press[m-1,n]+press_copy[m,n+1]+press[m,n-1]) -  omega*((((eff[m,n]-eff[m-1,n])/d_ex) + ((gee[m,n]-gee[m,n-1])/d_why))/d_tee)*((d_ex**2)/(4))  
                #CHECK ABOVE EQN FOR ACCURACY
                #print(F[m,n]/dx)
            
        #press_norm = np.

        
        #L2 vs L1 norm
        
        #p_error = max(abs(press.flatten(order='C')-press_copy.flatten(order='C')))
        p_error = np.linalg.norm(press-press_copy)
        
        """check for divergence by checking if any elements in a or b evaluate to NaN. if true, break"""
        if ((np.any(np.isnan(press)) or np.any(np.isnan(press_copy))) == True):
            print('iterations diverge')
            break
        # if p_itr % 40 == 0:
        #     print("p_error: ", p_error)
        
        """
        LOCATION OF P COPY: bottom after reset
        """
        #p_copy = np.copy(p)
        
        # p_itr +=1
        #print("p_itr: ", p_itr)
        
        # """set/reset pressure bounary conditions"""
        # p[:,0]=p[:,1] #dp/dx = 0 at the left wall
        # p[:,-1]=p[:,-2] #dp/dx = 0 at the right wall
        # p[-1,:]=p[-2,:] #dp/dy = 0 at the bottom
        # p[0,:] = p[1,:] #p = 0 at the top ##FIX THIS TO BE NEUMANN
        press_copy = np.copy(press)
    return press, press_copy



def make_new(you,vee,pea):
    U_new = np.transpose(you)
    #U_new = np.fliplr(UU) #was UU
    
    V_new = np.transpose(vee)
    #V_new = np.fliplr(VV)
    
    P_new = np.transpose(pea)
    #P_new = np.fliplr(PP)
    
    return U_new, V_new, P_new
    

    
#U,V,P,DELTA = MAC()


# UU = np.transpose(U)
# UUU = np.fliplr(UU)

# VV = np.transpose(V)
# VVV = np.fliplr(VV)

# PP = np.transpose(P)
# PPP = np.fliplr(PP)


#UUU, VVV, PPP = make_new(U,V,P)

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
must adjust NX and NY below to correspond to test parameters in MAC function
"""


#set Number of points in X direction
NX=21

#set Number of points in Y direction
NY=21

# u_two = np.zeros((NX,NY),dtype=np.float)
# v_two = np.zeros((NX,NY),dtype=np.float)
# p_two = np.zeros((NX,NY),dtype=np.float)

# print('******************************\n')
# print('elapsed time: ', DELTA)
# print(VV)
# print('************')
# print(VVV)

# for AA in range(0,20):
#     for BB in range(0,21):
#         u_new[AA,BB] = (UUU[AA,BB]+UUU[AA+1,BB])/2 #avg by columns

        #p_new[AA,BB] = (P[AA+1,BB])
        
def main():
    U,V,P,DELTA = MAC()
    
    UUU,VVV,PPP = make_new(U,V,P)
    
    return U, V, P, DELTA, UUU, VVV, PPP


old_u,old_v,old_p,sometime,new_u,new_v,new_p = main()

u_output = np.zeros((NX,NY),dtype=np.float)
v_output = np.zeros((NX,NY),dtype=np.float)
p_output = np.zeros((NX,NY),dtype=np.float)
u_midplane = []

for AA in range(0,NY):
    for BB in range(0,NX-1):
        u_output[AA,BB] = (new_u[AA,BB]+new_u[AA+1,BB])/2
        
for CC in range(0,NY-1):
    for DD in range(0,NX):
        v_output[CC,DD] = (new_v[CC,DD]+new_v[CC,DD+1])/2
        
for EE in range(0,NY):
    for FF in range(0,NX):
        p_output[EE,FF] = .25*(new_p[EE+1,FF]+new_p[EE-1,FF]+new_p[EE,FF+1]+new_p[EE,FF-1])



"""
for u
"""
new_uu = np.flip(u_output,0)
midpt = int((NY-1)/2) # NY must be an odd number
for HH in range(0,NY):
    u_midplane.append(new_uu[HH,midpt])

# #create the 2D contour map. also flip the matrix about the x-axis
# # f = np.flip(a,0)
# new_uu = np.flip(u_output,0)
# #plot it
# plt.contourf(x,y,new_uu,cmap=cm.RdPu)
# plt.colorbar()
# plt.title('actualRdPu')
# #plt.savefig('nicegraphSOR.png', dpi = 1200)
# plt.show()
# #figure,axis = plt.subplots(3,1)

print('time to converge: ', sometime)

"""
PLOT FUNCTIONALITY
"""
x,y = np.linspace(0,1,NX),np.linspace(0,1,NY)
xx,yy = np.meshgrid(x,y)
fig, [(ax1,ax2),(ax3,ax4),(ax5,ax6)] = plt.subplots(figsize=(9,12),nrows=3,ncols=2)
#fig.set_title('U,V and P for Re=100')
#fig, axes = plt.subplots(2,2)

"""vertical subplot domain"""

#PLOT U
# new_uu = np.flip(u_output,0)
ax1.set_aspect('equal')
ax1.set_title('PLOT OF U')
plotU = ax1.contourf(xx,yy,new_uu,cmap=cm.BrBG)
fig.colorbar(plotU, ax=ax1)


#create the 2D contour map. also flip the matrix about the x-axis
# f = np.flip(a,0)

#PLOT V
new_vv = np.flip(-1*v_output,0)
ax2.set_aspect('equal')
ax2.set_title('PLOT OF V')
plotV = ax2.contourf(xx,yy,new_vv,cmap=cm.gnuplot2) #RdPu
fig.colorbar(plotV,ax=ax2)


#PLOT P
new_pp = np.flip(p_output,0)
ax4.set_aspect('equal')
ax4.set_title('PLOT OF P')
PlotP = ax4.contourf(xx,yy,new_pp) # x vs xx, y vs yy for contour
fig.colorbar(PlotP,ax=ax4,cmap=cm.cividis)


#PLOT U MIDPLANE
ax3.set_aspect('equal')
ax3.set_title('U mid plane')
Plot_mid_U = ax3.plot(u_midplane,y)
ax3.set_xlim([-1,1])

#PLOT STREAMLINES
# ax5.set_aspect('equal')
# ax5.set_title('streamlines')
# PlotSL = ax5.quiver(xx[::2,::2],yy[::2,::2],new_uu[::2,::2],new_vv[::2,::2])

#PLOT STREAMPLOT
ax5.set_aspect('equal')
ax5.set_title('streamlines')
PlotSL = ax5.streamplot(xx,yy,new_uu,new_vv,density=.4)


#PLOT VEL MEGNITUDE
uv = np.zeros((NX,NY),dtype=np.float)
for QQ in range(0,NY):
    for RR in range(0,NX):
        uv[QQ,RR] = math.sqrt(new_uu[QQ,RR]**2 + new_vv[QQ,RR]**2)


ax6.set_aspect('equal')
ax6.set_title('PLOT OF MAGNITUDE')
mycmap = plt.get_cmap('rainbow_r')
#newcmap = plt.get_cmap()

PLOTMAG = ax6.contourf(xx,yy,uv)
fig.colorbar(PLOTMAG,ax=ax6, cmap=cm.plasma)

"""rectabngular subplot domain"""

# new_uu = np.flip(u_output,0)
# axes[0,0].contourf(xx,yy,new_uu)
# axes[0,0].colorbar(cmap=cm.RdPu)

# new_vv = np.flip(v_output,0)
# axes[1,0].contourf(xx,yy,new_vv)
# axes[1,0].colorbar(cmap=cm.plasma)

# new_pp = np.flip(p_output,0)
# axes[0,1].contourf(xx,yy,new_pp)
# axes[0,1].colorbar(cmap=cm.cividis)

# #new_pp = np.flip(p_output,0)
# axes[1,1].contourf(xx,yy,new_pp)
# axes[1,1].colorbar(cmap=cm.cividis)






#plt.savefig('nicegraphSOR.png', dpi = 1200)
#fig.savefig('129x129Re3200wSL.png', dpi=1200)
plt.show()
#figure,axis = plt.subplots(3,1)


        
