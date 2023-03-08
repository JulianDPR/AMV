def acp(x, n_comp = 2, plot = True):
    
    if n_comp<2 or n_comp>3:
        
        print("Error in n_comp, must be >=2 and <4")
        
    else:
        
        import numpy as np
        from decimal import Decimal
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import plotly.express as px
        import plotly
        import plotly.graph_objects as go
        
        #Obtenemos los espacios
        variables = [i for i in x.columns]
        p = len(variables)
        individuos = [i for i in x.index]
        n = len(individuos)
        
        
        #Ahora centramos y escalamos los datos usando numpy
        x = x.to_numpy()
        x_mean = x.mean(axis=0, dtype="float64")
        x_sd = x.var(axis=0, dtype="float64")
        x_n = n
        
        x_ = x.transpose()
        x__mean = x_.mean(axis=0, dtype="float64")
        x__sd = x_.var(axis=0, dtype="float64")
        x_p = p
        
        xc = (x-x_mean)/(x_n**(1/2))
        x_c = (x_ - x__mean)/(x_p**(1/2))
        
        covx = np.matmul(xc.transpose(), xc, dtype="float64")
        covx_ = np.matmul(x_c.transpose(), x_c, dtype="float64")
        
        #Con la matriz de covarianzas, usamos la diagonal para extraer las
        #desviaciones estandar correspondientes
        
        #x_var = np.array([covx[i][i] for i in range(len(covx))])
        #x__var = np.array([covx_[i][i] for i in range(len(covx_))])
        
        x_sd = np.diag(1/x_sd)**(1/2)
        x__sd = np.diag(1/x__sd)**(1/2)
        
        x =np.matmul(xc,x_sd)


        corx = np.matmul(x.transpose(), x)
        corx_ = np.matmul(np.matmul(x__sd, covx_), x__sd)
        
        covx = pd.DataFrame(covx, columns = variables,
                            index = variables)
        covx_ = pd.DataFrame(covx_, columns = individuos,
                             index = individuos)
        corx = pd.DataFrame(corx, columns = variables,
                            index = variables)
        corx_ = pd.DataFrame(corx_, columns = individuos,
                             index= individuos)
        
        #print(pd.DataFrame(corx, columns=variables, index=variables),
         #     pd.DataFrame(corx_, columns=individuos, index=individuos))
        
        #Con esto finalizamos la centralizacion de los datos y procedemos a
        #hallar las inercias y cuanto aportan estas a la total.
        
        #Con esto se hallan los nuevos ejes directores, a los cuales se les 
        #observara la contribucion a cada eje
        
        u = np.linalg.eig(corx)[1]
        #v = np.linalg.eig(corx_)[1]
        
        #Entonces se hallan las inercias, se puede hallar tambien como
        #Diag(U_XU), pero, realmente contribuye a sus valores propios
        
        lam = np.linalg.eig(corx)[0]
        
        #Se encuentra v a partir de u
        
        v = [np.matmul(x,
                       u[:,[i]].reshape(len(lam),1))/(lam[i]**(1/2)) for i in range(len(lam))]
        
        v = pd.DataFrame(np.matrix(np.array(v))).transpose()
        
        phi = np.matmul(x.transpose(),v)
        
        phi.index=variables
        
        #Ya con esto se sabe que la inercia total es la Traza(Diag(U_XU))
        
        cont = lam/lam.sum()
        
        
        #Hallamos las proyecciones ortogonales de los nuevos
        #Ejes directores
        
        psi = np.matmul(x, u)
        psi = pd.DataFrame(psi)
        
        #phi = np.matmul(corx_, v)
        
        Con, ax = plt.subplots()
        
        for i in range(len(lam)):
            
            ax.plot(np.linspace(0,cont[i],100),
                    np.linspace(lam[i],lam[i],100),
                    "*", label=f"{i+1}° factorial axis",
                    color = "black")
            
        ax.set_title(r"$\lambda$'s contribution")
        ax.set_xlabel("Frec")
        ax.set_ylabel(r"$\lambda$")
        ax.legend()
        
        #print(np.matmul(phi, phi.transpose()))
        
        if n_comp == 2 :
            
            var_, va = plt.subplots()
            
            ind_ , ia = plt.subplots()
            
            ind_var, indvar = plt.subplots()
            
            va.scatter(psi.iloc[:,0], psi.iloc[:,1],alpha=0.7, color='r')
            va.spines["right"].set_position(('data',0))
            va.spines['top'].set_position(('data',0))
            
            for i in range(n):
    
                va.text(psi.iloc[i,0], psi.iloc[i,1], s = individuos[i])
            
            va.set_title("Individuos en el eje de las variables")
            va.set_xlabel(f" axis contribution {np.round(cont[0],3)}")
            va.set_ylabel(f" axis contribution {np.round(cont[1], 3)}")
            
            
            ####################################
            
            
            f = np.linspace(0,2*np.pi,100)
            cos = np.cos(f)
            sen = np.sin(f)
            
            ia.plot(cos, sen, "--", color='black')
            ia.scatter(phi.iloc[:,0], phi.iloc[:,1], alpha=0.7, color='g',
                       marker="^")
            ia.spines["right"].set_position(('data',0))
            ia.spines['top'].set_position(('data',0))
            
            for i in range(p):
                
                
                ia.plot([0, phi.iloc[i,0]], [0,phi.iloc[i,1]],
                        color='g')
                ia.text(phi.iloc[i,0], phi.iloc[i,1], s = variables[i])
            
            ia.set_title("Variables en el eje de los individuos")
            ia.set_xlabel(f" axis contribution {np.round(cont[0],3)}")
            ia.set_ylabel(f" axis contribution {np.round(cont[1], 3)}")
            
            
            indvar.scatter(psi.iloc[:,0],
                           psi.iloc[:,1], alpha=0.7,
                           color="r")
            #indvar.plot(cos,sen,"--", color="black")
            indvar.scatter(u[0], u[1], alpha=0.7, color="g",
                           marker="^")
            indvar.spines["right"].set_position(('data', 0))
            indvar.spines["top"].set_position(("data", 0))
            
            for i in range(p):
                
                indvar.plot([0,u[0][i]],[0,u[1][i]], color="g", alpha=0.5)
                indvar.text(u[0][i], u[1][i], s = variables[i])
            
            indvar.set_title("Individuos y Variables")
            indvar.set_xlabel(f"axis contribution {np.round(cont[0], 3)}")
            indvar.set_ylabel(f"axis contribution {np.round(cont[1], 3)}")
            
        else:
            
            #########3d
            
            var_ = px.scatter_3d(x=psi.iloc[:,0],
                                 y=psi.iloc[:,1],
                                 z=psi.iloc[:,2],opacity=0.7,
                                 title="Individuos en ejes de las variables",
        labels={"x":f"axis contribution {np.round(cont[0],3)}",
                "y":f"axis contribution {np.round(cont[1],3)}",
                "z":f"axis contribution {np.round(cont[2],3)}"},
        text=psi.index)
            
           # plotly.offline.plot(var_)
            
            ind_=px.scatter_3d(x=phi.iloc[:,0],
                                 y=phi.iloc[:,1],
                                 z=phi.iloc[:,2], text=phi.index,
                                 title="Variables en el eje de los individuos",
        labels={"x":f"axis contribution {np.round(cont[0],3)}",
                "y":f"axis contribution {np.round(cont[1],3)}",
                "z":f"axis contribution {np.round(cont[2],3)}"},
        opacity=0.7)
            
            

            for i in range(p):
                
                x = px.line_3d(x=[0,phi.iloc[i,0]],
                                     y=[0,phi.iloc[i,1]],
                                     z=[0,phi.iloc[i,2]])

                ind_ = go.Figure(data=ind_.data+x.data)
                
            
            #plotly.offline.plot(ind_)
            
            ind_var = px.scatter_3d(x=psi.iloc[:,0],
                                 y=psi.iloc[:,1],
                                 z=psi.iloc[:,2], text=psi.index)
            
            ind_2=px.scatter_3d(x=phi.iloc[:,0],
                                 y=phi.iloc[:,1],
                                 z=phi.iloc[:,2], text=phi.index,
                                 title="Variables en el eje de los individuos",
                                 labels={"x":f"axis contribution {np.round(cont[0],3)}",
                                         "y":f"axis contribution {np.round(cont[1],3)}",
                                         "z":f"axis contribution {np.round(cont[2],3)}"},
                                 opacity=0.7)
            
            ind_var = go.Figure(data=ind_var.data+ind_2.data)

            for i in range(p):
                
                x = px.line_3d(x=[0,phi.iloc[i,0]],
                                     y=[0,phi.iloc[i,1]],
                                     z=[0,phi.iloc[i,2]])

                ind_var = go.Figure(data=ind_var.data+x.data)
                
            ind_var.update_layout(title={
                "text":"Variables e individuos"})
            
            #plotly.offline.plot(ind_var)
            
            

    return (variables, individuos,
            p, n, corx,pd.DataFrame(u), pd.DataFrame(v),
            lam, psi, phi, Con, var_, ind_, ind_var)