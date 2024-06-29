import numpy as np
def write_file_define(params,filename):
  fp = open(filename,'w')
  output = "Energy_Unit = au\n\n"
  fp.write(output)
  output = "nstate = "+str(params['ndvr'])+"\n"
  fp.write(output)
  for i in range(params['ndvr']):
    output = "V_S"+str(i)+" = "+str(params['e0'][i,i])+"\n"
    fp.write(output)
  for i in range(params['ndvr']):
    for j in range(i):
      if(np.abs(params['e0'][i,j])>1e-12):
        output = "V_S"+str(i)+"_S"+str(j)+" = "+str(params['e0'][i,j])+"\n"
        fp.write(output)
  fp.write("\n")
  output = "nmode = "+str(params['nlevel'])
  for i in range(params['nlevel']):
    if(i%10==0):
      output = output+"\n"
      fp.write(output)
      output = "modes "
    output = output + " | v" + str(i)
  output = output+"\n\n"
  fp.write(output)
  for i in range(params['nlevel']):
    output = "omega_v"+str(i)+" = "+str(params['omega'][i])+"\n"
    fp.write(output)
  fp.write("\n")
  for i in range(params['ndvr']):
    for k in range(params['nlevel']):
      if(np.abs(params['coef1'][i,i,k])>1e-12):
        output = "kl_S"+str(i)+"_v"+str(k)+" = "+str(params['coef1'][i,i,k])+"\n"
        fp.write(output)
  for i in range(params['ndvr']):
    for j in range(i):
      for k in range(params['nlevel']):
        if(np.abs(params['coef1'][i,j,k])>1e-12):
          output = "kl_S"+str(i)+"_S"+str(j)+"_v"+str(k)+" = "+str(params['coef1'][i,j,k])+"\n"
          fp.write(output)
  fp.write("\n")
  for i in range(params['ndvr']):
    for k in range(params['nlevel']):
      if(np.abs(params['coef2'][i,i,k,k])>1e-12):
        output = "kq_S"+str(i)+"_v"+str(k)+" = "+str(params['coef2'][i,i,k,k])+"\n"
        fp.write(output)
  for i in range(params['ndvr']):
    for j in range(i):
      for k in range(params['nlevel']):
        if(np.abs(params['coef2'][i,j,k,k])>1e-12):
          output = "kq_S"+str(i)+"_S"+str(j)+"_v"+str(k)+" = "+str(params['coef2'][i,j,k,k])+"\n"
          fp.write(output)
  fp.write("\n")
  for i in range(params['ndvr']):
    for k in range(params['nlevel']):
      for l in range(k):
        if(np.abs(params['coef2'][i,i,k,l])>1e-12):
          output = "kb_S"+str(i)+"_v"+str(k)+"_v"+str(l)+" = "+str(params['coef2'][i,i,k,l])+"\n"
          fp.write(output)
  for i in range(params['ndvr']):
    for j in range(i):
      for k in range(params['nlevel']):
        for l in range(k):
          if(np.abs(params['coef2'][i,j,k,l])>1e-12):
            output = "kb_S"+str(i)+"_S"+str(j)+"_v"+str(k)+"_v"+str(l)+" = "+str(params['coef2'][i,j,k,l])+"\n"
            fp.write(output)
  fp.write("\n")
  for key in ['small','nrtt','nrmax','init_state','custom_nb']:
    if (key in params.keys()):
      output = key+" = "+str(params[key])+"\n"
      fp.write(output)
  if ('custom_nb' in params.keys()):
    if (params['custom_nb']==False):
      output = 'nbv'+" = "+str(params['nbv'])+"\n"
      fp.write(output)
    else:
      fp.write("\n")
      for i in range(params['nlevel']):
        output = "v"+str(i)+" = "+str(params['nb1'][i])+"\n"
        fp.write(output)
    fp.flush()
  return
