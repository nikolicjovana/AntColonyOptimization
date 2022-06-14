import pandas as pd
import matplotlib.pyplot as plt

df_rho = pd.read_csv('./results/rho.csv')
df_alpha = pd.read_csv('./results/alpha.csv')
df_beta = pd.read_csv('./results/beta.csv')

iters = df_rho['Unnamed: 0']
df_rho.drop('Unnamed: 0', axis = 1, inplace = True)
df_alpha.drop('Unnamed: 0', axis = 1, inplace = True)
df_beta.drop('Unnamed: 0', axis = 1, inplace = True)

df_rho.index = iters.tolist()
df_alpha.index = iters.tolist()
df_beta.index = iters.tolist()

#pl = df_rho.plot(title = 'Просечна дужина најкраће путање за различите вредности параметра ρ')
pl = df_rho.plot()
pl.set_title('Просечна дужина најкраће путање за различите вредности параметра ρ', pad = 20)
pl.set_xlabel('Број итерација')
pl.set_ylabel('Просечна дужина најкраће путање')
plt.savefig('./plots/rho.png')
pl = df_alpha.plot()
pl.set_title('Просечна дужина најкраће путање за различите вредности параметра α', pad = 20)
pl.set_xlabel('Број итерација')
pl.set_ylabel('Просечна дужина најкраће путање')
plt.savefig('./plots/alpha.png')
pl = df_beta.plot()
pl.set_title('Просечна дужина најкраће путање за различите вредности параметра β', pad = 20)
pl.set_xlabel('Број итерација')
pl.set_ylabel('Просечна дужина најкраће путање')
plt.savefig('./plots/beta.png')