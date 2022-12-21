import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from copy import copy
from scipy import stats

def factors(n):
    
    '''
    Prime factor decomposition of n.
    '''
    
    result = []
    
    for i in range(2,n+1):
        
        while n/float(i) == int(n/float(i)):
            n = n/float(i)
            result.append(i)
        
        if n == 1:
            return result

        
def get_n_rowcol(fields):
    
    '''
    Function to define the number of cols and rows for a visually balanced multi-plot.
    '''
    
    n_fields = len(fields)
    
    n_factors = factors(n_fields)
    
    nrow = 1
    ncol = 1

    for i in range (0,len(n_factors)):
        val = n_factors[-(i+1)]

        #Ensure nrow >= ncol
        if ncol*val > nrow :
            nrow = nrow*val
        else:
            ncol = ncol*val

    return nrow, ncol


def plot_cm(data, method='pearson'):
    
    '''
    Correlation matrix visualization.
    '''
    
    corr_data = data.corr(method = method)
    mask = np.triu(np.ones_like(corr_data), 1)
    
    fig, ax = plt.subplots(figsize = (10,10))

    sns.heatmap(
        data = corr_data,
        center = 0,
        annot = True,
        fmt = '.2f',
        cbar = False,
        cmap = 'coolwarm',
        mask = mask
    )

    ax.set_title(f"Matrice de corrélation ({method})")
    
    return fig


def plot_corr_pvalues(data, method='pearson', thresh=0.05):
    
    '''
    Compute and display correlation p-value under null hypothesis for variables in a dataset.
    '''
    
    data = data.dropna()
    columns = pd.DataFrame(columns=data.columns, dtype='float64')
    data_pvalues = columns.transpose().join(columns)
    
    for index in columns:
        for row in columns:
            
            if method == 'pearson':
                data_pvalues.loc[index, row] = stats.pearsonr(data[index], data[row])[1]
            
            elif method == 'kendall':
                data_pvalues.loc[index, row] = stats.kendalltau(data[index], data[row])[1]
                
            elif method == 'spearman':
                data_pvalues.loc[index, row] = stats.spearmanr(data[index], data[row])[1]
            
            else:
                raise ValueError("'method' doit être l'une des suivantes :'pearson', 'kendall', 'spearman'")
   
    mask = np.triu(np.ones_like(data_pvalues), 1)    
    
    fig, ax = plt.subplots(figsize = (10,10))

    my_cmap = copy(plt.cm.Reds)
    my_cmap.set_under("white")
    
    sns.heatmap(
        data = data_pvalues,
        vmin = thresh,
        annot = True,
        fmt = '.2f',
        cbar = False,
        cmap = my_cmap,
        mask = mask
    )

    ax.set_title(f"p-values de corrélation ({method})")
    
    return fig


def tukey_outliers(data, values_col, IQR_mult=1.5, group_field=None):
    
    '''
    Fonction identifiant les outliers d'une distribution, en suivant l'approche de Tukey.
    
    Paramètres:
    -----------
    - data : base de données
    - values_col : nom (str) de colonne de data contenant les données à analyser
    - IQR_mult : nombre d'écarts interquartile (float) à utiliser pour la définition d'un outlier
    - group_field : nom (str) de la colonne de data contenant les catégories à utiliser en cas d'approche par catégorie
    
    Résultat:
    ---------
    list de bool indiquant si un élément est identifié comme outlier (True) ou non (False)
    '''
    
    data = data.copy()
    
    #Boucle en cas d'approche par groupe
    if group_field is not None:
        
        data['Outlier'] = False
        
        for group in data[group_field].unique():
            mapping = data[group_field]==group
            
            group_data = data.loc[mapping, values_col]
            Q1 = group_data.quantile(0.25)
            Q3 = group_data.quantile(0.75)
            IQR = Q3-Q1
            
            data.loc[mapping, 'Outlier'] = group_data.between(Q1-IQR_mult*IQR, Q3+IQR_mult*IQR)
            
        return ~data['Outlier']
    
    else:
        
        data = data[values_col]
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3-Q1
        
        return ~data.between(Q1-IQR_mult*IQR, Q3+IQR_mult*IQR)


def anova_test(data, group_field, indicator):
        
    '''
    Fonction calculant les éléments d'analyse ANOVA pour un ensemble de données numériques contenues dans un dataframe.
    
    Paramètres:
    -----------
    - data : dataframe contenant les données numériques à évaluer
    - group_field : nom (str) de la colonne de data contenant les catégories à utiliser pour l'ANOVA
    - indicator : nom (str) de la colonne contenant l'indicateur numérique pour lequel l'ANOVA est réalisé
    
    Résultat:
    ---------
    - valeur du eta^2 = SCE/SCT calculé
    - p_value de l'hypothèse nulle estimée sur la base d'une analyse ANOVA unidimensionnelle (certaines conditions statistiques requises)
    - p_value de l'hypothèse nulle estimée sur la base d'une analyse Alexander-Govern unidimensionnelle (conditions similaires hors homoscédasticité)
    '''
    
    groups_values = []

    ind_mean = data[indicator].mean(skipna = True)
    groups = data[group_field].unique().tolist()
    
    non_constant_groups = groups.copy()

    for group in groups:

        mapping = (data[group_field]==group)&(data[indicator].notna())
        mapping_ind = data[indicator].notna()
        data_group = data.loc[mapping, indicator]

        groups_values.append({
            'n_i': len(data_group),
            'group_mean': data_group.mean()
        })      

        if len(data_group.unique()) == 1:
            non_constant_groups.remove(group)

    SCT = sum([(value_ind - ind_mean)**2 for value_ind in data.loc[mapping_ind,indicator]])
    SCE = sum([group['n_i']*(group['group_mean'] - ind_mean)**2 for group in groups_values])

    # Calcul de la p-value de H0 avec méthode ANOVA one-way
    p_val_anova = stats.f_oneway(*[data.loc[data[group_field]==group, indicator].dropna() for group in groups]).pvalue
    # Calcul de la p-value de H0 avec méthode Alexander-Govern (ne nécessitant pas l'hypothèse d'homoscédasticité)
    p_val_alexgovern = stats.alexandergovern(*[data.loc[data[group_field]==group, indicator] for group in non_constant_groups], nan_policy = 'omit').pvalue

    return SCE/SCT, p_val_anova, p_val_alexgovern