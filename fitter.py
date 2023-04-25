#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:34:58 2023

@author: benjaminlear
"""
import numpy as np
import plotly
from pathlib import Path
import PySimpleGUI as sg
from lmfit import Model
import itertools

#define the models we want to use...
def scaled_log_norm(x, A, mu, sd):
    return A * (x * sd * (2 * np.pi)**0.5)**(-1) * np.exp(-1*(np.log(x) - mu)**2 / (2 * sd**2))

#used when formatting strings. 
def find_first_index(num):
    for i, x in enumerate(num):
        if x != "0":
            n = i
            break
    return n

#basically convert to geometric space, and then make sure numbers are long enough
def value_and_error(value, error):
    # convert to strings
    val = str(np.exp(value))
    u_err = str(np.exp(value + error) - np.exp(value))
    l_err = str(np.exp(value) - np.exp(value - error))
    #split value by decimal, and then by first  non-zero
    valD, vald = val.split(".")
    u_errD, u_errd = u_err.split(".")
    l_errD, l_errd = l_err.split(".")
    
    #find the indices where we no longer have zeros
    n_val = find_first_index(vald)
    n_u_err = find_first_index(u_errd)
    n_l_err = find_first_index(l_errd)
        
    #find the longest string we wilol need
    n = max([n_val, n_u_err, n_l_err])
    
    #take the longest version, and join the strings, taking 2 extra indices. 
    return f"{valD}.{vald[:n+2]} (+{u_errD}.{u_errd[:n+2]}) (-{l_errD}.{l_errd[:n+2]})"



#make the model to be used in fiting
ln_model = Model(scaled_log_norm)


# identify the files you wish to fit and the place you want to save them at
filenames = sg.popup_get_file("Choose files", multiple_files = True, file_types = '*.csv').split(";")
#filenames = ["/Users/benjaminlear/My Drive/PennState/Research/Manuscripts/2019/Santina+Vadim/Silver/TEM/AgSC12NP 06152017.csv"]


for data_file in filenames:
    data_file = Path(data_file)
    
    # import data to plot as a numpy array
    lengths = np.loadtxt(
        data_file, 
        delimiter = ",", 
        skiprows = 1, 
        usecols = (1),
        unpack = True # this makes it return arrays for each column
        )
    
    # get stats in ln space, for estimating starting values...
    ln_lengths = np.log(lengths)
    ln_mean = np.mean(ln_lengths)    
    
    ln_q75, ln_q25 = np.percentile(ln_lengths, [75 ,25])
    ln_iqr = ln_q75 = ln_q25
    ln_sd = ln_iqr / 1.35
    
    #bin data
    y_counts, x_edges = np.histogram(lengths, 
                                     bins = "auto"
                                     )
    x_bins = []
    for x in enumerate(x_edges[:-1]):
        x_bins.append((x_edges[x[0]] + x_edges[x[0]+1])/2)
    
    x_bins = np.array(x_bins)
    
    y_density = y_counts / len(lengths)
    h_bin = x_bins[1]-x_bins[0]
    
    result = ln_model.fit(y_density, x = x_bins, A = 1, mu = ln_mean, sd = ln_sd, method = "leastsq")
    print(result.fit_report())
    
    #plot the data
    fig = plotly.graph_objects.Figure( #make a figure, specifying default layouts
        layout = dict(
            template = "simple_white", 
            colorway = plotly.colors.qualitative.Dark2,
            showlegend = False,
            xaxis = dict(
                title = "size /nm", 
                range = [0, np.max(lengths) * 1.1]
                ),
            yaxis = dict(
                title = "density", 
                range = [0, max(y_density)*1.1]
                )
            )
        )
    
    x_sim = np.linspace(0.001, np.max(lengths)* 1.1, 1000)
    
    fig.add_bar(x = x_bins, y = y_density, width = h_bin) # add the data trace to the figure
    fig.add_scatter(x = x_sim, 
                    y = scaled_log_norm(x_sim, result.best_values["A"], result.best_values["mu"], result.best_values["sd"]))


    # annotate with mean and sigma... but only use enough decimal places as needed by the error...
    fig.add_annotation(
            x=np.max(lengths)*0.8,
            y=np.max(y_density),
            xref="x",
            yref="y",
            text=f"&#956; = {value_and_error(result.params['mu'].value, result.params['mu'].stderr)} nm",
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#000000"
                ),
            align="right",
            )
    fig.add_annotation(
            x=np.max(lengths)*0.8,
            y=np.max(y_density)*0.9,
            xref="x",
            yref="y",
            text=f"&#963; = {value_and_error(result.params['sd'].value, result.params['sd'].stderr)} nm",
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#000000"
                ),
            align="right",
            )
    fig.show('svg') # this one will be viewable in spyder
    fig.write_image(data_file.with_suffix('.fit.png')) #write image
    
    # write the final stuff to a file    
    g_mu = np.exp(result.params['mu'].value)
    g_mu_u = np.exp(result.params['mu'].value + result.params['mu'].stderr) - g_mu
    g_mu_l = g_mu - np.exp(result.params['mu'].value - result.params['mu'].stderr)
    
    g_sd = np.exp(result.params['sd'].value)
    g_sd_u = np.exp(result.params['sd'].value + result.params['sd'].stderr) - g_sd
    g_sd_l = g_sd - np.exp(result.params['sd'].value - result.params['sd'].stderr)
    to_write = []
    for i in itertools.zip_longest(
            lengths, 
            x_bins, 
            y_density,
            ["A", "mu", "sigma"],
            [result.params['A'].value, g_mu, g_sd],
            [result.params['A'].stderr, g_mu_u, g_sd_u],
            [result.params['A'].stderr, g_mu_l, g_mu_u],
            [f"{result.params['A'].value} * (x * {result.params['sd'].value} * (2 * 3.14159)**0.5)**(-1) * exp(-1*(log(x) - {result.params['mu'].value})**2 / (2 * {result.params['sd'].value}**2))"]
            ):
        to_write.append(i)
    
    with open(data_file.with_suffix('.fit.csv'), "w") as f:
        f.write("Lengths, bin centers, density, parameters, values, upper error, lower error, equation \n")
        for row in to_write:
            for entry in row:
                if entry != None:
                    f.write(str(entry))
                    f.write(",")
            f.write("\n")
