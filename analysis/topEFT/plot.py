from __future__ import print_function, division
from collections import defaultdict, OrderedDict
import gzip
import pickle
import json
import os
import uproot
import matplotlib.pyplot as plt
import numpy as np
from coffea import hist, processor 
from coffea.hist import plot
import os, sys
import optparse

from topcoffea.plotter.plotter import plotter

#path = ['histos/TTTo2L2Nu.pkl.gz', 'histos/DY.pkl.gz', 'histos/dataDM.pkl.gz', 'histos/dataEG.pkl.gz', 'histos/dataME.pkl.gz', 'histos/dataSM.pkl.gz']
path = 'histos/plotsTopEFT.pkl.gz'

processDic = {
  'Diboson': 'WZTo2L2Q, WZTo3LNu, ZZTo2L2Nu, ZZTo2L2Q, ZZTo4L, WWTo2L2Nu',
  'Triboson': 'WWW, WZG, WWZ, WZZ, ZZZ',
  'ttH': 'ttHnobb, ttHH',
  'ttll': 'TTZToLL_M_1to10, TTZToLLNuNu_M_10_a, TTG, ttZH, ttZZ',
  'ttlv': 'TTWJetsToLNu, ttWW, ttWZ',
  'tllq': 'tZq',
  'tHq': 'tHq',
  #'Fakes': 'TTTo2L2Nu, DYJetsToLL_M_10to50_MLM, DYJetsToLL_M_50_aMCatNLO',
  'TT': 'TTTo2L2Nu',
  'DY': 'DYJetsToLL_M_10to50_MLM, DYJetsToLL_M_50_aMCatNLO',
  #'Flips': '',
  'data' : 'EGamma, MuonEG,  SingleMuon, DoubleMuon',
}

bkglist = ['TT', 'DY', 'Diboson', 'Triboson', 'ttlv', 'ttll', 'ttH', 'tllq', 'tHq']
allbkg  = ['Diboson', 'Triboson', 'ttH', 'ttll', 'ttlv', 'tllq', 'tHq', 'Fakes']

processList = ''
for keys in processDic:
  if keys == 'data': continue
  if processDic[keys] == '': continue
  processList = processList + processDic[keys] + ','
processArray = processList.split(',')[:-1]
flipArray = [x + '_flip' for x in processArray]
flipList = flipArray[0]
for i in range(1, len(flipArray)):
  flipList = flipList + ', ' +flipArray[i]

#processDic['Flips'] = flipList
#bkglist = np.append(bkglist, 'Flips')
#allbkg = np.append(allbkg, 'Flips')

colordic ={
  'Other' : '#808080',
  'Diboson' : '#ff00ff',
  'Triboson': '#66ff66',
  'ttH' : '#CC0000',
  'ttll': '#00a278',
  'ttlv': '#009900',
  'tllq': '#ff66ff',
  'tHq' : '#00ffff',
  'Flips': '#66B2ff',
  'TT': '#ffff33',
  'DY': '#33ff33',
}

preset = {
  'ch3l'      : ['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ','eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ'],
  'ch3lSSonZ' : ['eemSSonZ', 'mmeSSonZ', 'eeeSSonZ', 'mmmSSonZ'],
  'ch2lss'    : ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS'],
  'ch2los'    : ['eeOSonZ', 'eeOSoffZ', 'mmOSonZ', 'mmOSoffZ', 'emOS'],
  'ch2lsfos'  : ['eeOSonZ', 'eeOSoffZ', 'mmOSonZ', 'mmOSoffZ'],
  'ch2lsfosonZ':['eeOSonZ', 'mmOSonZ'],
  'eeSS'      : ['eeSSonZ', 'eeSSoffZ'],
  'mmSS'      : ['mmSSonZ', 'mmSSoffZ'],
}
preset['ch3lp']      = [x+'_p' for x in preset['ch3l']]
preset['ch3lm']      = [x+'_m' for x in preset['ch3l']]
preset['ch3lSSonZp'] = [x+'_p' for x in preset['ch3lSSonZ']]
preset['ch3lSSonZm'] = [x+'_m' for x in preset['ch3lSSonZ']]
preset['ch2lssp']    = [x+'_p' for x in preset['ch2lss']]
preset['ch2lssm']    = [x+'_m' for x in preset['ch2lss']]

usage = 'usage: %prog [options]'
parser = optparse.OptionParser(usage)
parser.add_option('-v', '--variable',  dest='variable',  help='variable',   default='counts',    type='string')
parser.add_option('-c', '--channel',   dest='channel',   help='channels',   default='ch3l',      type='string')
parser.add_option('-l', '--level',     dest='level',     help='cut',        default='base',      type='string')
parser.add_option('-t', '--title',     dest='title',     help='title',      default='3 leptons', type='string')
parser.add_option('-o', '--output',    dest='output',    help='output',     default=None,        type='string')
(opt, args) = parser.parse_args()

for keys in preset:
    if opt.channel == keys:
        opt.channel = preset[keys]
else                           : channel = opt.channel
level = opt.level

categories = {
 'channel' : channel,#['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ','eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ'],#'eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS'],
 'cut' : level#['base', '2jets', '4jets', '4j1b', '4j2b'],
 #'Zcat' : ['onZ', 'offZ'],
 #'lepCat' : ['3l'],
}

colors = [colordic[k] for k in bkglist]

def Draw(var, categories, label=''):
  plt = plotter(path, prDic=processDic, bkgList=bkglist)
  plt.plotData = True
  plt.SetColors(colors)
  plt.SetCategories(categories)
  plt.SetRegion(label)
  plt.SetOutput(opt.output)
  plt.Stack(var, xtit='', ytit='')
  plt.PrintYields('counts')

Draw(opt.variable, categories, opt.title)
