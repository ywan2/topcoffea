#!/usr/bin/env python
import lz4.frame as lz4f
import cloudpickle
import json
import pprint
import coffea
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
#from coffea.arrays import Initialize # Not used and gives error
from coffea import hist, processor
from coffea.util import load, save
from optparse import OptionParser
from coffea.analysis_tools import PackedSelection

from topcoffea.modules.objects import *
from topcoffea.modules.corrections import SFevaluator, GetLeptonSF
from topcoffea.modules.selection import *
from topcoffea.modules.HistEFT import HistEFT, EFTHelper

#coffea.deprecations_as_errors = True

# In the future these names will be read from the nanoAOD files
#WCNames = ['ctW', 'ctp', 'cpQM', 'ctli', 'cQei', 'ctZ', 'cQlMi', 'cQl3i', 'ctG', 'ctlTi', 'cbW', 'cpQ3', 'ctei', 'cpt', 'ctlSi', 'cptb']

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, samples, wc_names_lst=[], do_errors=False):
        self._samples = samples

        # Create the histograms
        # In general, histograms depend on 'sample', 'channel' (final state) and 'cut' (level of selection)
        self._accumulator = processor.dict_accumulator({
        'SumOfEFTweights'  : HistEFT("SumOfWeights", wc_names_lst, hist.Cat("sample", "sample"), hist.Bin("SumOfEFTweights", "sow", 1, 0, 2)),
        'dummy'   : hist.Hist("Dummy" , hist.Cat("sample", "sample"), hist.Bin("dummy", "Number of events", 1, 0, 1)),
        'counts'  : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("counts", "Counts", 1, 0, 2)),
<<<<<<< HEAD
        'invmass' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut","cut"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 60, 0, 150)),
        'njets'   : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("njets",  "Jet multiplicitu ", 10, 0, 10)),
        'nbtags'  : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("nbtags", "btag multiplicitu ", 5, 0, 5)),
        'met'     : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("met",    "MET (GeV)", 40, 0, 400)),
        'm3l'     : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m3l",    "$m_{3\ell}$ (GeV) ", 20, 0, 200)),
        'wleppt'  : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("wleppt", "$p_{T}^{lepW}$ (GeV) ", 20, 0, 200)),
        'e0pt'    : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("e0pt",   "Leading elec $p_{T}$ (GeV)", 40, 0, 200)),
        'm0pt'    : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m0pt",   "Leading muon $p_{T}$ (GeV)", 40, 0, 200)),
        'j0pt'    : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("j0pt",   "Leading jet  $p_{T}$ (GeV)", 43, 0, 200)),
        'e0eta'   : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("e0eta",  "Leading elec $\eta$", 30, -3, 3)),
        'm0eta'   : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m0eta",  "Leading muon $\eta$", 30, -3, 3)),
        'j0eta'   : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("j0eta",  "Leading jet  $\eta$", 41, -3, 3)),
        'ht'      : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("ht",     "H$_{T}$ (GeV)", 40, 0, 800)),
        'l0pt'    : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("l0pt",   "Leading lepton $p_{T}$ (GeV)", 22, 0, 200)),
        'l0eta'   : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("l0eta",  "Leading lepton $\eta$", 43, -3, 3)),
        'jpt'     : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("jpt",   "All jets  $p_{T}$ (GeV)", 43, 0, 200)),
        'jeta'    : HistEFT("Events", WCNames, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("jeta",  "All jets  $\eta$", 41, -3, 3)),
=======
        'invmass' : hist.Hist("Events", hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut","cut"), hist.Bin("invmass", "$m_{\ell\ell}$ (GeV) ", 20, 0, 200)),
        'njets'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("njets",  "Jet multiplicity ", 10, 0, 10)),
        'nbtags'  : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("nbtags", "btag multiplicity ", 5, 0, 5)),
        'met'     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("met",    "MET (GeV)", 40, 0, 400)),
        'm3l'     : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m3l",    "$m_{3\ell}$ (GeV) ", 20, 0, 200)),
        'wleppt'  : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("wleppt", "$p_{T}^{lepW}$ (GeV) ", 20, 0, 200)),
        'e0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("e0pt",   "Leading elec $p_{T}$ (GeV)", 30, 0, 300)),
        'm0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m0pt",   "Leading muon $p_{T}$ (GeV)", 30, 0, 300)),
        'j0pt'    : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("j0pt",   "Leading jet  $p_{T}$ (GeV)", 20, 0, 400)),
        'e0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("e0eta",  "Leading elec $\eta$", 20, -2.5, 2.5)),
        'm0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("m0eta",  "Leading muon $\eta$", 20, -2.5, 2.5)),
        'j0eta'   : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("j0eta",  "Leading jet  $\eta$", 20, -2.5, 2.5)),
        'ht'      : HistEFT("Events", wc_names_lst, hist.Cat("sample", "sample"), hist.Cat("channel", "channel"), hist.Cat("cut", "cut"), hist.Bin("ht",     "H$_{T}$ (GeV)", 40, 0, 800)),
>>>>>>> 2df0772d6b533cee040227fbab3172607875bc17
        })

        self._eft_helper = EFTHelper(wc_names_lst)
        self._do_errors = do_errors # Whether to calculate and store the w**2 coefficients
        
    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):
        # Dataset parameters
        dataset = events.metadata['dataset']
        year   = self._samples[dataset]['year']
        xsec   = self._samples[dataset]['xsec']
        sow    = self._samples[dataset]['nSumOfWeights' ]
        isData = self._samples[dataset]['isData']
        datasets = ['SingleMuon', 'SingleElectron', 'EGamma', 'MuonEG', 'DoubleMuon', 'DoubleElectron']
        for d in datasets: 
          if d in dataset: dataset = dataset.split('_')[0]

        # Initialize objects
        met = events.MET
        e   = events.Electron
        mu  = events.Muon
        #tau = events.Tau
        j   = events.Jet

        try:         
            e['genPart'] = events.GenPart[e.genPartIdx]
            mu['genPart']= events.GenPart[mu.genPartIdx]
            e['flipmask'] = (e.genPart.pdgId + e.pdgId == 0)
            mu['flipmask']=(mu.genPart.pdgId +mu.pdgId == 0)
            #print(len(ak.flatten(e.flipmask[e.flipmask])))
        except:
            e['flipmask'] = (e.pdgId == 0)
            mu['flipmask'] = (mu.pdgId == 0)

        # Muon selection

        mu['isPres'] = isPresMuon(mu.dxy, mu.dz, mu.sip3d, mu.looseId)
        mu['isTight']= isTightMuon(mu.pt, mu.eta, mu.dxy, mu.dz, mu.pfRelIso03_all, mu.sip3d, mu.mvaTTH, mu.mediumPromptId, mu.tightCharge, mu.looseId, minpt=10)
        mu['isGood'] = mu['isPres'] & mu['isTight']

        leading_mu = mu[ak.argmax(mu.pt,axis=-1,keepdims=True)]
        leading_mu = leading_mu[leading_mu.isGood]
        
        mu = mu[mu.isGood]
        mu_pres = mu[mu.isPres]

        # Electron selection
        e['isPres']  = isPresElec(e.pt, e.eta, e.dxy, e.dz, e.miniPFRelIso_all, e.sip3d, e.lostHits, minpt=15)
        e['isTight'] = isTightElec(e.pt, e.eta, e.dxy, e.dz, e.miniPFRelIso_all, e.sip3d, e.mvaTTH, e.mvaFall17V2Iso, e.lostHits, e.convVeto, e.tightCharge, e.sieie, e.hoe, e.eInvMinusPInv, minpt=15)
        e['isClean'] = isClean(e, mu, drmin=0.05)
        e['isGood']  = e['isPres'] & e['isTight'] & e['isClean']

        leading_e = e[ak.argmax(e.pt,axis=-1,keepdims=True)]
        leading_e = leading_e[leading_e.isGood]

        e  =  e[e .isGood]
        e_pres = e[e .isPres & e .isClean]

        # Tau selection
        #tau['isPres']  = isPresTau(tau.pt, tau.eta, tau.dxy, tau.dz, tau.leadTkPtOverTauPt, tau.idAntiMu, tau.idAntiEle, tau.rawIso, tau.idDecayModeNewDMs, minpt=20)
        #tau['isClean'] = isClean(tau, e_pres, drmin=0.4) & isClean(tau, mu_pres, drmin=0.4)
        #tau['isGood']  = tau['isPres']# & tau['isClean'], for the moment
        #tau= tau[tau.isGood]

        nElec = ak.num(e)
        nMuon = ak.num(mu)
        #nTau  = ak.num(tau)

        twoLeps   = (nElec+nMuon) == 2
        threeLeps = (nElec+nMuon) == 3
        twoElec   = (nElec == 2)
        twoMuon   = (nMuon == 2)
        e0 = e[ak.argmax(e.pt,axis=-1,keepdims=True)]
        m0 = mu[ak.argmax(mu.pt,axis=-1,keepdims=True)]

<<<<<<< HEAD
        lep = ak.with_name(ak.concatenate([e, mu], axis=1), 'PtEtaPhiMCandidate')
        l0 = lep[ak.argmax(lep.pt,axis=-1,keepdims=True)]

=======
>>>>>>> 2df0772d6b533cee040227fbab3172607875bc17
        # Jet selection
        jetptname = 'pt_nom' if hasattr(j, 'pt_nom') else 'pt'
        j['isGood']  = isTightJet(getattr(j, jetptname), j.eta, j.jetId, j.neHEF, j.neEmEF, j.chHEF, j.chEmEF, j.nConstituents)
        #j['isgood']  = isGoodJet(j.pt, j.eta, j.jetId)
        #j['isclean'] = isClean(j, e, mu)
        j['isClean'] = isClean(j, e, drmin=0.4)& isClean(j, mu, drmin=0.4)# & isClean(j, tau, drmin=0.4)
        goodJets = j[(j.isClean)&(j.isGood)]
        njets = ak.num(goodJets)
        ht = ak.sum(goodJets.pt,axis=-1)
        j0 = goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]
        #nbtags = ak.num(goodJets[goodJets.btagDeepFlavB > 0.2770])
        nbtags = ak.num(goodJets[goodJets.btagDeepB > 0.4941])

        ##################################################################
        ### 2 same-sign leptons
        ##################################################################

        # emu
        singe = e [(nElec==1)&(nMuon==1)&(e .pt>-1)]
        singm = mu[(nElec==1)&(nMuon==1)&(mu.pt>-1)]
        em = ak.cartesian({"e":singe,"m":singm})
        emSSmask = (em.e.charge*em.m.charge>0)
        emSS = em[emSSmask]
        nemSS = len(ak.flatten(emSS))

        emOSmask = (em.e.charge*em.m.charge<0)
        emOS = em[emOSmask]
        nemOS = len(ak.flatten(emOS))

        year = 2018
        lepSF_emSS = GetLeptonSF(mu.pt, mu.eta, 'm', e.pt, e.eta, 'e', year=year)

        # ee and mumu
        # pt>-1 to preserve jagged dimensions
        ee = e [(nElec==2)&(nMuon==0)&(e.pt>-1)]
        mm = mu[(nElec==0)&(nMuon==2)&(mu.pt>-1)]

        eepairs = ak.combinations(ee, 2, fields=["e0","e1"])
        eeSSmask = (eepairs.e0.charge*eepairs.e1.charge>0)
        eeOSmask = (eepairs.e0.charge*eepairs.e1.charge<0)
        eeonZmask  = (np.abs((eepairs.e0+eepairs.e1).mass-91.2)<10)
        eeoffZmask = (eeonZmask==0)

        mmpairs = ak.combinations(mm, 2, fields=["m0","m1"])
        mmSSmask = (mmpairs.m0.charge*mmpairs.m1.charge>0)
        mmOSmask = (mmpairs.m0.charge*mmpairs.m1.charge<0)
        mmonZmask = (np.abs((mmpairs.m0+mmpairs.m1).mass-91.2)<10)
        mmoffZmask = (mmonZmask==0)

        eeSSonZ  = eepairs[eeSSmask &  eeonZmask]
        eeSSoffZ = eepairs[eeSSmask & eeoffZmask]
        mmSSonZ  = mmpairs[mmSSmask &  mmonZmask]
        mmSSoffZ = mmpairs[mmSSmask & mmoffZmask]
        neeSS = len(ak.flatten(eeSSonZ)) + len(ak.flatten(eeSSoffZ))
        nmmSS = len(ak.flatten(mmSSonZ)) + len(ak.flatten(mmSSoffZ))

        eeOSonZ  = eepairs[eeOSmask &  eeonZmask]
        eeOSoffZ = eepairs[eeOSmask & eeoffZmask]
        mmOSonZ  = mmpairs[mmOSmask &  mmonZmask]
        mmOSoffZ = mmpairs[mmOSmask & mmoffZmask]
        eeOS = eepairs[eeOSmask]
        mmOS = mmpairs[mmOSmask]
        neeOS = len(ak.flatten(eeOS))
        nmmOS = len(ak.flatten(mmOS))

        #print('Same-sign events: total [ee, emu, mumu] = %i [%i, %i, %i]'%(neeSS+nemSS+nmmSS, neeSS, nemSS, nmmSS))
        #print('Opposite-sign events: total [ee, emu, mumu] = %i [%i, %i, %i]'%(neeOS+nemOS+nmmOS, neeOS, nemOS, nmmOS))
        
        lepSF_eeSS = GetLeptonSF(eepairs.e0.pt, eepairs.e0.eta, 'e', eepairs.e1.pt, eepairs.e1.eta, 'e', year=year)
        lepSF_mumuSS = GetLeptonSF(mmpairs.m0.pt, mmpairs.m0.eta, 'm', mmpairs.m1.pt, mmpairs.m1.eta, 'm', year=year)
    
        # Cuts
        eeSSmask   = (ak.num(eeSSmask[eeSSmask])>0)
        mmSSmask   = (ak.num(mmSSmask[mmSSmask])>0)
        eeonZmask  = (ak.num(eeonZmask[eeonZmask])>0)
        eeoffZmask = (ak.num(eeoffZmask[eeoffZmask])>0)
        mmonZmask  = (ak.num(mmonZmask[mmonZmask])>0)
        mmoffZmask = (ak.num(mmoffZmask[mmoffZmask])>0)
        emSSmask   = (ak.num(emSSmask[emSSmask])>0)

        eeOSmask   = (ak.num(eeOSmask[eeOSmask])>0)
        mmOSmask   = (ak.num(mmOSmask[mmOSmask])>0)
        emOSmask   = (ak.num(emOSmask[emOSmask])>0)

        CR2LSSjetmask = ((njets==1)|(njets==2)) & (nbtags==1)
        CR2LSSlepmask = (eeSSmask) | (mmSSmask) | (emSSmask)
        CR2LSSmask = (CR2LSSjetmask) & (CR2LSSlepmask)

        CRttbarmask = (emOSmask) & (njets == 2) & (nbtags == 2)
        CRZmask = (((eeOSmask)) | ((mmOSmask))) & (nbtags == 0)

        ##################################################################
        ### 3 leptons
        ##################################################################

        # eem
        muon_eem = mu[(nElec==2)&(nMuon==1)&(mu.pt>-1)]
        elec_eem =  e[(nElec==2)&(nMuon==1)&( e.pt>-1)]
        ee_eem = ak.combinations(elec_eem, 2, fields=["e0", "e1"])

        ee_eemZmask     = (ee_eem.e0.charge*ee_eem.e1.charge<1)&(np.abs((ee_eem.e0+ee_eem.e1).mass-91.2)<10)
        ee_eemOffZmask  = (ee_eem.e0.charge*ee_eem.e1.charge<1)&(np.abs((ee_eem.e0+ee_eem.e1).mass-91.2)>10)
        ee_eemZmask     = (ak.num(ee_eemZmask[ee_eemZmask])>0)
        ee_eemOffZmask  = (ak.num(ee_eemOffZmask[ee_eemOffZmask])>0)

        eepair_eem  = (ee_eem.e0+ee_eem.e1)
        trilep_eem = eepair_eem+muon_eem #ak.cartesian({"e0":ee_eem.e0,"e1":ee_eem.e1, "m":muon_eem})

        lepSF_eem = GetLeptonSF(ee_eem.e0.pt, ee_eem.e0.eta, 'e', ee_eem.e1.pt, ee_eem.e1.eta, 'e', mu.pt, mu.eta, 'm', year)

        # mme
        muon_mme = mu[(nElec==1)&(nMuon==2)&(mu.pt>-1)]
        elec_mme =  e[(nElec==1)&(nMuon==2)&( e.pt>-1)]

        mm_mme = ak.combinations(muon_mme, 2, fields=["m0", "m1"])
        mm_mmeZmask     = (mm_mme.m0.charge*mm_mme.m1.charge<1)&(np.abs((mm_mme.m0+mm_mme.m1).mass-91.2)<10)
        mm_mmeOffZmask  = (mm_mme.m0.charge*mm_mme.m1.charge<1)&(np.abs((mm_mme.m0+mm_mme.m1).mass-91.2)>10)
        mm_mmeZmask     = (ak.num(mm_mmeZmask[mm_mmeZmask])>0)
        mm_mmeOffZmask  = (ak.num(mm_mmeOffZmask[mm_mmeOffZmask])>0)

        mmpair_mme     = (mm_mme.m0+mm_mme.m1)
        trilep_mme     = mmpair_mme+elec_mme
        
        mZ_mme  = mmpair_mme.mass
        mZ_eem  = eepair_eem.mass
        m3l_eem = trilep_eem.mass
        m3l_mme = trilep_mme.mass
        
        lepSF_mme = GetLeptonSF(mm_mme.m0.pt, mm_mme.m0.eta, 'm', mm_mme.m1.pt, mm_mme.m1.eta, 'm', e.pt, e.eta, 'e', year)

        # eee and mmm
        eee =   e[(nElec==3)&(nMuon==0)&( e.pt>-1)] 
        mmm =  mu[(nElec==0)&(nMuon==3)&(mu.pt>-1)] 

        eee_leps = ak.combinations(eee, 3, fields=["e0", "e1", "e2"])
        mmm_leps = ak.combinations(mmm, 3, fields=["m0", "m1", "m2"])
        ee_pairs = ak.combinations(eee, 2, fields=["e0", "e1"])
        mm_pairs = ak.combinations(mmm, 2, fields=["m0", "m1"])
        ee_pairs_index = ak.argcombinations(eee, 2, fields=["e0", "e1"])
        mm_pairs_index = ak.argcombinations(mmm, 2, fields=["m0", "m1"])


        lepSF_eee = GetLeptonSF(eee_leps.e0.pt, eee_leps.e0.eta, 'e', eee_leps.e1.pt, eee_leps.e1.eta, 'e', eee_leps.e2.pt, eee_leps.e2.eta, 'e', year)
        lepSF_mmm = GetLeptonSF(mmm_leps.m0.pt, mmm_leps.m0.eta, 'm', mmm_leps.m1.pt, mmm_leps.m1.eta, 'm', mmm_leps.m2.pt, mmm_leps.m2.eta, 'm', year)
        
        mmSFOS_pairs = mm_pairs[(np.abs(mm_pairs.m0.pdgId) == np.abs(mm_pairs.m1.pdgId)) & (mm_pairs.m0.charge != mm_pairs.m1.charge)]
        offZmask_mm = ak.all(np.abs((mmSFOS_pairs.m0 + mmSFOS_pairs.m1).mass - 91.2)>10., axis=1, keepdims=True) & (ak.num(mmSFOS_pairs)>0)
        onZmask_mm  = ak.any(np.abs((mmSFOS_pairs.m0 + mmSFOS_pairs.m1).mass - 91.2)<10., axis=1, keepdims=True)
      
        eeSFOS_pairs = ee_pairs[(np.abs(ee_pairs.e0.pdgId) == np.abs(ee_pairs.e1.pdgId)) & (ee_pairs.e0.charge != ee_pairs.e1.charge)]
        offZmask_ee = ak.all(np.abs((eeSFOS_pairs.e0 + eeSFOS_pairs.e1).mass - 91.2)>10, axis=1, keepdims=True) & (ak.num(eeSFOS_pairs)>0)
        onZmask_ee  = ak.any(np.abs((eeSFOS_pairs.e0 + eeSFOS_pairs.e1).mass - 91.2)<10, axis=1, keepdims=True)

        # Create masks **for event selection**
        eeeOnZmask  = (ak.num(onZmask_ee[onZmask_ee])>0)
        eeeOffZmask = (ak.num(offZmask_ee[offZmask_ee])>0)
        mmmOnZmask  = (ak.num(onZmask_mm[onZmask_mm])>0)
        mmmOffZmask = (ak.num(offZmask_mm[offZmask_mm])>0)

        CR3Ljetmask = (njets>=1) & (nbtags==0)
        CR3Llepmask = (eeeOnZmask) | (eeeOffZmask) | (mmmOnZmask) | (mmmOffZmask) | (ee_eemZmask) | (ee_eemOffZmask) | (mm_mmeZmask) | (mm_mmeOffZmask)
        CR3Lmask = (CR3Ljetmask) & (CR3Llepmask)        

        # Now we need to create masks for the leptons in order to select leptons from the Z boson candidate (in onZ categories)
        ZeeMask = ak.argmin(np.abs((eeSFOS_pairs.e0 + eeSFOS_pairs.e1).mass - 91.2),axis=1,keepdims=True)
        ZmmMask = ak.argmin(np.abs((mmSFOS_pairs.m0 + mmSFOS_pairs.m1).mass - 91.2),axis=1,keepdims=True)
  
        Zee = eeSFOS_pairs[ZeeMask]
        Zmm = mmSFOS_pairs[ZmmMask]
        eZ0= Zee.e0[ak.num(eeSFOS_pairs)>0]
        eZ1= Zee.e1[ak.num(eeSFOS_pairs)>0]
        eZ = eZ0+eZ1
        mZ0= Zmm.m0[ak.num(mmSFOS_pairs)>0]
        mZ1= Zmm.m1[ak.num(mmSFOS_pairs)>0]
        mZ = mZ0+mZ1
        mZ_eee  = eZ.mass
        mZ_mmm  = mZ.mass

        # And for the W boson
        ZmmIndices = mm_pairs_index[ZmmMask]
        ZeeIndices = ee_pairs_index[ZeeMask]
        eW = eee[~ZeeIndices.e0 | ~ZeeIndices.e1]
        mW = mmm[~ZmmIndices.m0 | ~ZmmIndices.m1]

        triElec = eee_leps.e0+eee_leps.e1+eee_leps.e2
        triMuon = mmm_leps.m0+mmm_leps.m1+mmm_leps.m2
        m3l_eee = triElec.mass
        m3l_mmm = triMuon.mass
    
        # Triggers
        trig_eeSS = passTrigger(events,'ee',isData,dataset)
        trig_mmSS = passTrigger(events,'mm',isData,dataset)
        trig_emSS = passTrigger(events,'em',isData,dataset)
        trig_eee  = passTrigger(events,'eee',isData,dataset)
        trig_mmm  = passTrigger(events,'mmm',isData,dataset)
        trig_eem  = passTrigger(events,'eem',isData,dataset)
        trig_mme  = passTrigger(events,'mme',isData,dataset)

        # MET filters

        # Weights
        genw = np.ones_like(events['event']) if isData else events['genWeight']

        ### We need weights for: normalization, lepSF, triggerSF, pileup, btagSF...
        weights = {}
        for r in ['all', 'ee', 'mm', 'em', 'eee', 'mmm', 'eem', 'mme']:
          weights[r] = coffea.analysis_tools.Weights(len(events))
          weights[r].add('norm',genw if isData else (xsec/sow)*genw)

        weights['ee'].add('lepSF_eeSS', lepSF_eeSS)
        weights['em'].add('lepSF_emSS', lepSF_emSS)
        weights['mm'].add('lepSF_mmSS', lepSF_mumuSS)
        weights['eee'].add('lepSF_eee', lepSF_eee)
        weights['mmm'].add('lepSF_mmm', lepSF_mmm)
        weights['mme'].add('lepSF_mme', lepSF_mme)
        weights['eem'].add('lepSF_eem', lepSF_eem)

        # Extract the EFT quadratic coefficients and optionally use them to calculate the coefficients on the w**2 quartic function
        # eft_coeffs is never Jagged so convert immediately to numpy for ease of use.
        eft_coeffs = ak.to_numpy(events['EFTfitCoefficients']) if hasattr(events, "EFTfitCoefficients") else None
        eft_w2_coeffs = self._eft_helper.calc_w2_coeffs(eft_coeffs) if (self._do_errors and eft_coeffs is not None) else None

        # Selections and cuts
        selections = PackedSelection()
        channels2LSS = ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS']
        selections.add('eeSSonZ',  (eeonZmask)&(eeSSmask)&(trig_eeSS))
        selections.add('eeSSoffZ', (eeoffZmask)&(eeSSmask)&(trig_eeSS))
        selections.add('mmSSonZ',  (mmonZmask)&(mmSSmask)&(trig_mmSS))
        selections.add('mmSSoffZ', (mmoffZmask)&(mmSSmask)&(trig_mmSS))
        selections.add('emSS',     (emSSmask)&(trig_emSS))

        channels2LSS+= ['eeOSonZ', 'eeOSoffZ', 'mmOSonZ', 'mmOSoffZ', 'emOS']
        selections.add('eeOSonZ',  (eeonZmask)&(eeOSmask)&(trig_eeSS))
        selections.add('eeOSoffZ', (eeoffZmask)&(eeOSmask)&(trig_eeSS))
        selections.add('mmOSonZ',  (mmonZmask)&(mmOSmask)&(trig_mmSS))
        selections.add('mmOSoffZ', (mmoffZmask)&(mmOSmask)&(trig_mmSS))
        selections.add('emOS',     (emOSmask)&(trig_emSS))

        channels3L = ['eemSSonZ', 'eemSSoffZ', 'mmeSSonZ', 'mmeSSoffZ']
        selections.add('eemSSonZ',   (ee_eemZmask)&(trig_eem))
        selections.add('eemSSoffZ',  (ee_eemOffZmask)&(trig_eem))
        selections.add('mmeSSonZ',   (mm_mmeZmask)&(trig_mme))
        selections.add('mmeSSoffZ',  (mm_mmeOffZmask)&(trig_mme))

        channels3L += ['eeeSSonZ', 'eeeSSoffZ', 'mmmSSonZ', 'mmmSSoffZ']
        selections.add('eeeSSonZ',   (eeeOnZmask)&(trig_eee))
        selections.add('eeeSSoffZ',  (eeeOffZmask)&(trig_eee))
        selections.add('mmmSSonZ',   (mmmOnZmask)&(trig_mmm))
        selections.add('mmmSSoffZ',  (mmmOffZmask)&(trig_mmm))

        levels = ['base', '2jets', '4jets', '4j1b', '4j2b', 'CR2L', 'CR3L', 'CRttbar', 'CRZ']
        selections.add('base', (nElec+nMuon>=2))
        selections.add('2jets',(njets>=2))
        selections.add('4jets',(njets>=4))
        selections.add('4j1b',(njets>=4)&(nbtags>=1))
        selections.add('4j2b',(njets>=4)&(nbtags>=2))

        selections.add('CR2L', CR2LSSmask)
        selections.add('CR3L', CR3Lmask)
        selections.add('CRttbar', CRttbarmask)
        selections.add('CRZ', CRZmask)

        eeflipmask = ak.fill_none(ak.firsts(eepairs.e0.flipmask | eepairs.e1.flipmask), False)
        mmflipmask = ak.fill_none(ak.firsts(mmpairs.m0.flipmask | mmpairs.m1.flipmask), False)
        emflipmask = ak.fill_none(ak.firsts(em.e.flipmask | em.m.flipmask), False)

        eemflipmask = ak.fill_none(ak.firsts(ee_eem.e0.flipmask | ee_eem.e1.flipmask), False)
        mmeflipmask = ak.fill_none(ak.firsts(mm_mme.m0.flipmask | mm_mme.m1.flipmask), False)
        eeeflipmask = ak.fill_none(ak.firsts(ee_pairs.e0.flipmask | ee_pairs.e1.flipmask), False)
        mmmflipmask = ak.fill_none(ak.firsts(mm_pairs.m0.flipmask | mm_pairs.m1.flipmask), False)
 
        isFlip = {
          'eeSSonZ'  : (eeflipmask)&(eeonZmask) &(eeSSmask),
          'eeSSoffZ' : (eeflipmask)&(eeoffZmask)&(eeSSmask),
          'mmSSonZ'  : (mmflipmask)&(mmonZmask) &(mmSSmask), 
          'mmSSoffZ' : (mmflipmask)&(mmoffZmask)&(mmSSmask), 
          'emSS'     : (emflipmask)&(emSSmask),

          'eeOSonZ'  : (eeflipmask)&(eeonZmask) &(eeOSmask),
          'eeOSoffZ' : (eeflipmask)&(eeoffZmask)&(eeOSmask),
          'mmOSonZ'  : (mmflipmask)&(mmonZmask) &(mmOSmask), 
          'mmOSoffZ' : (mmflipmask)&(mmoffZmask)&(mmOSmask), 
          'emOS'     : (emflipmask)&(emOSmask),

          'eeeSSonZ' : eeeflipmask,
          'eeeSSoffZ': eeeflipmask,
          'mmmSSonZ' : mmmflipmask,
          'mmmSSoffZ': mmmflipmask,
          'eemSSonZ' : eemflipmask,
          'eemSSoffZ': eemflipmask,
          'mmeSSonZ' : mmeflipmask,
          'mmeSSoffZ': mmeflipmask,
        }
        isFlip['eeSSonZ'] = (eeflipmask)&(eeonZmask) &(eeSSmask)
        isPrompt = {}
        for keys in isFlip:
            isPrompt[keys] = (isFlip[keys]==False)

        for keys in isFlip:
            nflip = len(isFlip[keys][isFlip[keys]])
            nprompt = len(isPrompt[keys][isPrompt[keys]])

        # Variables
        invMass_eeSSonZ  = ( eeSSonZ.e0+ eeSSonZ.e1).mass
        invMass_eeSSoffZ = (eeSSoffZ.e0+eeSSoffZ.e1).mass
        invMass_mmSSonZ  = ( mmSSonZ.m0+ mmSSonZ.m1).mass
        invMass_mmSSoffZ = (mmSSoffZ.m0+mmSSoffZ.m1).mass
        invMass_emSS     = (emSS.e+emSS.m).mass

        invMass_eeOSonZ  = ( eeOSonZ.e0+ eeOSonZ.e1).mass
        invMass_eeOSoffZ = (eeOSoffZ.e0+eeOSoffZ.e1).mass
        invMass_mmOSonZ  = ( mmOSonZ.m0+ mmOSonZ.m1).mass
        invMass_mmOSoffZ = (mmOSoffZ.m0+mmOSoffZ.m1).mass
        invMass_emOS     = (emOS.e+emOS.m).mass

        varnames = {}
        varnames['met'] = met.pt
        varnames['ht'] = ht
        varnames['njets'] = njets
        varnames['nbtags'] = nbtags
        varnames['invmass'] = {
          'eeSSonZ'   : invMass_eeSSonZ,
          'eeSSoffZ'  : invMass_eeSSoffZ,
          'mmSSonZ'   : invMass_mmSSonZ,
          'mmSSoffZ'  : invMass_mmSSoffZ,
          'emSS'      : invMass_emSS,
          'eeOSonZ'   : invMass_eeOSonZ,
          'eeOSoffZ'  : invMass_eeOSoffZ,
          'mmOSonZ'   : invMass_mmOSonZ,
          'mmOSoffZ'  : invMass_mmOSoffZ,
          'emOS'      : invMass_emOS,
          'eemSSonZ'  : mZ_eem,
          'eemSSoffZ' : mZ_eem,
          'mmeSSonZ'  : mZ_mme,
          'mmeSSoffZ' : mZ_mme,
          'eeeSSonZ'  : mZ_eee,
          'eeeSSoffZ' : mZ_eee,
          'mmmSSonZ'  : mZ_mmm,
          'mmmSSoffZ' : mZ_mmm,
        }
        varnames['m3l'] = {
          'eemSSonZ'  : m3l_eem,
          'eemSSoffZ' : m3l_eem,
          'mmeSSonZ'  : m3l_mme,
          'mmeSSoffZ' : m3l_mme,
          'eeeSSonZ'  : m3l_eee,
          'eeeSSoffZ' : m3l_eee,
          'mmmSSonZ'  : m3l_mmm,
          'mmmSSoffZ' : m3l_mmm,
        }
        varnames['e0pt' ] = e0.pt
        varnames['e0eta'] = e0.eta
        varnames['m0pt' ] = m0.pt
        varnames['m0eta'] = m0.eta
        varnames['l0pt']  = l0.pt
        varnames['l0eta'] = l0.eta
        varnames['j0pt' ] = j0.pt
        varnames['j0eta'] = j0.eta
<<<<<<< HEAD
        varnames['jpt']   = goodJets.pt
        varnames['jeta']  = goodJets.eta
=======
>>>>>>> 2df0772d6b533cee040227fbab3172607875bc17
        varnames['counts'] = np.ones_like(events['event'])

        # fill Histos
        hout = self.accumulator.identity()
        normweights = weights['all'].weight().flatten() # Why does it not complain about .flatten() here?
        hout['SumOfEFTweights'].fill(sample=dataset, SumOfEFTweights=varnames['counts'], weight=normweights, eft_coeff=eft_coeffs, eft_err_coeff=eft_w2_coeffs)

        for var, v in varnames.items():
         for ch in channels2LSS+channels3L:
          for lev in levels:
            weight = weights[ ch[:3] if (ch.startswith('eee') or ch.startswith('mmm') or ch.startswith('eem') or ch.startswith('mme')) else ch[:2]].weight()
            cuts = [ch] + [lev]
            cut = selections.all(*(cuts))
            weights_flat = weight[cut].flatten() # Why does it not complain about .flatten() here?
            weights_ones = np.ones_like(weights_flat, dtype=np.int)
            eft_coeffs_cut = eft_coeffs[cut] if eft_coeffs is not None else None
            eft_w2_coeffs_cut = eft_w2_coeffs[cut] if eft_w2_coeffs is not None else None
            if var == 'invmass':
              if   ch in ['eeeSSoffZ', 'mmmSSoffZ']: continue
              elif ch in ['eeeSSonZ' , 'mmmSSonZ' ]: continue #values = v[ch]
              else:
                values = ak.flatten(v[ch][cut])
                promptmask = isPrompt[ch][cut]
                flipmask = isFlip[ch][cut]
              hout['invmass'].fill(sample=dataset, channel=ch, cut=lev, invmass=values, weight=weights_flat)
              hout['invmass'].fill(sample=dataset+'_flip', channel=ch, cut=lev, invmass=values[flipmask], weight=weights_flat[flipmask])
            elif var == 'm3l': 
              if ch in ['eeSSonZ', 'eeSSoffZ', 'mmSSonZ', 'mmSSoffZ', 'emSS', 'eeOSonZ', 'eeOSoffZ', 'mmOSonZ', 'mmOSoffZ', 'emOS', 'eeeSSoffZ', 'mmmSSoffZ', 'eeeSSonZ', 'mmmSSonZ']: continue
              values = ak.flatten(v[ch][cut])
<<<<<<< HEAD
              promptmask = isPrompt[ch][cut]
              flipmask = isFlip[ch][cut]
              hout['m3l'].fill(eftweightsvalues, sample=dataset, channel=ch, cut=lev, m3l=values[promptmask], weight=weights_flat[promptmask])
              hout['m3l'].fill(eftweightsvalues, sample=dataset+'_flip', channel=ch, cut=lev, m3l=values[flipmask], weight=weights_flat[flipmask])
            else:
              values = v[cut]
              promptmask = isPrompt[ch][cut]
              flipmask = isFlip[ch][cut]
              if   var == 'ht'    :
                hout[var].fill(eftweightsvalues, ht=values[promptmask], sample=dataset, channel=ch, cut=lev, weight=weights_flat[promptmask])
                hout[var].fill(eftweightsvalues, ht=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
              elif var == 'met'   :
                hout[var].fill(eftweightsvalues, met=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
                hout[var].fill(eftweightsvalues, met=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
              elif var == 'njets' :
                hout[var].fill(eftweightsvalues, njets=values[promptmask], sample=dataset, channel=ch, cut=lev, weight=weights_flat[promptmask])
                hout[var].fill(eftweightsvalues, njets=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
              elif var == 'nbtags': 
                hout[var].fill(eftweightsvalues, nbtags=values[promptmask], sample=dataset, channel=ch, cut=lev, weight=weights_flat[promptmask])
                hout[var].fill(eftweightsvalues, nbtags=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
              elif var == 'counts': 
                hout[var].fill(counts=values[promptmask], sample=dataset, channel=ch, cut=lev, weight=weights_ones[promptmask])
                hout[var].fill(counts=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_ones[flipmask])
              elif var == 'j0eta' :
                if lev in ['base', 'CRZ']: continue
                values = ak.flatten(values)
                promptmask = isPrompt[ch][cut]
                flipmask = isFlip[ch][cut]
                hout[var].fill(eftweightsvalues, j0eta=values[promptmask], sample=dataset, channel=ch, cut=lev, weight=weights_flat[promptmask])
                hout[var].fill(eftweightsvalues, j0eta=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
=======
              hout['m3l'].fill(sample=dataset, channel=ch, cut=lev, m3l=values, weight=weights_flat,eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
            else:
              values = v[cut] 
              if   var == 'ht'    : hout[var].fill(ht=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat, eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
              elif var == 'met'   : hout[var].fill(met=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat, eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
              elif var == 'njets' : hout[var].fill(njets=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat, eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
              elif var == 'nbtags': hout[var].fill(nbtags=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat, eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
              elif var == 'counts': hout[var].fill(counts=values, sample=dataset, channel=ch, cut=lev, weight=weights_ones)
              elif var == 'j0eta' : 
                if lev == 'base': continue
                values = ak.flatten(values)
                #values=np.asarray(values)
                hout[var].fill(j0eta=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat, eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
>>>>>>> 2df0772d6b533cee040227fbab3172607875bc17
              elif var == 'e0pt'  : 
                if ch in ['mmSSonZ', 'mmSSoffZ', 'mmOSonZ', 'mmOSoffZ', 'mmmSSoffZ', 'mmmSSonZ']: continue
                values = ak.flatten(values)
<<<<<<< HEAD
                promptmask = isPrompt[ch][cut]
                flipmask = isFlip[ch][cut]
                hout[var].fill(eftweightsvalues, e0pt=values[promptmask], sample=dataset, channel=ch, cut=lev, weight=weights_flat[promptmask])
                hout[var].fill(eftweightsvalues, e0pt=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
=======
                #values=np.asarray(values)
                hout[var].fill(e0pt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat, eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut) # Crashing here, not sure why. Related to values?
>>>>>>> 2df0772d6b533cee040227fbab3172607875bc17
              elif var == 'm0pt'  : 
                if ch in ['eeSSonZ', 'eeSSoffZ', 'eeOSonZ', 'eeOSoffZ', 'eeeSSoffZ', 'eeeSSonZ']: continue
                values = ak.flatten(values)
<<<<<<< HEAD
                promptmask = isPrompt[ch][cut]
                flipmask = isFlip[ch][cut]
                hout[var].fill(eftweightsvalues, m0pt=values[promptmask], sample=dataset, channel=ch, cut=lev, weight=weights_flat[promptmask])
                hout[var].fill(eftweightsvalues, m0pt=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
=======
                #values=np.asarray(values)
                hout[var].fill(m0pt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat, eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
>>>>>>> 2df0772d6b533cee040227fbab3172607875bc17
              elif var == 'e0eta' : 
                if ch in ['mmSSonZ', 'mmSSoffZ', 'mmOSonZ', 'mmOSoffZ', 'mmmSSoffZ', 'mmmSSonZ']: continue
                values = ak.flatten(values)
<<<<<<< HEAD
                promptmask = isPrompt[ch][cut]
                flipmask = isFlip[ch][cut]
                hout[var].fill(eftweightsvalues, e0eta=values[promptmask], sample=dataset, channel=ch, cut=lev, weight=weights_flat[promptmask])
                hout[var].fill(eftweightsvalues, e0eta=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
=======
                #values=np.asarray(values)
                hout[var].fill(e0eta=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat, eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
>>>>>>> 2df0772d6b533cee040227fbab3172607875bc17
              elif var == 'm0eta':
                if ch in ['eeSSonZ', 'eeSSoffZ', 'eeOSonZ', 'eeOSoffZ', 'eeeSSoffZ', 'eeeSSonZ']: continue
                values = ak.flatten(values)
<<<<<<< HEAD
                promptmask = isPrompt[ch][cut]
                flipmask = isFlip[ch][cut]
                hout[var].fill(eftweightsvalues, m0eta=values[promptmask], sample=dataset, channel=ch, cut=lev, weight=weights_flat[promptmask])
                hout[var].fill(eftweightsvalues, m0eta=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
=======
                #values=np.asarray(values)
                hout[var].fill(m0eta=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat, eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
>>>>>>> 2df0772d6b533cee040227fbab3172607875bc17
              elif var == 'j0pt'  : 
                if lev in ['base', 'CRZ']: continue
                values = ak.flatten(values)
<<<<<<< HEAD
                promptmask = isPrompt[ch][cut]
                flipmask = isFlip[ch][cut]
                hout[var].fill(eftweightsvalues, j0pt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
                hout[var].fill(eftweightsvalues, j0pt=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
              elif var == 'l0pt'  :
                values = ak.flatten(values)
                promptmask = isPrompt[ch][cut]
                flipmask = isFlip[ch][cut]
                hout[var].fill(eftweightsvalues, l0pt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
                hout[var].fill(eftweightsvalues, l0pt=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
                #if (lev == 'CRZ') & (ch == 'mmOSonZ') :  print(weights_flat)
              elif var == 'l0eta' :
                values = ak.flatten(values)
                promptmask = isPrompt[ch][cut]
                flipmask = isFlip[ch][cut]
                hout[var].fill(eftweightsvalues, l0eta=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat)
                hout[var].fill(eftweightsvalues, l0eta=values[flipmask], sample=dataset+'_flip', channel=ch, cut=lev, weight=weights_flat[flipmask])
              elif var == 'jpt'  : 
                if lev in ['base', 'CRZ']: continue
                weights_flat_jet = ak.broadcast_arrays(weights_flat, values)
                #eftweightsvalues_jet = ak.broadcast_arrays(eftweightsvalues, values)
                values = ak.flatten(values)
                weights_flat_jet = ak.flatten(ak.flatten(weights_flat_jet))[:len(values)]
                #eftweightsvalues_jet = ak.flatten(eftweightsvalues_jet)
                promptmask = isPrompt[ch][cut]
                flipmask = isFlip[ch][cut]
                hout[var].fill(eftweightsvalues, jpt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat_jet)
              elif var == 'jeta'  : 
                if lev in ['base', 'CRZ']: continue
                weights_flat_jet = ak.broadcast_arrays(weights_flat, values)
                #eftweightsvalues_jet = ak.broadcast_arrays(eftweightsvalues, values)
                values = ak.flatten(values)
                weights_flat_jet = ak.flatten(ak.flatten(weights_flat_jet))[:len(values)]
                #eftweightsvalues_jet = ak.flatten(eftweightsvalues_jet)
                promptmask = isPrompt[ch][cut]
                flipmask = isFlip[ch][cut]
                hout[var].fill(eftweightsvalues, jeta=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat_jet)
=======
                #values=np.asarray(values)
                hout[var].fill(j0pt=values, sample=dataset, channel=ch, cut=lev, weight=weights_flat, eft_coeff=eft_coeffs_cut, eft_err_coeff=eft_w2_coeffs_cut)
>>>>>>> 2df0772d6b533cee040227fbab3172607875bc17
        return hout

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    # Load the .coffea files
    outpath= './coffeaFiles/'
    samples     = load(outpath+'samples.coffea')
    topprocessor = AnalysisProcessor(samples)
