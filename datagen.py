#!/usr/bin/env python3
import sys
import os
import yaml
import pprint
import time
import warnings

from typing import Optional, Union
from pathlib import Path
from multiprocessing import Process, Queue

import numpy as np


def export_torchscript_models(outputDir, queue):
    with open("model_config.txt") as f:
        lines = f.readlines()

    lines = [ os.path.expandvars(l.rstrip()) for l in lines]

    assert len(lines) == 3
    assert os.path.exists(lines[0])
    assert os.path.exists(lines[1])
    assert os.path.exists(lines[2])


    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    import torch

    sys.path.append("/home/iwsatlas1/bhuth/exatrkx/Tracking-ML-Exa.TrkX/Pipelines/TrackML_Example")
    from LightningModules.Embedding.Models.layerless_embedding import LayerlessEmbedding
    from LightningModules.Filter.Models.vanilla_filter import VanillaFilter
    from LightningModules.GNN.Models.interaction_gnn import InteractionGNN

    sample_x = torch.rand((100,3)).cuda()
    sample_edge_index = torch.randint(0, len(sample_x), (1000,2)).cuda()

    def dummy_checkpoint(f, *args):
        return f(*args)

    def make_torchscript_model(ModelClass, checkpoint_path, sample_input):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with torch.no_grad():
                model = ModelClass.load_from_checkpoint(checkpoint_path)
                model.checkpoint = dummy_checkpoint
                return torch.jit.trace(model.cuda(), sample_input), model.hparams

    print("Load Embedding model from '{}'".format(lines[0]))
    emb_model, emb_hparams = make_torchscript_model(LayerlessEmbedding, lines[0], (sample_x,))
    emb_model.save(outputDir / "embed.pt")

    print("Load Filter model from '{}'".format(lines[1]))
    filt_model, filt_hparams = make_torchscript_model(VanillaFilter, lines[1], (sample_x, sample_edge_index))
    filt_model.save(outputDir / "filter.pt")

    print("Load GNN model from '{}'".format(lines[2]))
    gnn_model, gnn_hparams = make_torchscript_model(InteractionGNN, lines[2], (sample_x, sample_edge_index))
    gnn_model.save(outputDir / "gnn.pt")

    hyperparameter_dict = queue.get()
    hyperparameter_dict["spacepointFeatures"] = 3
    hyperparameter_dict["embeddingDim"] = emb_hparams["emb_dim"]
    hyperparameter_dict["rVal"] = emb_hparams["r_test"]
    hyperparameter_dict["knnVal"] = 500 #filt_hparams["knn"]
    hyperparameter_dict["filterCut"] = filt_hparams["filter_cut"]
    hyperparameter_dict["n_chunks"] = 5
    hyperparameter_dict["edgeCut"] = gnn_hparams["edge_cut"]
    queue.put(hyperparameter_dict)

#########################
# Command line handling #
#########################

def usage():
    print("Usage: {} <generate|reconstruct> (<target_path>) <n_events>".format(sys.argv[0]))
    
if len(sys.argv) < 3 or len(sys.argv) > 4:
    usage()
    exit(1)

try:
    mode = sys.argv[1]
    assert mode == "generate" or mode == "reconstruct"
except:
    usage()
    exit(1)

try:
    n_events=int(sys.argv[-1])
except:
    usage()
    exit(1)

base_dir = Path(os.path.dirname(__file__))

if len(sys.argv) == 4:
    outputDir = Path(sys.argv[2])
    if not os.path.exists(outputDir):
        outputDir.mkdir()
    if mode == "generate" and not os.path.exists(outputDir/"train_all"):
        (outputDir/"train_all").mkdir()
else:
    if mode == "generate":
        outputDir = Path.cwd() / "training_data"
    elif mode == "reconstruct":
        outputDir = Path.cwd() / "reconstruction"

############################
# Load models if necessary #
############################

if mode == "reconstruct":
    queue = Queue()
    queue.put({})

    # Use a process here to avoid import conflicts of acts and torch
    p = Process(target=export_torchscript_models, args=(outputDir / "torchscript", queue))
    p.start()
    p.join()

    exaTrkXConfig = queue.get()

    exaTrkXConfig["verbose"] = False
    exaTrkXConfig["modelDir"] = str(outputDir)

#####################
# Import ACTS stuff #
#####################

import acts
import acts.examples

u = acts.UnitConstants

from common import getOpenDataDetector, getOpenDataDetectorDirectory
from particle_gun import addParticleGun, EtaConfig, ParticleConfig, MomentumConfig
from pythia8 import addPythia8
from digitization import addDigitization
from fatras import addFatras
from exatrkx import addExaTrkx
from seeding import addSeeding, SeedingAlgorithm, TruthSeedRanges, TrackParamsEstimationConfig, SeedfinderConfigArg
from ckf_tracks import addCKFTracks, CKFPerformanceConfig


###########################
# Load Open Data Detector #
###########################

oddDir = getOpenDataDetectorDirectory()

oddMaterialMap = oddDir / "data/odd-material-maps.root"
assert os.path.exists(oddMaterialMap)

oddDigiConfigSmear = oddDir / "config/odd-digi-smearing-config.json"
assert os.path.exists(oddDigiConfigSmear)

oddDigiConfigGeometric = base_dir / "detector/odd-digi-geometry-config.json"
assert os.path.exists(oddDigiConfigGeometric)

oddDigiConfigTrue = base_dir / "detector/odd-digi-true-config.json"
assert os.path.exists(oddDigiConfigTrue)

oddGeoSelectionExaTrkX = base_dir / "detector/odd-geo-selection-whole-detector.json"
assert os.path.exists(oddGeoSelectionExaTrkX)

oddGeoSelectionSeeding = oddDir / "config/odd-seeding-config.json"
assert os.path.exists(oddGeoSelectionSeeding)

oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
detector, trackingGeometry, decorators = getOpenDataDetector(mdecorator=oddMaterialDeco)


if not "GENERIC_DETECTOR" in os.environ:
    oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)
    detector, trackingGeometry, decorators = getOpenDataDetector(mdecorator=oddMaterialDeco)
else:
    print("!!! Use generic detector instead of ODD !!!")
    detector, trackingGeometry, decorators = acts.examples.GenericDetector.create()
    oddDigiConfigGeometric = "/home/iwsatlas1/bhuth/acts/Examples/Algorithms/Digitization/share/default-smearing-config-generic.json"
    oddGeoSelectionSeeding = "/home/iwsatlas1/bhuth/acts/Examples/Algorithms/TrackFinding/share/geoSelection-genericDetector.json"


#############################
# Prepare and run sequencer #
#############################

if mode == "generate":
    n_jobs=-1
else:
    n_jobs=1

field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

s = acts.examples.Sequencer(events=n_events, numThreads=n_jobs)

s.config.logLevel = acts.logging.INFO

try:
    rnd_seed = int(os.environ["SEED"])
except:
    rnd_seed = int(time.time())

print("SEED:",rnd_seed)
rnd = acts.examples.RandomNumbers(seed=rnd_seed)

#s = addParticleGun(
    #s,
    #MomentumConfig(1.0 * u.GeV, 10.0 * u.GeV, True),
    #EtaConfig(-3.0, 3.0, True),
    #ParticleConfig(1, acts.PdgParticle.eMuon, True),
    #rnd=rnd,
    #multiplicity=50,
#)

s = addPythia8(
    s,
    rnd=rnd,
    outputDirCsv=str(outputDir/"train_all"),
)

s = addFatras(
    s,
    trackingGeometry,
    field,
    outputDirCsv=None,
    rnd=rnd,
)

s = addDigitization(
    s,
    trackingGeometry,
    field,
    #digiConfigFile=oddDigiConfigSmear,
    #digiConfigFile=oddDigiConfigTrue,
    digiConfigFile=oddDigiConfigGeometric,
    outputDirRoot=None,
    outputDirCsv=str(outputDir/"train_all"),
    rnd=rnd,
)


s.addWriter(
    acts.examples.CsvSimHitWriter(
        level=acts.logging.INFO,
        inputSimHits="simhits",
        outputDir=str(outputDir/"train_all"),
        outputStem="truth",
    )
)

s.addWriter(
    acts.examples.CsvMeasurementWriter(
        level=acts.logging.INFO,
        inputMeasurements="measurements",
        inputClusters="clusters",
        inputSimHits="simhits",
        inputMeasurementSimHitsMap="measurement_simhits_map",
        outputDir=str(outputDir/"train_all"),
    )
)

s.addWriter(acts.examples.CsvTrackingGeometryWriter(
    level=acts.logging.INFO,
    trackingGeometry=trackingGeometry,
    outputDir=str(outputDir),
    writePerEvent=False,
))

if mode == "reconstruct":
    #################################
    # ExaTrkX + performance writing #
    #################################
    if False:
        s.addAlgorithm(
            acts.examples.SpacePointMaker(
                level=acts.logging.INFO,
                inputSourceLinks="sourcelinks",
                inputMeasurements="measurements",
                outputSpacePoints="exatrkx_spacepoints",
                trackingGeometry=trackingGeometry,
                geometrySelection=acts.examples.readJsonGeometryList(
                    str(oddGeoSelectionExaTrkX)
                ),
            )
        )

        print("Exa.TrkX Configuration")
        pprint.pprint(exaTrkXConfig, indent=4)

        s.addAlgorithm(
            acts.examples.TrackFindingAlgorithmExaTrkX(
                level=acts.logging.INFO,
                inputSpacePoints="exatrkx_spacepoints",
                outputProtoTracks="exatrkx_prototracks",
                trackFinderML=acts.examples.ExaTrkXTrackFinding(**exaTrkXConfig),
                rScale = 1000.,
                phiScale = np.pi,
                zScale = 1000.,
            )
        )

        s.addWriter(
            acts.examples.TrackFinderPerformanceWriter(
                level=acts.logging.INFO,
                inputProtoTracks="exatrkx_prototracks",
                inputParticles="particles_initial",  # the original selected particles after digitization
                inputMeasurementParticlesMap="measurement_particles_map",
                filePath=str(outputDir / "exatrkx_performance.root"),
            )
        )

    #############################################
    # Truth track finding + Performance writing #
    #############################################
    if False:
        s.addAlgorithm(
            acts.examples.TruthSeedSelector(
                level=acts.logging.INFO,
                ptMin=0 * u.MeV,
                nHitsMin=4,
                inputParticles="particles_initial",
                inputMeasurementParticlesMap="measurement_particles_map",
                outputParticles="particles_seed_selected",
            )
        )

        s.addAlgorithm(
            acts.examples.TruthTrackFinder(
                level=acts.logging.INFO,
                inputParticles="particles_seed_selected",
                inputMeasurementParticlesMap="measurement_particles_map",
                outputProtoTracks="truth_prototracks",
            )
        )

        s.addWriter(
            acts.examples.TrackFinderPerformanceWriter(
                level=acts.logging.INFO,
                inputProtoTracks="truth_prototracks",
                inputParticles="particles_initial",  # the original selected particles after digitization
                inputMeasurementParticlesMap="measurement_particles_map",
                filePath=str(outputDir / "truth_track_finding_performance.root"),
            )
        )
        
    #############################
    # CKF + Performance writing #
    #############################
    if True:
        s = addSeeding(
            s,
            trackingGeometry,
            field,
            #TruthSeedRanges(pt=(1.0 * u.GeV, None), eta=(-2.7, 2.7), nHits=(9, None)),
            TruthSeedRanges(pt=(1.0 * u.GeV, None), eta=(-2.7, 2.7), nHits=(9, None)),
            geoSelectionConfigFile=oddGeoSelectionSeeding,
            initialVarInflation=[100, 100, 100, 100, 100, 100],
            seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
        )
        
        s.addAlgorithm(
            acts.examples.TrackFindingAlgorithm(
                level=acts.logging.INFO,
                measurementSelectorCfg=acts.MeasurementSelector.Config(
                    [(acts.GeometryIdentifier(), ([], [15.0], [10]))]
                ),
                inputMeasurements="measurements",
                inputSourceLinks="sourcelinks",
                inputInitialTrackParameters="estimatedparameters",
                outputTrajectories="ckf_trajectories",
                findTracks=acts.examples.TrackFindingAlgorithm.makeTrackFinderFunction(
                    trackingGeometry, field
                ),
            )
        )
        
        s.addAlgorithm(
            acts.examples.TrajectoriesToPrototracks(
                level=acts.logging.INFO,
                inputTrajectories="ckf_trajectories",
                outputPrototracks="ckf_prototracks",
            )
        )

        s.addWriter(
            acts.examples.TrackFinderPerformanceWriter(
                level=acts.logging.INFO,
                inputProtoTracks="ckf_prototracks",
                inputParticles="particles_initial",  # the original selected particles after digitization
                inputMeasurementParticlesMap="measurement_particles_map",
                filePath=str(outputDir / "ckf_track_finding_performance.root"),
            )
        )

    #kalmanOptions = {
        #"multipleScattering": True,
        #"energyLoss": True,
        #"reverseFilteringMomThreshold": 0.0,
        #"freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
    #}

    #fitAlg = acts.examples.TrackFittingAlgorithm(
        #level=acts.logging.INFO,
        #inputMeasurements="measurements",
        #inputSourceLinks="sourcelinks",
        #inputProtoTracks="protoTracks",
        #inputInitialTrackParameters="estimatedparameters",
        #outputTrajectories="trajectories",
        #directNavigation=False,
        #pickTrack=-1,
        #trackingGeometry=trackingGeometry,
        #dFit=acts.examples.TrackFittingAlgorithm.makeKalmanFitterFunction(
            #field, **kalmanOptions
        #),
        #fit=acts.examples.TrackFittingAlgorithm.makeKalmanFitterFunction(
            #trackingGeometry, field, **kalmanOptions
        #),
    #)
    #s.addAlgorithm(fitAlg)

    

    #s.addWriter(
        #acts.examples.TrackFitterPerformanceWriter(
            #level=acts.logging.INFO,
            #inputTrajectories="trajectories",
            #inputParticles="truth_seeds_selected",
            #inputMeasurementParticlesMap="measurement_particles_map",
            #filePath=str(outputDir / "reconstruction" / "performance_track_fitter.root"),
        #)
    #)



s.run()
