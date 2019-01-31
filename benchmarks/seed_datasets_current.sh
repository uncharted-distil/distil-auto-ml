#!/bin/bash

# seed_datasets_current.sh

mkdir -p results/seed_datasets_current

# --
# Tabular

# classification
python -m exline.main --prob-name 1491_one_hundred_plants_margin
python -m exline.main --prob-name 185_baseball
python -m exline.main --prob-name LL0_acled         # w/ text data
python -m exline.main --prob-name LL0_acled_reduced # w/ text data
python -m exline.main --prob-name 27_wordLevels     # need manual feature creation
python -m exline.main --prob-name 38_sick
python -m exline.main --prob-name 4550_MiceProtein
python -m exline.main --prob-name 1567_poker_hand
python -m exline.main --prob-name 57_hypothyroid
python -m exline.main --prob-name uu4_SPECT
python -m exline.main --prob-name 313_spectrometer      # small
python -m exline.main --prob-name LL0_1100_popularkids  # small
python -m exline.main --prob-name LL0_186_braziltourism # small
python -m exline.main --prob-name LL1_726_TIDY_GPS_carpool_bus_service_rating_prediction # *** no beat baseline

# regression (all pretty small)
python -m exline.main --prob-name  196_autoMpg
python -m exline.main --prob-name  534_cps_85_wages
python -m exline.main --prob-name  LL0_207_autoPrice
python -m exline.main --prob-name  26_radon_seed   # extra trees > randomforest, but not in OOB.  Should be using metadata.
python -m exline.main --prob-name  299_libras_move # mislabeled as regression, is actually classification

python -m exline.main --prob-name  LL1_336_MS_Geolife_transport_mode_prediction                  # trigger "final_subset"
python -m exline.main --prob-name  LL1_336_MS_Geolife_transport_mode_prediction_separate_lat_lon # trigger "final_subset"

# --
# Timeseries

python -m exline.main --prob-name 66_chlorineConcentration # +

# --
# Graph matching

# python -m exline.main --prob-name 49_facebook # +

# !! SGM doesn't beat baseline, but we get perfect score because of "null transform" checking
# I suspect there's something wrong with the baseline.

# --
# Vertex nomination

# python -m exline.main --prob-name LL1_net_nomination_seed # +

# !! This is not a great VN problem.  Don't have any others.  Should add CORA or something.

# --
# Recommender system

# python -m exline.main --prob-name 60_jester # +

# !! Simple ensemble is reliably better than single model, though hyperband could do better I'm sure.

# --
# Link prediction

# python -m exline.main --prob-name 59_umls # +

# --
# Question answering

# python -m exline.main --prob-name 32_wikiqa # =

# --
# Clustering

# python -m exline.main --prob-name 1491_one_hundred_plants_margin_clust
# !! kmeans, same as baseline, but there's no training data so not really any way to tune

# python -m exline.main --prob-name 6_70_com_amazon
# python -m exline.main --prob-name 6_86_com_DBLP
# !! bigclam or "null clustering" hack.  Task is ill defined, I think.

# --
# Misc

# timeseries that are dataframes
python -m exline.main --prob-name LL1_736_stock_market

# multilabel, multitable regression
# python -m exline.main --prob-name uu2_gp_hyperparameter_estimation    # !!!! our metric is wrong for one of these
# python -m exline.main --prob-name uu2_gp_hyperparameter_estimation_v2
 
# audio classification
python -m exline.main --prob-name 31_urbansound

# sparse timeseries
# python -m exline.main --prob-name uu1_datasmash

# text classification
# python -m exline.main --prob-name 30_personae

CUDA_VISIBLE_DEVICES=6 python -m exline.main --prob-name 22_handgeometry

# --
# No support

# LL1_3476_HMDB_actio_recognition - video (classification?)
# LL1_penn_fudan_pedestrian - image (object detection)

# --
# Punt

# uu3_world_development_indicators - skipping multi table
# LL1_EDGELIST_net_nomination_seed - skipping support for edgelist graph format -- is this OK?
# 56_sunspots - directory is broken?
# DS01876 - broken?