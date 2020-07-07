import os
import subprocess
import sys
import unittest
import time
from d3m.metadata.problem import PerformanceMetricBase, PerformanceMetric
import typing
import pandas as pd
import signal

#test
SEEDATASETS = os.getenv("D3MINPUTDIR", "seed_datasets_current")
D3MOUTPUTDIR = "output"
D3MINPUTDIR = SEEDATASETS
D3MSTATICDIR = "/static"

env = {}
env.update(os.environ.copy())
env['PYTHONPATH'] = ':'.join(sys.path)
env['D3MLOCAL'] = "True"

lower_is_better = {
    "ACCURACY": False,
    "PRECISION": False,
    "RECALL": False,
    "F1": False,
    "F1_MICRO": False,
    "F1_MACRO": False,
    "MEAN_SQUARED_ERROR": True,
    "ROOT_MEAN_SQUARED_ERROR": True,
    "MEAN_ABSOLUTE_ERROR": True,
    "R_SQUARED": True,
    "NORMALIZED_MUTUAL_INFORMATION": False,
    "JACCARD_SIMILARITY_SCORE": False,
    "PRECISION_AT_TOP_K": False,
    "OBJECT_DETECTION_AVERAGE_PRECISION": False,
    "HAMMING_LOSS": True,
}

problem_thresholds = {'LL1_Haptics_MIN_METADATA': 0.41,  # F1_MACRO
                      "124_174_cifar10_MIN_METADATA": 0.8,  # Accuracy
                      "124_188_usps_MIN_METADATA": 0.9,  # Accuracy
                      "124_214_coil20_MIN_METADATA": 0.95,  # Accuracy
                      "124_95_uc_merced_land_use_MIN_METADATA": 0.85,  # Accuracy
                      "1491_one_hundred_plants_margin_MIN_METADATA": 0.8,  # F1_MACRO
                      "1567_poker_hand_MIN_METADATA": 0.0,  # F1_MACRO todo what score do we get?
                      "185_baseball_MIN_METADATA": 0.7,  # F1_MACRO
                      "196_autoMpg_MIN_METADATA": 7,  # MSE,
                      "22_handgeometry_MIN_METADATA": 0.3,  # MSE,
                      "26_radon_seed_MIN_METADATA": 0.05,  # RMSE
                      "27_wordLevels_MIN_METADATA": 0.15,  # F1_MACRO
                      "299_libras_move_MIN_METADATA": 0.75,  # Accuracy
                      "30_personae_MIN_METADATA": 0.6,  # F1_MACRO
                      "313_spectrometer_MIN_METADATA": 0.4,  # F1_MACRO,
                      "31_urbansound_MIN_METADATA": 0.9,  # Accuracy,
                      "32_fma_MIN_METADATA": 0,  # Accuracy
                      "32_wikiqa_MIN_METADATA": 0.45,  # F1
                      "38_sick_MIN_METADATA": 0.9,  # F1
                      "4550_MiceProtein_MIN_METADATA": 1,  # F1
                      "49_facebook_MIN_METADATA": 0.85,  # Accuracy
                      "534_cps_85_wages_MIN_METADATA": 20,  # MSE
                      "56_sunspots_MIN_METADATA": 55,  # RMSE
                      "56_sunspots_monthly_MIN_METADATA": 60,  # RMSE
                      "57_hypothyroid_MIN_METADATA": 0.98,  # F1_MACRO
                      "59_LP_karate_MIN_METADATA": 0.4,  # Accuracy,
                      "59_umls_MIN_METADATA": 0.93,  # Accuracy
                      "60_jester_MIN_METADATA": 99,  # MAE TODO what score?
                      "66_chlorineConcentration_MIN_METADATA": 0.75,  # F1_MACRO
                      "6_70_com_amazon_MIN_METADATA": 0.8,  # NORMALIZED_MUTUAL_INFORMATION
                      "6_86_com_DBLP_MIN_METADATA": 0.7,  # NORMALIZED_MUTUAL_INFORMATION
                      "kaggle_music_hackathon_MIN_METADATA": 99,  # RMSE TODO what score
                      "LL0_1100_popularkids_MIN_METADATA": 0.35,  # F1_MACRO
                      "LL0_186_braziltourism_MIN_METADATA": 0.17,  # F1_MACRO
                      "LL0_207_autoPrice_MIN_METADATA": 5000000,  # MSE
                      "LL0_acled_reduced_MIN_METADATA": 0.9,  # Accuracy
                      "LL1_336_MS_Geolife_transport_mode_prediction_MIN_METADATA": 0.9,  # Accuracy
                      "LL1_336_MS_Geolife_transport_mode_prediction_separate_lat_lon_MIN_METADATA": 0.9,  # Accuracy
                      "LL1_50words_MIN_METADATA": 0.43,  # F1_MACRO
                      "LL1_726_TIDY_GPS_carpool_bus_service_rating_prediction_MIN_METADATA": 0.35,  # F1_MACRO
                      "LL1_736_population_spawn_MIN_METADATA": 1600,  # MAE
                      "LL1_736_population_spawn_simpler_MIN_METADATA": 1350,  # MAE
                      "LL1_736_stock_market_MIN_METADATA": 1.6,  # MAE
                      "LL1_Adiac_MIN_METADATA": 0.65,  # F1_MACRO
                      "LL1_ArrowHead_MIN_METADATA": 0.65,  # F1_MACRO
                      "LL1_bn_fly_drosophila_medulla_net_MIN_METADATA": 0.85,  # NORMALIZED_MUTUAL_INFORMATION
                      "LL1_CinC_ECG_torso_MIN_METADATA": 0.5,  # F1_MACRO
                      "LL1_Cricket_Y_MIN_METADATA": 0.5,  # F1_MACRO
                      "LL1_crime_chicago_MIN_METADATA": 0.65,  # Accuracy,
                      "LL1_DIC28_net_MIN_METADATA": 0.75,  # Accuracy
                      "LL1_ECG200_MIN_METADATA": 0.88,  # F1
                      "LL1_EDGELIST_net_nomination_seed_MIN_METADATA": 0.68,  # Accuracy
                      "LL1_ElectricDevices_MIN_METADATA": 0.45,  # F1_MACRO
                      "LL1_FaceFour_MIN_METADATA": 0.85,  # F1_MACRO
                      "LL1_FISH_MIN_METADATA": 0.7,  # F1_MACRO
                      "LL1_FordA_MIN_METADATA": 0.65,  # F1
                      "LL1_GS_process_classification_tabular_MIN_METADATA": 0.125,  # F1 TODO these are low
                      "LL1_GS_process_classification_text_MIN_METADATA": 0.1,  # F1 TODO these are low
                      "LL1_GT_actor_group_association_prediction_MIN_METADATA": 0.2,  # RMSE
                      "LL1_HandOutlines_MIN_METADATA": 0.88,  # F1
                      "LL1_ItalyPowerDemand_MIN_METADATA": 0.95,  # F1
                      "LL1_Meat_MIN_METADATA": 0.91,  # F1_MACRO
                      "LL1_net_nomination_seed_MIN_METADATA": 0.77,  # Accuracy
                      "LL1_OSULeaf_MIN_METADATA": 0.47,  # F1_MACRO
                      "LL1_penn_fudan_pedestrian_MIN_METADATA": 0.94,  # OBJECT_DETECTION_AVERAGE_PRECISION
                      "LL1_PHEM_Monthly_Malnutrition_MIN_METADATA": 830,  # MAE
                      "LL1_PHEM_weeklyData_malnutrition_MIN_METADATA": 3.5,  # MAE
                      "LL1_retail_sales_total_MIN_METADATA": 2150,  # RMSE
                      "LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA": 85,  # MAE
                      "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA": 190,  # MAE
                      "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA": 110,  # MAE
                      "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA": 70000,  # MAE
                      "LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA": 0.9,  # MAE
                      "LL1_tidy_terra_panicle_detection_MIN_METADATA": 0.26,  # OBJECT_DETECTION_AVERAGE_PRECISION
                      "LL1_TXT_CLS_3746_newsgroup_MIN_METADATA": 0.06,  # F1 MACRO
                      "LL1_TXT_CLS_airline_opinion_MIN_METADATA": 0.62,  # Accuracy,
                      "LL1_TXT_CLS_apple_products_sentiment_MIN_METADATA": 0.55,  # Accuracy
                      "LL1_VTXC_1343_cora_MIN_METADATA": 0.06,  # F1_MACRO
                      "LL1_VTXC_1369_synthetic_MIN_METADATA": 0.20,  # F1_MACRO
                      "loan_status_MIN_METADATA": 0.35,  # F1_MACRO
                      "political_instability_MIN_METADATA": 0.83,  # F1
                      "SEMI_1040_sylva_prior_MIN_METADATA": 0.932,  # F1
                      "SEMI_1044_eye_movements_MIN_METADATA": 0.6,  # F1_MACRO
                      "SEMI_1053_jm1_MIN_METADATA": 0.39,  # F1
                      "SEMI_1217_click_prediction_small_MIN_METADATA": 0.15,  # F1
                      "SEMI_1459_artificial_characters_MIN_METADATA": 0.63,  # F1_MACRO
                      "SEMI_155_pokerhand_MIN_METADATA": 0,  # F1_MACRO #todo what score
                      "uu_101_object_categories_MIN_METADATA": 0,  # Accuracy #todo what score
                      "uu10_posts_3_MIN_METADATA": 0.66,  # F1_MACRO
                      "uu1_datasmash_MIN_METADATA": 0,  # F1_MACRO #Todo what score
                      "uu2_gp_hyperparameter_estimation_MIN_METADATA": 99,  # MSE # todo what score
                      "uu3_world_development_indicators_MIN_METADATA": 700000000000,  # RMSE
                      "uu4_SPECT_MIN_METADATA": 0.89,  # F1
                      "uu5_heartstatlog_MIN_METADATA": 0.65,  # F1
                      "uu6_hepatitis_MIN_METADATA": 0.56,  # F1
                      "uu7_pima_diabetes_MIN_METADATA": 0.41,  # F1
                      "uu8_posts_1_MIN_METADATA": 55,  # RMSE
                      "uu9_posts_2_MIN_METADATA": 6,  # RMSE
                      }


def _run_seed_dataset(problem):
    if problem in ["LL1_3476_HMDB_actio_recognition_MIN_METADATA", "LL1_VID_UCF11_MIN_METADATA", ]:
        raise Exception("Problem type not supported")

    command = "python main.py"
    with open('d3m_test_server.txt', 'w') as f_server:
        env["CUDA_VISIBLE_DEVICES"] = '2'  # no cuda for test, OOM is a pain with multiprocessing.
        server_process = subprocess.Popen(command.split(' '), env=env, stderr=f_server, stdout=f_server,
                                          preexec_fn=os.setsid)
    time.sleep(10)

    with open('d3m_test_dummy.txt', 'w') as f:
        env["CUDA_VISIBLE_DEVICES"] = '2'  # no cuda for test, OOM is a pain with multiprocessing.
        command = f"python -m dummy_ta3.dummy_ta3 -p ./seed_datasets_current/{problem}/TRAIN/problem_TRAIN/problemDoc.json -d ./seed_datasets_current -e 0.0.0.0 -t 45042"
        process = subprocess.Popen(command.split(' '), env=env, stderr=f, stdout=f, preexec_fn=os.setsid)

        start = time.time()
        while time.time() - start <= 60 * 60:
            poll = process.poll()
            if poll != None:
                break
            time.sleep(1)
        else:
            process.terminate()
            process.communicate()
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)  # Really make sure everything is closed
            except Exception:
                pass
            raise TimeoutError("timeout on pipeline generation")

    # shutdown server
    os.killpg(os.getpgid(server_process.pid), signal.SIGTERM)

    pipeline_ids = []
    with open("pipeline_id.txt", "r") as f:
        for line in f:
            pipeline_ids.append(line.strip())

    for pipeline_id in pipeline_ids:
        run_pipeline_command = ("python -m d3m runtime "
                                f"--volumes {D3MSTATICDIR} "
                                f"-d {problem} "
                                f"--context TESTING --random-seed 0 "
                                f"fit-score "
                                f"--scores {D3MOUTPUTDIR}/score/{pipeline_id}.csv "
                                f"-p {D3MOUTPUTDIR}/pipelines_ranked/{pipeline_id}.json "
                                f"-r {D3MINPUTDIR}/{problem}/{problem}_problem/problemDoc.json "
                                f"-i {D3MINPUTDIR}/{problem}/TRAIN/dataset_TRAIN/datasetDoc.json "
                                f"-t {D3MINPUTDIR}/{problem}/TEST/dataset_TEST/datasetDoc.json "
                                f"-a {D3MINPUTDIR}/{problem}/SCORE/dataset_SCORE/datasetDoc.json")

        with open('d3m_test_fit.txt', 'w') as f:
            process = subprocess.Popen(run_pipeline_command.split(' '), env=env, stderr=f, stdout=f,
                                       preexec_fn=os.setsid)
            env["CUDA_VISIBLE_DEVICES"] = '2'  #

            start = time.time()
            while time.time() - start <= 60 * 60 * 2:
                poll = process.poll()
                if poll != None:
                    break
                time.sleep(1)
            else:
                process.terminate()
                process.communicate()
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                raise TimeoutError("timeout on fit")

    best_metric = None
    for pipeline_id in pipeline_ids:
        try:
            score = pd.read_csv(f"{D3MOUTPUTDIR}/score/{pipeline_id}.csv", "r")
        except pd.errors.EmptyDataError:
            # raise Exception(f"No score was generated for pipeline {pipeline_id}")
            continue  # only one pipeline needs to work
        metric = score['met'][0].split(',')
        if best_metric is None:
            best_metric = metric[1]
        print(f"\n {metric[0]}: {metric[1]}")
        if lower_is_better[metric[0]]:
            if metric[1] < best_metric:
                best_metric = metric[1]
        else:
            if metric[1] > best_metric:
                best_metric = metric[1]
    if best_metric is None:
        raise Exception("No valid pipeline was fitted")
    if problem in problem_thresholds:
        print(f"{best_metric} <> {problem_thresholds[problem]}")
        if lower_is_better[metric[0]]:
            assert float(best_metric) <= problem_thresholds[problem]
        else:
            assert float(best_metric) >= problem_thresholds[problem]
    else:
        # there are some new datasets that we don't know the threshold of yet.
        assert best_metric.is_digit()


def test_fn():
    problems = os.listdir(SEEDATASETS)
    for problem in problems:
    # for problem in ["185_baseball_MIN_METADATA"]:
        yield _run_seed_dataset, problem

    # server_process.terminate()
    # server_process.communicate()
