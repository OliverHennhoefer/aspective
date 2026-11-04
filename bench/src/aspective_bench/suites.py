"""Frozen benchmark suites.

A suite is a fixed list of datasets, so results stay comparable over time.
The lists were chosen once by a rule and are now edited only on purpose:

* PMLB classification and regression: real-world data only (no Friedman,
  Feynman, GAMETES, BNG or other simulated or generated sets, no
  ``_deprecated`` duplicates), one copy of each underlying dataset (one
  set from the Garvan thyroid family, no further views of the mfeat digits,
  one target per table), no missing values, 100 to 60k
  rows, at most 150 features. Classification additionally needs at least 10
  rows in the smallest class after subsampling, so stratified 5-fold CV with
  an inner 3-fold search sees every class, and drops trivial datasets that a
  default CART tree classifies with at least 99.5% 5-fold accuracy (as
  OpenML-CC18 does), since every model ties on them. PMLB holds 31 regression
  datasets that meet the rule.
* OpenML-CTR23 regression, the same rule applied to its 35 datasets. Real
  data rules out the six produced by a simulator (naval_propulsion_plant,
  kin8nm, pumadyn32nh, grid_stability, auction_verification,
  energy_efficiency); one copy of each dataset rules out the four PMLB
  already has (california_housing is 537_houses, cpu_activity is
  197_cpu_act, geographical_origin_of_music is 4544, solar_flare); no
  missing values rules out fps_benchmark and Moneyball; at most 60k rows
  rules out video_transcoding and wave_energy. That leaves 21, which join
  the 31 PMLB datasets in ``pmlb-ctr23-regression``.
* ADBench Classical: every dataset with at least 6 anomalies after
  subsampling to 5000 rows (the smallest count among the original 37). This
  leaves out only smtp, which keeps about 2.

The rule was fixed before any benchmarked model was run on the added
datasets, so the suites were not chosen to favour any estimator.

Large datasets are subsampled to ``max_samples`` rows once, with a fixed
seed and stratified by class, so every model sees the same data.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Suite:
    name: str
    source: str
    task: str
    datasets: tuple
    max_samples: int = 5000
    description: str = ""


PMLB_BINARY = (
    "analcatdata_lawsuit", "biomed", "breast_cancer", "breast_cancer_wisconsin_diagnostic", "bupa",
    "churn", "congressional_voting_records", "credit_approval_germany", "glass2", "haberman",
    "hepatitis", "hypothyroid", "ionosphere", "kr_vs_kp", "profb", "saheart", "sonar",
    "spambase", "spect", "spectf", "tic_tac_toe", "tokyo1",
    # added in the second suite version
    "adult", "analcatdata_boxing2", "analcatdata_creditscore", "appendicitis", "backache", "coil2000",
    "molecular_biology_promoters", "phoneme", "prnn_crabs",
)

PMLB_MULTICLASS = (
    "balance_scale", "car_evaluation", "cars", "contraceptive_method", "dermatology", "ecoli",
    "iris", "led7", "mfeat_morphological", "mfeat_zernike", "new_thyroid", "segmentation",
    "soybean", "tae", "vehicle", "vowel", "waveform_21", "wine_quality_red", "wine_recognition",
    "yeast",
    # added in the second suite version
    "analcatdata_authorship", "analcatdata_dmft", "analcatdata_germangss", "calendarDOW", "cloud",
    "collins", "flags", "letter", "movement_libras", "nursery", "optdigits", "page_blocks",
    "pendigits", "satimage", "schizo", "splice", "texture",
)

PMLB_REGRESSION = (
    "1027_ESL", "1028_SWD", "1029_LEV", "1030_ERA", "197_cpu_act", "201_pol", "210_cloud",
    "218_house_8L", "225_puma8NH", "230_machine_cpu", "294_satellite_image",
    "4544_GeographicalOriginalofMusic", "503_wind", "505_tecator", "522_pm10", "529_pollen",
    "537_houses", "547_no2", "560_bodyfat", "574_house_16H", "665_sleuth_case2002",
    "666_rmftsa_ladata", "690_visualizing_galaxy", "695_chatfield_4", "solar_flare",
    # added in the second suite version
    "519_vinnie", "523_analcatdata_neavote", "556_analcatdata_apnea2", "663_rabe_266",
    "678_visualizing_environmental", "712_chscase_geyser1",
)

# OpenML-CTR23 (study 353) datasets that meet the rule; see datasets.OPENML.
CTR23_REGRESSION = (
    "openml:44956_abalone", "openml:44957_airfoil_self_noise", "openml:44959_concrete_compressive_strength",
    "openml:44962_forest_fires", "openml:44963_physiochemical_protein", "openml:44964_superconductivity",
    "openml:44967_student_performance_por", "openml:44970_QSAR_fish_toxicity", "openml:44971_white_wine",
    "openml:44972_red_wine", "openml:44976_sarcos", "openml:44979_diamonds", "openml:44983_miami_housing",
    "openml:44984_cps88wages", "openml:44987_socmob", "openml:44989_kings_county",
    "openml:44990_brazilian_houses", "openml:44993_health_insurance", "openml:44994_cars",
    "openml:45012_fifa", "openml:45402_space_ga",
)

ADBENCH_CLASSICAL = (
    "1_ALOI", "2_annthyroid", "4_breastw", "6_cardio", "7_Cardiotocography", "12_fault",
    "14_glass", "15_Hepatitis", "17_InternetAds", "18_Ionosphere", "19_landsat", "20_letter",
    "21_Lymphography", "22_magic.gamma", "23_mammography", "24_mnist", "25_musk", "26_optdigits",
    "27_PageBlocks", "28_pendigits", "29_Pima", "30_satellite", "31_satimage-2", "32_shuttle",
    "35_SpamBase", "36_speech", "37_Stamps", "38_thyroid", "39_vertebral", "40_vowels",
    "41_Waveform", "42_WBC", "43_WDBC", "44_Wilt", "45_wine", "46_WPBC", "47_yeast",
    # added in the second suite version, subsampled to 5000 rows like the others
    "3_backdoor", "5_campaign", "8_celeba", "9_census", "10_cover", "11_donors", "13_fraud",
    "16_http", "33_skin",
)

SUITES = {
    s.name: s
    for s in [
        Suite("pmlb-classification", "pmlb", "classification", PMLB_BINARY + PMLB_MULTICLASS,
              description=f"{len(PMLB_BINARY + PMLB_MULTICLASS)} real-world PMLB classification datasets "
                          f"({len(PMLB_BINARY)} binary, {len(PMLB_MULTICLASS)} multiclass)"),
        Suite("pmlb-binary", "pmlb", "classification", PMLB_BINARY,
              description=f"the {len(PMLB_BINARY)} binary datasets of pmlb-classification"),
        Suite("pmlb-regression", "pmlb", "regression", PMLB_REGRESSION,
              description=f"{len(PMLB_REGRESSION)} real-world PMLB regression datasets"),
        Suite("pmlb-ctr23-regression", "pmlb", "regression", PMLB_REGRESSION + CTR23_REGRESSION,
              description=f"{len(PMLB_REGRESSION + CTR23_REGRESSION)} real-world regression datasets "
                          f"({len(PMLB_REGRESSION)} from PMLB, {len(CTR23_REGRESSION)} from OpenML-CTR23)"),
        Suite("adbench-classical", "adbench", "anomaly", ADBENCH_CLASSICAL,
              description=f"{len(ADBENCH_CLASSICAL)} ADBench Classical anomaly detection datasets (ODDS and others)"),
    ]
}


def get_suite(name):
    try:
        return SUITES[name]
    except KeyError:
        raise ValueError(f"unknown suite {name!r}; choose from {sorted(SUITES)}") from None
