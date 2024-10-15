from os import path

DATASET_PATH = path.join(*['dataset','UCRArchive_2018'])

ALL_DATASET_NAME = ["Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF", "ChlorineConcentration",
                    "CinCECGTorso", "Coffee", "Computers", "CricketX", "CricketY", "CricketZ", "DiatomSizeReduction",
                    "DistalPhalanxOutlineCorrect", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxTW", "Earthquakes",
                    "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR",
                    "FiftyWords", "Fish", "FordA", "FordB", "GunPoint", "Ham", "HandOutlines", "Haptics", "Herring",
                    "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2",
                    "Lightning7", "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineCorrect",
                    "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFetalECGThorax1",
                    "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme", "Plane",
                    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxTW",
                    "RefrigerationDevices", "ScreenType", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances",
                    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf",
                    "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG",
                    "TwoPatterns", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ",
                    "UWaveGestureLibraryAll", "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga",
                    'ACSF1', 'BME', 'Chinatown', 'Crop',
                    'EOGHorizontalSignal', 'EOGVerticalSignal', 'EthanolLevel', 'FreezerRegularTrain', 'FreezerSmallTrain',
                    'Fungi', 'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'HouseTwenty',
                    'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'MelbournePedestrian', 'MixedShapesRegularTrain',
                    'MixedShapesSmallTrain', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'PowerCons', 'Rock',
                    'SemgHandGenderCh2', 'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'SmoothSubspace', 'UMD']

CLUSTER_DATASET_NAME = ['ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car','ChlorineConcentration', 'Coffee',
                        'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect','ECG200',
                        'ECGFiveDays','GunPoint', 'Ham', 'Herring','Lightning2','Meat', 'MiddlePhalanxOutlineAgeGroup',
                        'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain', 'OSULeaf', 'Plane',
                        'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW', 'SonyAIBORobotSurface1',
                        'SonyAIBORobotSurface2', 'SwedishLeaf', 'Symbols', 'ToeSegmentation1', 'ToeSegmentation2',
                        'TwoPatterns','TwoLeadECG','Wafer', 'Wine','WordSynonyms']

