from enum import Enum


class LandmarkModelEnum(Enum):
    Holistic = "Holistic"
    HandAndPose = "HandAndPose"
    PoseOnly = "PoseOnly"
    HandOnly = "HandOnly"


### Landmarks ###
LandModelType = LandmarkModelEnum.HandAndPose
RandomFlipImage = True
ValidHandsNeeded = 1

### Images ###
Labels = ["Hase", "Schlange", "Ziege", "Affe", "Elefant", "Baer", "Waschbaer",
          "JaBeidhaendig", "NeinBeidhaendig", "JaEinhaendig", "NeinEinhaendig",
          "Klein", "Mittel", "Gross",
          "StehtRumZurKamera", "StehtRumSeite", "StehtRumHinten",
          "Y", "M", "C", "A",
          "HaendeVorsGesicht", "FlaschrummerWaschbaer", "HaendeVorDieBrustNebeneinander", "HaendeVorDieBrustKreuz",
          "EinhaendigerHase", "EinhaendigeSchlange", "EinhaendigeZiege", "EinhaendigerAffe",
          "EinhaendigeElefantRuessel", "EinhaendigerElefantNase", "EinhaendigerBaer", "EinhaendigerWascbaer"]

### Training ###
InFeaturesHolistic = 1629  # don't change that
InFeaturesHandPose = 225  # don't change that
InFeaturesHand = 126  # don't change that
InFeaturesPose = 99  # don't change that
Hiddenlayer = 450 # muss durch 2 Teilbar sein
LearningRate = 0.00025
batch_size = 16
num_epochs = 40

### Hoerspiel ###
GestureThreshold = 1
NotGestures = ["StehtRumZurKamera", "StehtRumSeite", "StehtRumHinten",
               "Y", "M", "C", "A",
               "HaendeVorsGesicht", "FlaschrummerWaschbaer", "HaendeVorDieBrust",
               "EinhaendigerHase", "EinhaendigeSchlange", "EinhaendigeZiege", "EinhaendigerAffe",
               "EinhaendigeElefantRuessel", "EinhaendigerElefantNase", "EinhaendigerBaer", "EinhaendigerWascbaer"]


def GetLabelMap():
    label_map = {label: index for index, label in enumerate(Labels)}
    return label_map


def GetInversLableMap():
    label_map = GetLabelMap()
    inv_label_map = {v: k for k, v in label_map.items()}
    return inv_label_map


def GetInFetures():
    if LandModelType == LandModelType.Holistic:
        return InFeaturesHolistic
    else:
        InFeatures = 0
        if LandModelType == LandModelType.HandAndPose or LandModelType == LandModelType.HandOnly:
            InFeatures += InFeaturesHand
        if LandModelType == LandModelType.HandAndPose or LandModelType == LandModelType.PoseOnly:
            InFeatures += InFeaturesPose
        return InFeatures
