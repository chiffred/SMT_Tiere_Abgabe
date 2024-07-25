import math
import sys
import time
from typing import List

import cv2 as cv
import mediapipe as mp
import torch
from HumanimalClassifier import HumanimalClassifier
import config
import numpy as np
from enum import Enum
from UtilFunctions import extract_landmarks_holistic, extract_landmarks_hand_pose, process_landmarks, \
    predict_landmark_class
import argparse
import mido

# Global Tick List
PreTickList = []
PostTickList = []

# Test Mode
TestMode = True
MP3Mode = False
if not TestMode:
    # MIDI Ports
    print(mido.get_output_names())
    print(mido.get_input_names())
    input_port = mido.open_input('DesktopMidi 0')
    output_port = mido.open_output('DesktopMidi 1')

percent_decay_rate = 1
percent_rise_rate = 4

TestText = ""


class MidiNotes(Enum):
    Intro_Intro = 0
    Intro_BirthdayChild = 1
    Intro_PartyGame = 8
    Intro_Music = 15
    Intro_Drinks = 22
    Intro_Location = 29
    Intro_LocationSize = 30
    Intro_Food = 43
    Intro_Outro = 50 ###

    BirthChild_Affe = 2
    BirthChild_Baer = 3
    BirthChild_Elefant = 4
    BirthChild_Hase = 5
    BirthChild_Schlange = 6
    BirthChild_Ziege = 7
    PartyGame_Affe = 9
    PartyGame_Baer = 10
    PartyGame_Elefant = 11
    PartyGame_Hase = 12
    PartyGame_Schlange = 13
    PartyGame_Ziege = 14
    Music_Affe = 16
    Music_Baer = 17
    Music_Elefant = 18
    Music_Hase = 19
    Music_Schlange = 20
    Music_Ziege = 21
    Drinks_Affe = 23
    Drinks_Baer = 24
    Drinks_Elefant = 25
    Drinks_Hase = 26
    Drinks_Schlange = 27
    Drinks_Ziege = 28
    Location_Affe = 37
    Location_Baer = 38
    Location_Elefant = 39
    Location_Hase = 40
    Location_Schlange = 41
    Location_Ziege = 42
    Location_Affe_E = 31
    Location_Baer_E = 32
    Location_Elefant_E = 33
    Location_Hase_E = 34
    Location_Schlange_E = 35
    Location_Ziege_E = 36
    Food_Affe = 44
    Food_Baer = 45
    Food_Elefant = 46
    Food_Hase = 47
    Food_Schlange = 48
    Food_Ziege = 49

    Outro_Location_Affe = 51
    Outro_Location_Baer = 52
    Outro_Location_Elefant = 53
    Outro_Location_Elefant_toSmall = 54
    Outro_Location_Hase = 55
    Outro_Location_Schlange = 56
    Outro_Location_Ziege = 57

    Outro_BirthChild_Affe = 58
    Outro_BirthChild_Baer = 59
    Outro_BirthChild_Elefant = 60
    Outro_BirthChild_Hase = 61
    Outro_BirthChild_Schlange = 62
    Outro_BirthChild_Ziege = 63

    Outro_Music_Affe = 64
    Outro_Music_Baer = 65
    Outro_Music_Elefant = 66
    Outro_Music_Hase = 67
    Outro_Music_Schlange = 68
    Outro_Music_Ziege = 69

    Outro_PartyGame_Affe = 70
    Outro_PartyGame_Baer = 71
    Outro_PartyGame_Elefant = 72
    Outro_PartyGame_Hase = 73
    Outro_PartyGame_Schlange = 74
    Outro_PartyGame_Ziege = 75

    Outro_Food_Affe = 76
    Outro_Food_Baer = 77
    Outro_Food_Elefant = 78
    Outro_Food_Hase = 79
    Outro_Food_Schlange = 80
    Outro_Food_Ziege = 81

    Outro_Drinks_Affe = 82
    Outro_Drinks_Baer = 83
    Outro_Drinks_Elefant = 84
    Outro_Drinks_Hase = 85
    Outro_Drinks_Schlange = 86
    Outro_Drinks_Ziege = 87

    Outro_Secret_Racoon = 88

    Outro_Background_Music_Affe = 89
    Outro_Background_Music_Baer = 90
    Outro_Background_Music_Elefant = 91
    Outro_Background_Music_Hase = 92
    Outro_Background_Music_Schlange = 93
    Outro_Background_Music_Ziege = 94
    Outro_Background_Music_YMCA = 95


class GestureType(Enum):
    Animal = "Animal"
    Start = "Start"
    Size = "Size"
    Other = "Other"


class GestureClass:
    def __init__(self, path: str, name: str, gesture_type: GestureType):
        self.Path = path
        self.Name = name
        self.Type = gesture_type
        self.Percent = 0
        self.Selected = False

    def tick(self):
        if self.Selected:
            return
        if self.Percent >= 100:
            self.Selected = True
        elif self.Percent > 0:
            self.Percent -= percent_decay_rate
        if self.Percent < 0:
            self.Percent = 0


class StoryClass:
    def __init__(self):
        self.finished = False
        self.Intro = False

        self.BirthdayChild = None
        self.Food = None
        self.PartyGame = None
        self.Music = None
        self.Drinks = None
        self.Location = None
        self.WaitingForLocation = False
        self.LocationSize = None
        self.BirthdayChild = None

        self.RacoonStoleThePresent = False
        self.YMCA_Y = False
        self.YMCA_M = False
        self.YMCA_C = False
        self.YMCA_A = False

    def tick(self):
        pass


class MidiPlayerSong:
    def __init__(self, play_duration: float = 1, midi_note: int = 0, path: str = None):
        self.MP3Mode = MP3Mode
        self.PlayDuration = play_duration
        self.MidiNote = midi_note
        self.Path = path

    def play_song(self):
        if self.MP3Mode:
            print("Not implemented yet")
        else:
            msg = mido.Message('note_on', note=self.MidiNote, velocity=64, time=1)
            output_port.send(msg)
            print(f'Played {self.MidiNote}')


class MidiReverbEffect:
    def __init__(self, value: float = 0):
        self.Value = value
        self.PlayDuration = 0

    def play_song(self):
        msg = mido.Message('control_change', channel=0, control=92, value=self.Value)
        output_port.send(msg)


class ArrayPlayer:
    def __init__(self, song_array: List[MidiPlayerSong] = None):
        self.SongArray = song_array if song_array is not None else []
        self.PlayIndex = 0
        self.IsPlaying = False
        self.PlayedComplete = False
        self.end_time = 0
        self.PauseBetweenSongs = 0
        self.PlayedComplete = None
        self.CurrentSong = None
        PostTickList.append(self)

    def add_song(self, song):
        self.SongArray.append(song)

    def set_pause(self, pause: float):
        self.PauseBetweenSongs = pause

    def play_all(self, pause_between_songs: float = 0.1):
        self.IsPlaying = True

    def get_total_length(self):
        if self.SongArray:
            length = 0
            for song in self.SongArray:
                length += song.PlayDuration
                length += self.PauseBetweenSongs
            return length
        return 0

    def tick(self):
        if not self.IsPlaying:
            return
        if self.end_time <= time.time():
            if not self.PlayIndex < len(self.SongArray):
                self.end_play()
            else:
                self.CurrentSong = self.SongArray[self.PlayIndex]
                self.CurrentSong.play_song()
                self.end_time = time.time() + self.CurrentSong.PlayDuration + self.PauseBetweenSongs
                self.PlayIndex += 1

    def end_play(self):
        PostTickList.remove(self)
        self.PlayedComplete = True
        self.IsPlaying = False
        self.CurrentSong = None
        self.SongArray = None


class AnimalParty:
    def __init__(self, webcam=0):
        self.TestMode = TestMode
        self.TestText = TestText
        self.IsRunning = True
        self.model = None
        self.previous_pred = None
        self.start_time = None
        self.wait_time = None

        self.inv_label_map = config.GetInversLableMap()
        self.OutClasses = len(config.Labels)

        self.mp_hands = None
        self.mp_pose = None
        self.mp_holistic = None
        self.Webcam = webcam
        self.vid = None

        self.gestures = self.init_gestures()
        self.story_progress = StoryClass()
        self.isWaitingForInput = False

        self.extra_counter = 0

    def trigger_function(self, gesture):
        if not self.isWaitingForInput:
            return
        gesture = self.get_gesture_from_string(gesture)
        if not gesture:
            return
        if gesture.Name == "EinhaendigeSchlange":
            gesture = self.get_gesture_from_string("Schlange")

        # It not Startet
        if not self.story_progress.Intro:
            if gesture.Type == GestureType.Start:
                gesture.Percent += percent_rise_rate
                if gesture.Selected:
                    self.story_progress.Intro = True
                    self.TriggerStart()
            return

        if gesture.Type == GestureType.Other:
            print("Other")
            gesture.Percent += percent_rise_rate
            if gesture.Percent >= 100 or gesture.Selected:
                print("Other+")
                match gesture.Name:
                    case "Waschbaer":
                        print("Other++")
                        self.story_progress.RacoonStoleThePresent = True
                    case "Y":
                        self.story_progress.YMCA_Y = True
                    case "M":
                        self.story_progress.YMCA_M = True
                    case "C":
                        self.story_progress.YMCA_C = True
                    case "A":
                        self.story_progress.YMCA_A = True

        # is selecting size
        if self.story_progress.WaitingForLocation:
            if gesture.Type == GestureType.Size:
                gesture.Percent += percent_rise_rate
                if gesture.Selected:
                    print("Groesse ist gewaehlt")
                    self.story_progress.WaitingForLocation = False
                    self.story_progress.LocationSize = gesture.Name
                    self.TriggerSize(gesture.Name)
            return

        # is selecting animal
        if gesture.Type == GestureType.Animal:
            gesture.Percent += percent_rise_rate
            if gesture.Percent >= 100 and not gesture.Selected:
                print("Tier ist gewaehlt")
                if self.story_progress.BirthdayChild is None:
                    self.story_progress.BirthdayChild = gesture.Name
                    self.TriggerBirthChild(gesture.Name)
                    return
                if self.story_progress.PartyGame is None:
                    self.story_progress.PartyGame = gesture.Name
                    self.TriggerPartyGame(gesture.Name)
                    return
                if self.story_progress.Music is None:
                    self.story_progress.Music = gesture.Name
                    self.TriggerMusic(gesture.Name)
                    return
                if self.story_progress.Drinks is None:
                    self.story_progress.Drinks = gesture.Name
                    self.TriggerDrinks(gesture.Name)
                    return
                if self.story_progress.Location is None:
                    self.story_progress.Location = gesture.Name
                    self.story_progress.WaitingForLocation = True
                    self.TriggerLocation(gesture.Name)
                    return
                if self.story_progress.Food is None:
                    self.story_progress.Food = gesture.Name
                    self.TriggerFood(gesture.Name)
                    return

    def start(self):
        self.begin_play()
        while self.IsRunning:
            self.system_tick()
        self.destruct()

    def system_tick(self):
        for TickObject in PreTickList:
            TickObject.tick()
        self.tick()
        for animal in self.gestures:
            animal.tick()
        for TickObject in PostTickList:
            TickObject.tick()


    def destruct(self):
        self.vid.release()
        cv.destroyAllWindows()
        if self.mp_holistic:
            self.mp_holistic.close()
        if self.mp_hands:
            self.mp_hands.close()
        if self.mp_pose:
            self.mp_pose.close()

    def begin_play(self):
        # Initialize a model
        self.model = HumanimalClassifier(config.GetInFetures(), hiddenlayer=config.Hiddenlayer,
                                         num_classes=self.OutClasses)

        # Load the weights from the saved model file
        if config.LandModelType == config.LandModelType.Holistic:
            ModelFilename = './Data/Model_classifier_holistic.pth'
            self.mp_holistic = mp.solutions.holistic.Holistic()
        else:
            ModelFilename = './Data/Model_classifier'
            if config.LandModelType == config.LandModelType.HandAndPose or config.LandModelType == config.LandModelType.HandOnly:
                self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.3)
                ModelFilename += '_hand'
            if config.LandModelType == config.LandModelType.HandAndPose or config.LandModelType == config.LandModelType.PoseOnly:
                self.mp_pose = mp.solutions.pose.Pose()
                ModelFilename += '_pose'
            ModelFilename += '.pth'

        try:
            self.model.load_state_dict(torch.load(ModelFilename))
        except RuntimeError:
            print("No saved model found")
            sys.exit()

        # Put the model in evaluation mode
        self.model.eval()

        # Start Video Capture
        self.vid = cv.VideoCapture(self.Webcam)
        cv.namedWindow('AnimalParty', cv.WINDOW_NORMAL)
        self.vid.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.vid.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

        if not TestMode:
            # Deactivate Reverb
            self.send_cc_message(92)

        self.isWaitingForInput = True

    def draw_pip(self, main_image, image_address, position, percent=0):
        distance_horizontal = 10
        distance_vertical = 10
        pip_img = cv.imread(image_address)
        pip_img = cv.resize(pip_img, (100, 100))
        start_row = position * (100 + distance_vertical) + distance_vertical
        end_row = start_row + pip_img.shape[0]
        end_col = main_image.shape[1] - distance_horizontal
        start_col = end_col - pip_img.shape[1]
        main_image[start_row:end_row, start_col:end_col] = pip_img

        color = (0, 255, 0)
        section = percent * pip_img.shape[1] // 100
        loading_bar = np.zeros_like(pip_img)
        loading_bar[:, :section] = color
        trans_bar = cv.addWeighted(pip_img, 0.5, loading_bar, 0.5, 0)
        main_image[start_row:start_row + trans_bar.shape[0], start_col:start_col + trans_bar.shape[1]] = trans_bar
        return main_image

    def draw_and_check_landmarks_on_image(self, frame):
        if config.LandModelType == config.LandModelType.Holistic:
            landmarks, frame = extract_landmarks_holistic(frame, self.mp_holistic, config.ValidHandsNeeded)
        else:
            landmarks, frame = extract_landmarks_hand_pose(frame, self.mp_hands, self.mp_pose, config.ValidHandsNeeded,
                                                           shouldDrawOnImage=True)
        if landmarks is None:
            pass
            # self.write_on_image(frame, "Keine Geste erkannt")
            # self.start_time = datetime.datetime.now()
            # self.previous_pred = None
        else:
            landmarks = process_landmarks(landmarks)
            prediction = predict_landmark_class(landmarks, self.model)
            # self.write_on_image(frame, "Ohh: " + str(self.inv_label_map[prediction]))
            self.trigger_function(str(self.inv_label_map[prediction]))
        return frame

    def write_on_image(self, frame, text, line=1):
        cv.putText(frame, text, (50, 50 * line), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    def get_gesture_from_string(self, gesture_name):
        result = list(filter(lambda gesture: gesture.Name == gesture_name, self.gestures))
        if result:
            return result[0]
        return None

    def activate_in_sec(self, seconds):
        self.isWaitingForInput = False
        self.start_time = time.time()
        self.wait_time = seconds

    def on_sound_is_finished(self):
        self.isWaitingForInput = True
        if self.story_progress.Food is not None and self.story_progress.finished is False:
            self.TriggerEndScene()
            self.story_progress.finished = True
        else:
            if self.story_progress.finished:
                self.restart_game()

    def init_gestures(self):
        gestures = [GestureClass('./Icons/Affe.png', 'Affe', GestureType.Animal),
                    GestureClass('./Icons/baer.png', 'Baer', GestureType.Animal),
                    GestureClass('./Icons/elefant.png', 'Elefant', GestureType.Animal),
                    GestureClass('./Icons/Hase.png', 'Hase', GestureType.Animal),
                    GestureClass('./Icons/schlange.png', 'Schlange', GestureType.Animal),
                    GestureClass('./Icons/Ziege.png', 'Ziege', GestureType.Animal),
                    GestureClass('./Icons/Affe.png', 'Waschbaer', GestureType.Other),
                    GestureClass('./Icons/Affe.png', 'EinhaendigeSchlange', GestureType.Other),
                    GestureClass('./Icons/Affe.png', 'Y', GestureType.Other),
                    GestureClass('./Icons/Affe.png', 'M', GestureType.Other),
                    GestureClass('./Icons/Affe.png', 'C', GestureType.Other),
                    GestureClass('./Icons/Affe.png', 'A', GestureType.Other),
                    GestureClass('./assets/start.jpg', 'JaBeidhaendig', GestureType.Start),
                    GestureClass('./assets/klein.jpg', 'Klein', GestureType.Size),
                    GestureClass('./assets/mittel.jpg', 'Mittel', GestureType.Size),
                    GestureClass('./assets/gross.jpg', 'Gross', GestureType.Size)]
        return gestures

    def init_midi_songs(self):
        pass

    def restart_game(self):
        self.gestures = self.init_gestures()
        self.story_progress = StoryClass()
        self.isWaitingForInput = True
        self.start_time = None
        self.wait_time = None
        self.send_cc_message(92)
        self.StopAll()
        print("Game Restartet")

    def tick(self):
        ret, frame = self.vid.read()

        # Check if waiting for Input or waiting for sample to finish
        if self.start_time is not None and time.time() - self.start_time >= self.wait_time:
            self.start_time = None
            self.on_sound_is_finished()

        if self.isWaitingForInput:
            frame = self.draw_and_check_landmarks_on_image(frame)
            if self.story_progress.RacoonStoleThePresent:
                self.write_on_image(frame, "MC Waschbaer in da House", 1)
            YMCAString = ""
            if self.story_progress.YMCA_Y:
                YMCAString = "Y"
            if self.story_progress.YMCA_M:
                YMCAString += "M"
            if self.story_progress.YMCA_C:
                YMCAString += "C"
            if self.story_progress.YMCA_A:
                YMCAString += "A"
            if YMCAString is not "":
                self.write_on_image(frame, YMCAString, 2)

        # Check if Game has started
        if not self.story_progress.Intro:
            # Draw Images on Side
            position = 0
            for gesture in self.gestures:
                if gesture.Type == GestureType.Start:
                    frame = self.draw_pip(frame, gesture.Path, position, math.floor(gesture.Percent))
                    position += 1
        elif self.story_progress.WaitingForLocation:
            # Draw Images on Side
            position = 0
            for gesture in self.gestures:
                if gesture.Type == GestureType.Size:
                    frame = self.draw_pip(frame, gesture.Path, position, math.floor(gesture.Percent))
                    position += 1
        else:
            # Draw Images on Side
            position = 0
            for gesture in self.gestures:
                if gesture.Type == GestureType.Animal and not gesture.Selected:
                    frame = self.draw_pip(frame, gesture.Path, position, math.floor(gesture.Percent))
                    position += 1

        cv.imshow('AnimalParty', frame)

        if cv.waitKey(1) == ord('q'):
            self.IsRunning = False
        if cv.waitKey(1) == ord('r'):
            self.restart_game()

    def PlayMidiNote(self, note, velocity=64, playtime=1, msg_print=True):
        msg = mido.Message('note_on', note=note, velocity=velocity, time=playtime)
        if msg_print:
            print(msg)
        output_port.send(msg)

    # Sende eine CC-Nachricht, um den Reverb-Wert zu ändern
    def send_cc_message(self, controller, value=0, channel=0):
        msg = mido.Message('control_change', channel=channel, control=controller, value=value)
        output_port.send(msg)
        print(msg)

    #######################################################
    ############### Alles hier drunter!!! #################
    #######################################################
    def TriggerStart(self):
        IntroIntroPlayer = ArrayPlayer()

        if TestMode:
            self.TestText = "Start Text"
            print("Trigger Start")
        else:
            #IntroIntroPlayer.add_song(MidiPlayerSong(72, MidiNotes.Intro_Intro.value))
            IntroIntroPlayer.add_song(MidiPlayerSong(45, MidiNotes.Intro_Intro.value))
            IntroIntroPlayer.add_song(MidiPlayerSong(6, MidiNotes.Intro_BirthdayChild.value))

        self.activate_in_sec(IntroIntroPlayer.get_total_length())
        IntroIntroPlayer.play_all()

    def StopAll(self):
        self.activate_in_sec(3)
        if TestMode:
            self.TestText = ""
            print("Trigger Stop")
        else:
            pass


    def TriggerBirthChild(self, Name):
        BirthChildPlayer = ArrayPlayer()
        BirthChildPlayer.set_pause(0.5)
        if TestMode:
            self.TestText = "Birthday Child Audio"
            print("Trigger BirthdayChild")
        else:
            match Name:
                case "Affe":
                    BirthChildPlayer.add_song(MidiPlayerSong(16, MidiNotes.BirthChild_Affe.value))
                case "Baer":
                    BirthChildPlayer.add_song(MidiPlayerSong(18, MidiNotes.BirthChild_Baer.value))
                case "Elefant":
                    BirthChildPlayer.add_song(MidiPlayerSong(15, MidiNotes.BirthChild_Elefant.value))
                case "Hase":
                    BirthChildPlayer.add_song(MidiPlayerSong(16, MidiNotes.BirthChild_Hase.value))
                case "Schlange":
                    BirthChildPlayer.add_song(MidiPlayerSong(16, MidiNotes.BirthChild_Schlange.value))
                case "Ziege":
                    BirthChildPlayer.add_song(MidiPlayerSong(19, MidiNotes.BirthChild_Ziege.value))
            BirthChildPlayer.add_song(MidiPlayerSong(11, MidiNotes.Intro_PartyGame.value))

        self.activate_in_sec(BirthChildPlayer.get_total_length())
        BirthChildPlayer.play_all()

    def TriggerPartyGame(self, Name):
        PartyGamePlayer = ArrayPlayer()
        PartyGamePlayer.set_pause(0.5)
        if TestMode:
            self.TestText = "Party Game Audio"
            print("Trigger Party Game")
        else:
            match Name:
                case "Affe":
                    PartyGamePlayer.add_song(MidiPlayerSong(12, MidiNotes.PartyGame_Affe.value))
                case "Baer":
                    PartyGamePlayer.add_song(MidiPlayerSong(24, MidiNotes.PartyGame_Baer.value))
                case "Elefant":
                    PartyGamePlayer.add_song(MidiPlayerSong(13, MidiNotes.PartyGame_Elefant.value))
                case "Hase":
                    PartyGamePlayer.add_song(MidiPlayerSong(13, MidiNotes.PartyGame_Hase.value))
                case "Schlange":
                    PartyGamePlayer.add_song(MidiPlayerSong(16, MidiNotes.PartyGame_Schlange.value))
                case "Ziege":
                    PartyGamePlayer.add_song(MidiPlayerSong(24, MidiNotes.PartyGame_Ziege.value))
            PartyGamePlayer.add_song(MidiPlayerSong(8, MidiNotes.Intro_Music.value))
        self.activate_in_sec(PartyGamePlayer.get_total_length())
        PartyGamePlayer.play_all()

    def TriggerMusic(self, Name):
        MusicPlayer = ArrayPlayer()
        if TestMode:
            self.TestText = "Music Audio"
            print("Trigger Music")
        else:
            match Name:
                case "Affe":
                    MusicPlayer.add_song(MidiPlayerSong(22, MidiNotes.Music_Affe.value))
                case "Baer":
                    MusicPlayer.add_song(MidiPlayerSong(22, MidiNotes.Music_Baer.value))
                case "Elefant":
                    MusicPlayer.add_song(MidiPlayerSong(22, MidiNotes.Music_Elefant.value))
                case "Hase":
                    MusicPlayer.add_song(MidiPlayerSong(22, MidiNotes.Music_Hase.value))
                case "Schlange":
                    MusicPlayer.add_song(MidiPlayerSong(22, MidiNotes.Music_Schlange.value))
                case "Ziege":
                    MusicPlayer.add_song(MidiPlayerSong(22, MidiNotes.Music_Ziege.value))
            MusicPlayer.add_song(MidiPlayerSong(8, MidiNotes.Intro_Drinks.value))
        self.activate_in_sec(MusicPlayer.get_total_length())
        MusicPlayer.play_all()

    def TriggerDrinks(self, Name):
        DrinkPlayer = ArrayPlayer()
        if TestMode:
            self.TestText = "Drink Audio"
            print("Trigger Drinks")
        else:
            match Name:
                case "Affe":
                    DrinkPlayer.add_song(MidiPlayerSong(13, MidiNotes.Drinks_Affe.value))
                case "Baer":
                    DrinkPlayer.add_song(MidiPlayerSong(19, MidiNotes.Drinks_Baer.value))
                case "Elefant":
                    DrinkPlayer.add_song(MidiPlayerSong(15, MidiNotes.Drinks_Elefant.value))
                case "Hase":
                    DrinkPlayer.add_song(MidiPlayerSong(15, MidiNotes.Drinks_Hase.value))
                case "Schlange":
                    DrinkPlayer.add_song(MidiPlayerSong(15, MidiNotes.Drinks_Schlange.value))
                case "Ziege":
                    DrinkPlayer.add_song(MidiPlayerSong(18, MidiNotes.Drinks_Ziege.value))
            DrinkPlayer.add_song(MidiPlayerSong(8, MidiNotes.Intro_Location.value))
        self.activate_in_sec(DrinkPlayer.get_total_length())
        DrinkPlayer.play_all()

    def TriggerLocation(self, Name):
        LocationPlayer = ArrayPlayer()
        if TestMode:
            self.TestText = "Location Audio"
            print("Trigger Location")
        else:
            LocationPlayer.add_song(MidiPlayerSong(9, MidiNotes.Intro_LocationSize.value))
        self.activate_in_sec(LocationPlayer.get_total_length())
        LocationPlayer.play_all()

    def TriggerSize(self, Name):
        LocationSizePlayer = ArrayPlayer()
        if TestMode:
            self.TestText = "Size Audiot"
            print("Trigger Size")
        else:
            reverb_code = None
            match Name:
                case "Klein":
                    reverb_code = MidiReverbEffect(0)
                case "Mittel":
                    reverb_code = MidiReverbEffect(30)
                case "Gross":
                    reverb_code = MidiReverbEffect(90)

            match self.story_progress.Location:
                case "Affe":
                    LocationSizePlayer.add_song(MidiPlayerSong(7, MidiNotes.Location_Affe_E.value))
                    LocationSizePlayer.add_song(reverb_code)
                    LocationSizePlayer.add_song(MidiPlayerSong(7, MidiNotes.Location_Affe.value))
                case "Baer":
                    LocationSizePlayer.add_song(MidiPlayerSong(7, MidiNotes.Location_Baer_E.value))
                    LocationSizePlayer.add_song(reverb_code)
                    LocationSizePlayer.add_song(MidiPlayerSong(7, MidiNotes.Location_Baer.value))
                case "Elefant":
                    LocationSizePlayer.add_song(MidiPlayerSong(7, MidiNotes.Location_Elefant_E.value))
                    LocationSizePlayer.add_song(reverb_code)
                    LocationSizePlayer.add_song(MidiPlayerSong(7, MidiNotes.Location_Elefant.value))
                case "Hase":
                    LocationSizePlayer.add_song(MidiPlayerSong(7, MidiNotes.Location_Hase_E.value))
                    LocationSizePlayer.add_song(reverb_code)
                    LocationSizePlayer.add_song(MidiPlayerSong(7, MidiNotes.Location_Hase.value))
                case "Schlange":
                    LocationSizePlayer.add_song(MidiPlayerSong(7, MidiNotes.Location_Schlange_E.value))
                    LocationSizePlayer.add_song(reverb_code)
                    LocationSizePlayer.add_song(MidiPlayerSong(9, MidiNotes.Location_Schlange.value))
                case "Ziege":
                    LocationSizePlayer.add_song(MidiPlayerSong(7, MidiNotes.Location_Ziege_E.value))
                    LocationSizePlayer.add_song(reverb_code)
                    LocationSizePlayer.add_song(MidiPlayerSong(10, MidiNotes.Location_Ziege.value))
            LocationSizePlayer.add_song(MidiReverbEffect(0))
            LocationSizePlayer.add_song(MidiPlayerSong(12, MidiNotes.Intro_Food.value))
        self.activate_in_sec(LocationSizePlayer.get_total_length())
        LocationSizePlayer.play_all()
    def TriggerFood(self, Name):
        FoodPlayer = ArrayPlayer()
        if TestMode:
            print("Trigger Food")
        else:
            match Name:
                case "Affe":
                    FoodPlayer.add_song(MidiPlayerSong(15, MidiNotes.Food_Affe.value))
                case "Baer":
                    FoodPlayer.add_song(MidiPlayerSong(21, MidiNotes.Food_Baer.value))
                case "Elefant":
                    FoodPlayer.add_song(MidiPlayerSong(16, MidiNotes.Food_Elefant.value))
                case "Hase":
                    FoodPlayer.add_song(MidiPlayerSong(13, MidiNotes.Food_Hase.value))
                case "Schlange":
                    FoodPlayer.add_song(MidiPlayerSong(15, MidiNotes.Food_Schlange.value))
                case "Ziege":
                    FoodPlayer.add_song(MidiPlayerSong(15, MidiNotes.Food_Ziege.value))
            FoodPlayer.add_song(MidiPlayerSong(18, MidiNotes.Intro_Outro.value)) ##############
        self.activate_in_sec(FoodPlayer.get_total_length())
        FoodPlayer.play_all()
    def TriggerEndScene(self):
        print('Trigger End Scene')
        PartyScenePlayer = ArrayPlayer()
        PartyScenePlayer.set_pause(1)  # Pause zwischen den einzelnen Teilen
        PartyMusic = ArrayPlayer()

        match self.story_progress.Location:
            case "Affe":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Location_Affe.value))
            case "Baer":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Location_Baer.value))
            case "Elefant":
                if self.story_progress.LocationSize == "Klein":
                    PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Location_Elefant_toSmall.value))
                else:
                    PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Location_Elefant.value))
            case "Hase":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Location_Hase.value))
            case "Schlange":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Location_Schlange.value))
            case "Ziege":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Location_Ziege.value))
        match self.story_progress.BirthdayChild:
            case "Affe":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_BirthChild_Affe.value))
            case "Baer":
                PartyScenePlayer.add_song(MidiPlayerSong(9, MidiNotes.Outro_BirthChild_Baer.value))
            case "Elefant":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_BirthChild_Elefant.value))
            case "Hase":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_BirthChild_Hase.value))
            case "Schlange":
                PartyScenePlayer.add_song(MidiPlayerSong(4, MidiNotes.Outro_BirthChild_Schlange.value))
            case "Ziege":
                PartyScenePlayer.add_song(MidiPlayerSong(4, MidiNotes.Outro_BirthChild_Ziege.value))
        match self.story_progress.Music:
            case "Affe":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Music_Affe.value))
                PartyMusic.add_song(MidiPlayerSong(6, MidiNotes.Outro_Background_Music_Affe.value))
            case "Baer":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Music_Baer.value))
                PartyMusic.add_song(MidiPlayerSong(6, MidiNotes.Outro_Background_Music_Baer.value))
            case "Elefant":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Music_Elefant.value))
                PartyMusic.add_song(MidiPlayerSong(6, MidiNotes.Outro_Background_Music_Elefant.value))
            case "Hase":
                PartyScenePlayer.add_song(MidiPlayerSong(7, MidiNotes.Outro_Music_Hase.value))
                PartyMusic.add_song(MidiPlayerSong(6, MidiNotes.Outro_Background_Music_Hase.value))
            case "Schlange":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Music_Schlange.value))
                PartyMusic.add_song(MidiPlayerSong(6, MidiNotes.Outro_Background_Music_Schlange.value))
            case "Ziege":
                PartyScenePlayer.add_song(MidiPlayerSong(7, MidiNotes.Outro_Music_Ziege.value))
                PartyMusic.add_song(MidiPlayerSong(6, MidiNotes.Outro_Background_Music_Ziege.value))
        match self.story_progress.PartyGame:
            case "Affe":
                PartyScenePlayer.add_song(MidiPlayerSong(9, MidiNotes.Outro_PartyGame_Affe.value))
            case "Baer":
                PartyScenePlayer.add_song(MidiPlayerSong(7, MidiNotes.Outro_PartyGame_Baer.value))
            case "Elefant":
                PartyScenePlayer.add_song(MidiPlayerSong(7, MidiNotes.Outro_PartyGame_Elefant.value))
            case "Hase":
                PartyScenePlayer.add_song(MidiPlayerSong(7, MidiNotes.Outro_PartyGame_Hase.value))
            case "Schlange":
                PartyScenePlayer.add_song(MidiPlayerSong(9, MidiNotes.Outro_PartyGame_Schlange.value))
            case "Ziege":
                PartyScenePlayer.add_song(MidiPlayerSong(7, MidiNotes.Outro_PartyGame_Ziege.value))
        match self.story_progress.Food:
            case "Affe":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Food_Affe.value))
            case "Baer":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Food_Baer.value))
            case "Elefant":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Food_Elefant.value))
            case "Hase":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Food_Hase.value))
            case "Schlange":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Food_Schlange.value))
            case "Ziege":
                PartyScenePlayer.add_song(MidiPlayerSong(10, MidiNotes.Outro_Food_Ziege.value))
        match self.story_progress.Drinks:
            case "Affe":
                PartyScenePlayer.add_song(MidiPlayerSong(7, MidiNotes.Outro_Drinks_Affe.value))
            case "Baer":
                PartyScenePlayer.add_song(MidiPlayerSong(7, MidiNotes.Outro_Drinks_Baer.value))
            case "Elefant":
                PartyScenePlayer.add_song(MidiPlayerSong(6, MidiNotes.Outro_Drinks_Elefant.value))
            case "Hase":
                PartyScenePlayer.add_song(MidiPlayerSong(11, MidiNotes.Outro_Drinks_Hase.value))
            case "Schlange":
                PartyScenePlayer.add_song(MidiPlayerSong(10, MidiNotes.Outro_Drinks_Schlange.value))
            case "Ziege":
                PartyScenePlayer.add_song(MidiPlayerSong(7, MidiNotes.Outro_Drinks_Ziege.value))

        if self.story_progress.RacoonStoleThePresent:
            PartyScenePlayer.add_song(MidiPlayerSong(7, MidiNotes.Outro_Secret_Racoon.value))
        if self.story_progress.YMCA_Y and self.story_progress.YMCA_M and self.story_progress.YMCA_C and self.story_progress.YMCA_A:
            PartyMusic = ArrayPlayer()
            PartyMusic.add_song(MidiPlayerSong(6, MidiNotes.Outro_Background_Music_YMCA.value))

        self.activate_in_sec(PartyScenePlayer.get_total_length() + 7)  # Add Wait Time before restart
        PartyScenePlayer.play_all()
        PartyMusic.play_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-Webcam', type=int, required=False, help='Webcam parameter', default=0)
    args = parser.parse_args()
    animal_party = AnimalParty(webcam=args.Webcam)
    animal_party.start()
