#Test
import mido
import time
msg = mido.Message('note_on', note=1, velocity=64, time=1)


print(mido.get_output_names()) #gibt alle Portname von MIDI Ausgänge aus, bei open_output den richtigen von loopMIDI eintragen
print(mido.get_input_names())

output_port = mido.open_output('DesktopMidi 1')

# Sende eine CC-Nachricht, um den Reverb-Wert zu ändern
def send_cc_message(controller, value, channel=0):
    msg3 = mido.Message('control_change', channel=channel, control=controller, value=value)
    output_port.send(msg3)
    print(f"Gesendete CC Nachricht: Controller={controller}, Wert={value}, Kanal={channel}")

# Beispiel: Reverb mit CC#91 steuern
reverb_value = 100  # Setze den gewünschten Reverb-Wert (0-127)
output_port.send(msg)
send_cc_message(92, reverb_value)
time.sleep(4)
send_cc_message(92, 0)