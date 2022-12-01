from scamp import *

s = Session()

sc_part = s.new_osc_part("shardtalk", 57120)

note: NoteHandle = None


def mouse_move(x, y):
    if note is not None:
        note.change_pitch((1 - y) * 127)
        note.change_volume(x)


def mouse_press(x, y, button):
    global note
    note = sc_part.start_note((1 - y) * 127, x)


def mouse_release(x, y, button):
    global note
    note.end()
    note = None

s.register_mouse_listener(on_move=mouse_move, on_release=mouse_release, on_press=mouse_press, relative_coordinates=True,
                          suppress=True)

s.wait_forever()