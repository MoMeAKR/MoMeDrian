
import numpy as np
import glob
import random
import json
import os
import cv2


def translate_element(element):
    """
    Fixed translate_element:
      - Renders `parallel` as comma-separated bracket groups:  [a,b,c]!2
      - Strips redundant outer wrappers when appropriate
      - Special-case: handles cycle patterns like [{brackets: [...]}, {note: ~}]
        so they render as [<[...parallel groups...]> ~] (matches your expected nesting)
    """
    def get_multiplier(elem):
        return str(elem.get("multiplier", "")) if isinstance(elem, dict) else ""

    def _strip_outer_wrapper(s):
        if len(s) >= 2 and ((s[0] == "<" and s[-1] == ">") or (s[0] == "[" and s[-1] == "]")):
            return s[1:-1]
        return s

    if isinstance(element, dict):
        # PARALLEL -> [a,b,c]{mult}
        if "parallel" in element:
            par = element["parallel"]
            mult = ""
            if isinstance(par, dict):
                mult = get_multiplier(par) or get_multiplier(element)
                items = []
                if "cycle" in par:
                    for child in par["cycle"]:
                        s = translate_element(child) if isinstance(child, (dict, list)) else str(child)
                        items.append(_strip_outer_wrapper(s))
                elif "brackets" in par:
                    for child in par["brackets"]:
                        s = translate_element(child) if isinstance(child, (dict, list)) else str(child)
                        items.append(_strip_outer_wrapper(s))
                elif "note" in par:
                    notes = par["note"]
                    if isinstance(notes, list):
                        for n in notes:
                            items.append(str(n))
                    else:
                        items.append(str(notes))
                elif isinstance(par, list):
                    for child in par:
                        s = translate_element(child) if isinstance(child, (dict, list)) else str(child)
                        items.append(_strip_outer_wrapper(s))
                else:
                    items = [str(par)]
            elif isinstance(par, list):
                mult = get_multiplier(element)
                items = []
                for p in par:
                    s = translate_element(p) if isinstance(p, (dict, list)) else str(p)
                    items.append(_strip_outer_wrapper(s))
            else:
                items = [str(par)]
                mult = get_multiplier(element)
            inner = ",".join(items)
            return f"[{inner}]{mult}"

        # CYCLE
        if "cycle" in element:
            items = element["cycle"]
            # Special case: [{brackets: [...]}, {note: ~}] -> [<...> ~]
            if (len(items) == 2 and isinstance(items[0], dict) and "brackets" in items[0] and
                (isinstance(items[1], dict) and "note" in items[1] or isinstance(items[1], (str, int)))):
                bracket_children = items[0]["brackets"]
                angle_parts = []
                for child in bracket_children:
                    s = translate_element(child) if isinstance(child, (dict, list)) else str(child)
                    angle_parts.append(_strip_outer_wrapper(s))
                angle_inner = " ".join(angle_parts)
                angle_str = f"<{angle_inner}>"
                note_s = translate_element(items[1]) if isinstance(items[1], (dict, list)) else str(items[1])
                mult = get_multiplier(element)
                return f"[{angle_str} {note_s}]{mult}"

            # General cycle formatting (angle brackets)
            parts = []
            for e in items:
                s = translate_element(e) if isinstance(e, (dict, list)) else str(e)
                parts.append(s)
            inner = " ".join(parts)
            mult = get_multiplier(element)
            return f"<{inner}>{mult}"

        # BRACKETS
        if "brackets" in element:
            items = element["brackets"]
            parts = []
            for e in items:
                s = translate_element(e) if isinstance(e, (dict, list)) else str(e)
                parts.append(s)
            # If all children are already parallel-bracketed, don't add another outer bracket
            if parts and all(p.startswith("[") and "]" in p for p in parts):
                inner = " ".join(parts)
                mult = get_multiplier(element)
                return f"{inner}{mult}"
            inner = " ".join(parts)
            mult = get_multiplier(element)
            return f"[{inner}]{mult}"

        # NOTE
        if "note" in element:
            notes = element["note"]
            mult = get_multiplier(element)
            if isinstance(notes, list):
                return " ".join(f"{str(n)}{mult}" for n in notes)
            else:
                return f"{str(notes)}{mult}"

        return str(element)

    elif isinstance(element, list):
        return " ".join(translate_element(e) for e in element)

    elif isinstance(element, (str, int)):
        return str(element)

    else:
        return str(element)


# Compatible melody translator that uses translate_element above.
def translate_melody_element(elem):
    if isinstance(elem, dict):
        if "cycle" in elem:
            mult = elem.get("multiplier", "*1")
            inner = " ".join(
                translate_element(e) if isinstance(e, (dict, list)) else str(e)
                for e in elem["cycle"]
            )
            return f"[{inner}]{mult}"
        elif "brackets" in elem:
            mult = elem.get("multiplier", "*1")
            inner = " ".join(
                translate_element(b) if isinstance(b, (dict, list)) else str(b)
                for b in elem["brackets"]
            )
            return f"<{inner}>{mult}"
        elif "note" in elem:
            mult = elem.get("multiplier", "")
            notes = elem["note"]
            if isinstance(notes, list):
                return " ".join(f"{str(n)}{mult}" for n in notes)
            else:
                return f"{notes}{mult}"
        else:
            return str(elem)
    elif isinstance(elem, list):
        return " ".join(translate_element(e) for e in elem)
    else:
        return str(elem)


def translate_drums_structure(structure):
    """
    Translates the 'drums' part of the structure into the target string format.
    Handles 'parallel' and 'bank' keys.
    """
    content = structure.get("drums", {})

    sequence_parts = []
    for elem in content:
        part_str = translate_element(elem)
        # If this element has a bank, wrap it
        bank = elem.get("bank", "RolandTR707")
        effects = elem.get("effects", dict())
        part_str = f"sound(\"{part_str}\").bank(\"{bank}\")"
        for effect in effects.keys(): 
            part_str += f".{effect}({effects[effect]})"
        

        sequence_parts.append(part_str)
    sequence_str = " ".join(sequence_parts)
    return sequence_str

def translate_melody_structure(structure):
    """
    Translates the 'melody' part of the structure into the target string format.
    Uses n("...") if the element has a 'scale' key or if notes look numeric,
    otherwise uses note("...").
    """
    content = structure.get("melody", [])
    sequence_parts = []

    def is_numeric_note(note):
        """Check if a note is purely numeric (int/str digits)."""
        if isinstance(note, int):
            return True
        if isinstance(note, str) and note.strip().isdigit():
            return True
        return False

    for elem in content:
        # Choose default function name
        func = "note"

        # Decide if we need to use n(...) instead
        if "scale" in elem:
            func = "n"
        elif "note" in elem:
            notes = elem["note"]
            if isinstance(notes, list) and all(is_numeric_note(n) for n in notes):
                func = "n"
            elif isinstance(notes, (int, str)) and is_numeric_note(notes):
                func = "n"

        # Build main pattern string
        if "brackets" in elem:
            pattern = f'{func}("<{translate_element(elem["brackets"])}>")'
        elif "cycle" in elem:
            pattern = f'{func}("[{translate_element(elem["cycle"])}]")'
        elif "note" in elem:
            pattern = f'{func}("{translate_element({"note": elem["note"]})}")'
        elif "sound" in elem or "instrument" in elem:
            # For non-note patterns (e.g., drums)
            if "note" in elem:
                pattern = f'sound("{translate_element({"note": elem["note"]})}")'
            else:
                pattern = ""
        else:
            pattern = ""

        # Add scale if present
        if "scale" in elem:
            pattern += f'.scale("{elem["scale"]}")'

        # Add instrument/sound
        if "instrument" in elem:
            pattern += f'.sound("{elem["instrument"]}")'
        elif "sound" in elem:
            if isinstance(elem["sound"], list):
                pattern += f'.sound("{",".join(elem["sound"])}")'
            else:
                pattern += f'.sound("{elem["sound"]}")'

        # Add effects
        if "effects" in elem:
            for eff, val in elem["effects"].items():
                val_str = f'.{eff}({val})'
                pattern += val_str

        sequence_parts.append(f"$: {pattern}")

    return "\n".join(sequence_parts)


def translate_drums_structure(structure):
    """
    Translates the 'drums' part of the structure into the target string format.
    Uses translate_element to handle cycles, brackets, parallels, etc.
    """
    content = structure.get("drums", [])
    sequence_parts = []
    
    for elem in content:
        # Get translated pattern (handles note, cycle, brackets, parallel)
        pattern_str = translate_element(elem)
        
        # Default bank if not specified
        bank = elem.get("bank", "RolandTR707")
        
        # Build base
        part_str = f'sound("{pattern_str}").bank("{bank}")'
        
        # Add effects
        effects = elem.get("effects", {})
        for eff, val in effects.items():
            part_str += f".{eff}({val})"
        
        sequence_parts.append(f"$: {part_str}")
    
    return "\n".join(sequence_parts)



def translate_current_structure(): 
    # first_processing_function()
    struct_path = os.path.join(os.path.dirname(__file__), "current_strudel_structure.json")
    structure = json.load(open(struct_path))
    result = translate_drums_structure(structure)
    result_melody = translate_melody_structure(structure)
    print('Drums: {}\n\nMelody: {}'.format(result, result_melody))
    
    with open(os.path.join(os.path.dirname(__file__), "my_test.txt"), "w") as f: 
        f.write("{}\n{}".format(result.strip(), result_melody.strip()))    

def get_seven_bin_normalized_histogram(image_input):
    """
    Computes a 7-bin normalized histogram of pixel intensities for a grayscale image.

    Parameters
    ----------
    image_input : str or np.ndarray
        Path to the image file or a NumPy array representing the image.
        If a color image is provided, it will be converted to grayscale.

    Returns
    -------
    np.ndarray
        A 1D array of length 7, where each element represents the normalized
        fraction of pixels falling into each intensity bin (from dark to bright).
    """

    # Load image if a file path is provided
    if isinstance(image_input, str):
        image = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    else:
        image = image_input
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define bin edges for 7 bins over the range [0, 256)
    bin_edges = np.linspace(0, 256, 8, endpoint=True)

    # Compute histogram
    hist, _ = np.histogram(image, bins=bin_edges)

    # Normalize histogram to sum to 1
    hist_normalized = hist.astype(np.float32) / hist.sum()

    return hist_normalized

def split_image_vertically(image_input, num_parts):
    """
    Splits an image vertically into `num_parts` equal sections.

    Parameters
    ----------
    image_input : str or np.ndarray
        Path to the image file or a NumPy array representing the image.
    num_parts : int
        Number of vertical sections to split the image into.

    Returns
    -------
    List[np.ndarray]
        List of image sections as NumPy arrays.
    """
    # Load image if a file path is provided
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input

    height, width = image.shape[:2]
    part_height = height // num_parts
    parts = []
    for i in range(num_parts):
        start = i * part_height
        end = (i + 1) * part_height if i < num_parts - 1 else height
        parts.append(image[start:end, :])
    return parts

def interpret_hist(available_notes, hist, th = 0.1):

    result = []
    for n, h in zip(available_notes, hist): 
        if h < th: 
            result.append(n)
    return result

def process_image_test(source_path, available_drum_notes, available_melody_notes): 
    """
    Build a Strudel-like structure from an image by sampling histograms and mapping them to
    drum and melody patterns.

    Melody requirements:
    - Drop 'brackets' usage; ensure 'note' sequences are always non-empty.
    - If 'melody_parallel' is True, organize melody as multiple layers (multiple parts in the list).

    Returns
    -------
    dict
        Structure containing 'drums' and 'melody' entries suitable for translation.
    """
    parts = split_image_vertically(source_path, 8)
    hists = [get_seven_bin_normalized_histogram(p) for p in parts]

    # Drums sampling
    notes = interpret_hist(available_drum_notes, hists[0])
    cycle = interpret_hist(available_drum_notes, hists[1])
    brackets = interpret_hist(available_drum_notes, hists[2])

    # Melody sampling 
    melody_notes = interpret_hist(available_melody_notes, hists[3], th = 0.5)
    melody_cycles = interpret_hist(available_melody_notes, hists[4], th = 0.5)
    melody_brackets = interpret_hist(available_melody_notes, hists[5], th = 0.5)  # reused as a third note pool
    melody_parallel = True if len(interpret_hist(available_melody_notes, hists[6], th = 0.5)) > 3 else False
    
    # Build drums as a list with a single element containing a 'parallel' list of parts.
    # This matches translate_drums_structure -> translate_element expectations.
    drums_parts = [
        # {"cycle": [{"note": notes}]},
        {"cycle": notes},
        {"cycle": [{"note": cycle}]},
        {"note": brackets}
    ]
    results = {
        "drums": [
            {
                "parallel": drums_parts
                # Optional: "bank": "RolandTR707"  # translate_drums_structure defaults to RolandTR707
            }
        ]
    }

    # Melody section:
    # - No 'brackets' used; only 'note' and optional 'cycle'.
    # - If parallel: create multiple melody parts (layers) with different instruments/effects.
    # - If not parallel: single part, simple cycle combining two ensured note pools.
    if melody_parallel:
        melody = [
            {
                "cycle": [
                    {"note": melody_notes}
                ],
                "instrument": "gm_synth_bass_1",
                "effects": {"gain": 0.9}
            },
            {
                "cycle": [
                    {"note": melody_cycles}
                ],
                "instrument": "gm_electric_piano_1",
                "effects": {"gain": 0.8}
            },
            {
                "cycle": [
                    {"note": melody_brackets}
                ],
                "instrument": "gm_marimba",
                "effects": {"gain": 0.7}
            }
        ]
    else:
        melody = [
            {
                "cycle": [
                    {"note": melody_notes},
                    {"note": melody_cycles}
                ],
                "instrument": "gm_synth_bass_1",
                "effects": {"gain": 0.85}
            }
        ]

    results.update({"melody": melody})


    results = {
        # "drums": [
        #     {"note": ["bd", "rim"],
        #      "effects": {"delay": 0.5, "gain": 0.85}}, 
        # ], 
        "drums": [
        # Layered bass + rim
        {
            "note": ["bd", "rim"],
            "effects": {"delay": 0.5, "gain": 0.85}
        },
        # A repeating hi-hat cycle
        {
            "cycle": [
                {"note": "hh"}, 
                {"note": "hh"}, 
                {"note": "hh"}, 
                {"note": "hh"}
            ],
            "multiplier": "*4",
            "effects": {"gain": 0.6}
        },
        # Bracketed snare roll
        {
            "brackets": [
                {"note": "sd"},
                {"note": "sd"},
                {"note": "sd"}
            ],
            "multiplier": "/2",
            "effects": {"reverb": 0.3}
        },
        # Parallel bass drum and hi-hat groove
        {
            "parallel": [
                {"cycle": ["bd", "~"], "multiplier": "*2"},
                {"cycle": ["hh", "hh"], "multiplier": "*4"}
            ],
            "effects": {"room": 1.2, "gain": 0.7}
        },
        # Complex nested structure
        {
            "cycle": [
                {"brackets": [
                    {"note": "bd"},
                    {"note": "sd"},
                    {"parallel": [
                        {"note": "hh"},
                        {"note": "oh"}
                    ]}
                ]},
                {"note": "~"},
                {"cycle": ["rim", "rim", "~"]}
            ],
            "multiplier": "!2",
            "effects": {"delay": 0.25, "gain": 0.9}
        }
    ],
        "melody": [
            {
                "brackets": [
                    {"note": ["4"]},
                    {"cycle": ["3@3", "4"]}, 
                    {"cycle": [{"brackets": [{"note": 2}, {"note": 0}]}, {"note": ["~@16"]}, {"note": "~"}]}
                ], 
                "scale": "D4:minor", 
                "instrument": "gm_accordion:2",
                "effects": {"room": 2, "gain": 0.4}
            }, 
            {
                "cycle": [
                    {"note": "0"}, 
                    {"cycle": ["~", "0"]}, 
                    {"note": "4"}, 
                    {"cycle": ["3","2"]}, 
                    {"cycle": ["0", "~"] }, 
                    {"cycle": ["0", "~"] }, 
                    {"brackets": ["0", "2"]}, 
                    {"note": ["~"]}
                ],
                "scale": "D2:minor", 
                "sound": ["sawtooth", "triangle"],
                "effects": {"lpf": 800}
            }, 
            {
                "cycle": [
                    {"note": "~"}, 
                    {"cycle": [
                        {"brackets": [
                            {"parallel": 
                            {"cycle": ["d3", "a3", "f4"], 
                             "multiplier": "!2"}}, 
                             {
                                "parallel":
                             
                            {"cycle": ["d3", "bb3", "g4"], 
                             "multiplier": "!2"}}
                        ]}, 
                        {"note": "~"}
                    ]}
                ], 
                "instrument": "gm_electric_guitar_muted", 
                "effects": {"delay": 0.5}
            }

        ]
    }


    print(json.dumps(results, indent = 4))

    return results


def first_processing_function(): 

    imgs= glob.glob(os.path.join(os.path.expanduser('~'), "Images", "*.jpg"))
    target_img_path = random.sample(imgs, 1)[0]
    print(target_img_path)
    # target_img_path = os.path.join(os.path.expanduser('~'), "Images", "90y7n1.jpg")
    available_notes = ["bd", "hh", "-", "sd", "oh", "ht", "rd"]
    available_melody_notes = ["c3", 'd3', 'e3', 'f3', 'g3', 'a3', 'b3']

    # =========================== 

    new_struct = process_image_test(target_img_path, available_notes, available_melody_notes)
    
    with open(os.path.join(os.path.dirname(__file__), "current_strudel_structure.json"), "w") as f: 
        json.dump(new_struct, f, indent= 4)
    

def my_test(): 
    # print(get_seven_bin_normalized_histogram(os.path.join(os.path.expanduser('~'), "Images", "90y7n1.jpg")))
    first_processing_function()
    translate_current_structure()


# Fill the melody part in process_image_test