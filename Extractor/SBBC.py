import essentia
import essentia.standard
import numpy as np


class SBBC(object):
    """Based on https://github.com/u201212551u201611810/PerfectMelody/tree/master
    "Query by Humming for Song Identification Using Voice Isolation" Edwin Alfaro-Paredes, 
    Leonardo Alfaro-Carrasco, Willy Ugarte (2021)
    Args:
        object (_type_): _description_
    """
    def __init__(self, melodia_algo=essentia.standard.PredominantPitchMelodia) -> None:
        self.melodia_algo= melodia_algo
    
    def __call__(self, mp3_path):
        pitch_values = self._estimate_melody(mp3_path)
        chroma_descriptor = self._compute_descriptor(pitch_values)
        return chroma_descriptor
        
    def _estimate_melody(self, mp3_path):
        loader = essentia.standard.EasyLoader(filename=mp3_path, sampleRate=44100)
        audio = loader()
        pitch_extractor = self.melodia_algo(frameSize=2048, hopSize=512)
        pitch_values, _ = pitch_extractor(audio)
        return pitch_values  
    
    @staticmethod
    def _compute_descriptor(melody):
        def to_cents(song):
            cents = []
            for i, f in enumerate(song):
                if song[i] > 0:
                    cents.append(1200 * log2(song[i] / 55))
                else:
                    cents.append(0)
            return np.array(cents)

        def to_semitones(cents):
            semitones = []
            for i, f in enumerate(cents):
                semitones.append(cents[i] // 100)
            return np.array(semitones)

        def map_into_single_octave(semitones):
            mapped = []
            min_n = 1
            max_n = 12
            min_d = np.min(semitones[np.nonzero(semitones)])
            max_d = max(semitones)
            for i, f in enumerate(semitones):
                if semitones[i] > 0:
                    mapped.append(((semitones[i] - min_d) * (max_n - min_n)) // (max_d - min_d) + min_n)
                else:
                    mapped.append(0)
            return np.array(mapped)

        def get_histogram(pitch_class, hop_size=2):
            limitator = 0
            histogram = []
            while limitator < len(pitch_class):
                frame = []
                ini = limitator
                fin = ini + hop_size - 1
                if fin > len(pitch_class) - 1:
                    fin = len(pitch_class) - 1
                counter = Counter(pitch_class[ini:fin])
                for i in range(12):
                    frame.append(counter[i + 1])
                for x in range(len(frame)):
                    if frame[x] == 0:
                        continue
                    num = frame[x] - min(frame)
                    denom = max(frame) - min(frame)
                    frame[x] = num / denom
                histogram.append(frame)
                limitator += hop_size
            return np.array(histogram, dtype=np.single)

        cents = to_cents(melody)
        semitones = to_semitones(cents)
        mapped = map_into_single_octave(semitones)
        chroma = get_histogram(mapped)
        return chroma.T
