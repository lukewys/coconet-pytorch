import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import mido
import soundfile
import pretty_midi
import os

from hparams import I, T, P, MIN_MIDI_PITCH, MAX_MIDI_PITCH


def render_midi_save_wav(path):
  """
  A script for render the midi files and save the wav.
  path is the path to the midi file to be played.
  """
  save_path = path.replace('.mid', '.wav')
  if os.path.exists(save_path):
    os.remove(save_path)

  midi_data = pretty_midi.PrettyMIDI(path)
  wav = midi_data.fluidsynth()
  soundfile.write(save_path, wav, 44100)


# function for converting arrays of shape (T, 4) into midi files
# the input array has entries that are np.nan (representing a rest)
# of an integer between 0 and 127 inclusive

def piano_roll_to_midi(piece):
  """
  piece is a an array of shape (T, 4) for some T.
  The (i,j)th entry of the array is the midi pitch of the jth voice at time i. It's an integer in range(128).
  outputs a mido object mid that you can convert to a midi file by called its .save() method
  """
  piece = np.concatenate([piece, [[np.nan, np.nan, np.nan, np.nan]]], axis=0)

  bpm = 50
  microseconds_per_beat = 60 * 1000000 / bpm

  mid = mido.MidiFile()
  tracks = {'soprano': mido.MidiTrack(), 'alto': mido.MidiTrack(),
            'tenor': mido.MidiTrack(), 'bass': mido.MidiTrack()}
  past_pitches = {'soprano': np.nan, 'alto': np.nan,
                  'tenor': np.nan, 'bass': np.nan}
  delta_time = {'soprano': 0, 'alto': 0, 'tenor': 0, 'bass': 0}

  # create a track containing tempo data
  metatrack = mido.MidiTrack()
  metatrack.append(mido.MetaMessage('set_tempo',
                                    tempo=int(microseconds_per_beat), time=0))
  mid.tracks.append(metatrack)

  # create the four voice tracks
  for voice in tracks:
    mid.tracks.append(tracks[voice])
    tracks[voice].append(mido.Message(
      'program_change', program=52, time=0))

  # add notes to the four voice tracks
  for i in range(len(piece)):
    pitches = {'soprano': piece[i, 0], 'alto': piece[i, 1],
               'tenor': piece[i, 2], 'bass': piece[i, 3]}
    for voice in tracks:
      if np.isnan(past_pitches[voice]):
        past_pitches[voice] = None
      if np.isnan(pitches[voice]):
        pitches[voice] = None
      if pitches[voice] != past_pitches[voice]:
        if past_pitches[voice]:
          tracks[voice].append(mido.Message('note_off', note=int(past_pitches[voice]),
                                            velocity=64, time=delta_time[voice]))
          delta_time[voice] = 0
        if pitches[voice]:
          tracks[voice].append(mido.Message('note_on', note=int(pitches[voice]),
                                            velocity=64, time=delta_time[voice]))
          delta_time[voice] = 0
      past_pitches[voice] = pitches[voice]
      # 480 ticks per beat and each line of the array is a 16th note
      delta_time[voice] += 120

  return mid


def save_heatmaps(model, y, C, step, device):
  """
  Plugs (y, C) into model and converts the (logprob) output to probabilities.
  In other words, in the output, the (i,j,k)th entry is the probability of getting the kth pitch when you sample for the ith voice at time j.
  """
  compressed = y.reshape(-1)
  x = np.zeros((I * T, P))
  r = np.arange(I * T)
  x[r, compressed] = 1
  x = x.reshape(I, T, P)
  x = torch.tensor(x).type(torch.FloatTensor).to(device)
  x = x.view(1, I, T, P)
  C2 = torch.tensor(C).type(torch.FloatTensor).to(device)
  model.eval()
  with torch.no_grad():
    out = model.forward(x, C2).view(I, T, P).cpu().numpy().transpose(2, 0, 1)
    probs = np.exp(out) / np.sum(np.exp(out), axis=0)
    probs = probs.transpose(1, 2, 0)

    """
    The output of a inputting a single sample into the net is an array of shape (I, T, P) that is interpreted as log probabilities.
    After normalizing to probabilities, it can be interpreted as four arrays (once for each voice soprano, alto, tenor, bass) of shape (T, P)
    That consist of the probabilities of selecting given pitches for each voice at each time. These probabilities can be visualized in heatmaps,
    and this function stores those four heatmaps in the arrays soprano_probs, alto_probs, tenor_probs, bass_probs.
    """

    soprano_probs = probs[0].transpose()
    alto_probs = probs[1].transpose()
    tenor_probs = probs[2].transpose()
    bass_probs = probs[3].transpose()

    #Displays and save the latest heatmaps produced by store_heatmaps.
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(np.flip(soprano_probs, axis=0), cmap='hot', interpolation='nearest')
    axs[0].set_title('soprano')
    axs[1].imshow(np.flip(alto_probs, axis=0), cmap='hot', interpolation='nearest')
    axs[1].set_title('alto')
    axs[2].imshow(np.flip(tenor_probs, axis=0), cmap='hot', interpolation='nearest')
    axs[2].set_title('tenor')
    axs[3].imshow(np.flip(bass_probs, axis=0), cmap='hot', interpolation='nearest')
    axs[3].set_title('bass')
    fig.set_figheight(5)
    fig.set_figwidth(15)

    os.makedirs('results/heatmaps', exist_ok=True)
    plt.savefig(f'results/heatmaps/heatmaps_step_{step}')
    plt.close()


def pad_number(n):
  """
  prepare numbers for better file storage
  """
  if n == 0:
    return '00000'
  else:
    digits = int(np.ceil(np.log10(n)))
    pad_zeros = 5 - digits
    return '0' * pad_zeros + str(n)


def harmonize_melody_and_save_midi(melody, id_number, model, device):
  """
  Generate an artificial chorale which has melody in the soprano line and a Bach-like harmonization in the other lines.
  Save the result in a midi file named {id_number}midi.mid
  """
  y = np.random.randint(P, size=(I, T))
  y[0] = np.array(melody) - 30  # subtract 30 because 30 is the minimum midi_value
  D0 = np.ones((1, T)).astype(int)
  D1 = np.zeros((3, T)).astype(int)
  D = np.concatenate([D0, D1], axis=0)
  prediction = harmonize(y, D, model, device) + 30  # 30 back on before passing to piano_roll_to_midi
  prediction = prediction.transpose().tolist()
  prediction = np.array(prediction)
  midi_output = piano_roll_to_midi(prediction)

  os.makedirs('results/generated_midi', exist_ok=True)
  midi_output.save(f'results/generated_midi/harmonize_val_set_step_{str(pad_number(id_number))}.mid')
  render_midi_save_wav(f'results/generated_midi/harmonize_val_set_step_{str(pad_number(id_number))}.mid')


# harmonize a melody
def harmonize(y, C, model, device):
  """
  Generate an artificial Bach Chorale starting with y, and keeping the pitches where C==1.
  Here C is an array of shape (4, 128) whose entries are 0 and 1.
  The pitches outside of C are repeatedly resampled to generate new values.
  For example, to harmonize the soprano line, let y be random except y[0] contains the soprano line, let C[1:] be 0 and C[0] be 1.
  """
  model.eval()
  with torch.no_grad():
    x = y
    C2 = C.copy()
    num_steps = int(2 * I * T)
    alpha_max = .999
    alpha_min = .001
    eta = 3 / 4
    for i in range(num_steps):
      p = np.maximum(alpha_min, alpha_max - i * (alpha_max - alpha_min) / (eta * num_steps))
      sampled_binaries = np.random.choice(2, size=C.shape, p=[p, 1 - p])
      C2 += sampled_binaries
      C2[C == 1] = 1
      x_cache = x
      x = model.pred(x, C2, device)
      x[C2 == 1] = x_cache[C2 == 1]
      C2 = C.copy()
    return x


def generate_random_chorale(model, device):
  """
  Calls harmonize with random initialization and C=0, and so generates a new sample that sounds like Bach.
  """
  y = np.random.randint(P, size=(I, T)).astype(int)
  C = np.zeros((I, T)).astype(int)
  return harmonize(y, C, model, device)

def generate_random_chorale_and_save(model, device, name='generated', render_midi=True):
  """
  Calls harmonize with random initialization and C=0, and so generates a new sample that sounds like Bach.
  """

  prediction = generate_random_chorale(model, device) + MIN_MIDI_PITCH
  prediction = prediction.transpose().tolist()
  prediction = np.array(prediction)
  midi_output = piano_roll_to_midi(prediction)

  os.makedirs('generated_midi', exist_ok=True)
  midi_output.save(f'generated_midi/{name}.mid')
  if render_midi:
    render_midi_save_wav(f'generated_midi/{name}.mid')

class Unit(nn.Module):
  """
  Two convolution layers each followed by batchnorm and relu, plus a residual connection.
  """

  def __init__(self, hidden_size):
    super(Unit, self).__init__()
    self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
    self.batchnorm1 = nn.BatchNorm2d(hidden_size)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
    self.batchnorm2 = nn.BatchNorm2d(hidden_size)
    self.relu2 = nn.ReLU()

  def forward(self, x):
    y = x
    y = self.conv1(y)
    y = self.batchnorm1(y)
    y = self.relu1(y)
    y = self.conv2(y)
    y = self.batchnorm2(y)
    y = y + x
    y = self.relu2(y)
    return y


class Net(nn.Module):
  """
  A CNN that where you input a starter chorale and a mask and it outputs a prediction for the values
  in the starter chorale away from the mask that are most like the training data.
  """

  def __init__(self, n_layers, hidden_size):
    super(Net, self).__init__()
    self.initial_conv = nn.Conv2d(2 * I, hidden_size, 3, padding=1)
    self.initial_batchnorm = nn.BatchNorm2d(hidden_size)
    self.initial_relu = nn.ReLU()
    self.conv_layers = nn.ModuleList()
    # n_layer // 2 because there are 2 convs in each "unit" to handle residual connect
    for _i in range(n_layers // 2):
      self.conv_layers.append(Unit(hidden_size))
    self.conv_pitch_linear = nn.Conv2d(P, P, 1, padding=0)
    self.conv_instrument_downproj = nn.Conv2d(hidden_size, I, 1, padding=0)

  def forward(self, x, C):
    # x is a tensor of shape (N, I, T, P)
    # C is a tensor of 0s and 1s of shape (N, I, T)
    # returns a tensor of shape (N, I, T, P)

    # get the number of batches
    N = x.shape[0]
    T = x.shape[2]

    # tile the array C out of a tensor of shape (N, I, T, P)
    tiled_C = C.view(N, I, T, 1)
    tiled_C = tiled_C.repeat(1, 1, 1, P)

    # mask x and combine it with the mask to produce a tensor of shape (N, 2*I, T, P)
    y = torch.cat((tiled_C * x, tiled_C), dim=1)

    # apply the convolution and relu layers
    y = self.initial_conv(y)
    y = self.initial_batchnorm(y)
    y = self.initial_relu(y)
    for _n in range(len(self.conv_layers)):
      y = self.conv_layers[_n](y)
    y = torch.permute(y, (0, 3, 1, 2))
    y = self.conv_pitch_linear(y)
    y = torch.permute(y, (0, 2, 3, 1))
    y = self.conv_instrument_downproj(y)
    return y

  def expand(self, y, C, device):
    # y is an array of shape (I, T) with integer entries in [0, P)
    # C is an array of shape (I, T) consisting of 0s and 1s
    # the entries of y away from the support of C should be considered 'unknown'

    # x is shape (I, T, P) one-hot representation of y
    compressed = y.reshape(-1)
    x = np.zeros((I * T, P))
    r = np.arange(I * T)
    x[r, compressed] = 1
    x = x.reshape(I, T, P)

    # prep x and C for the plugging into the model
    x = torch.tensor(x).type(torch.FloatTensor).to(device)
    x = x.view(1, I, T, P)
    C2 = torch.tensor(C).type(torch.FloatTensor).view(1, I, T).to(device)
    return x, C2

  def pred(self, y, C, device, temperature=1.0, seed=100):
    x, C2 = self.expand(y, C, device)
    # plug x and C2 into the model
    rs = np.random.RandomState(seed)
    with torch.no_grad():
      out = self.forward(x, C2).view(I, T, P).cpu().numpy()
      out = out.transpose(2, 0, 1)  # shape (P, I, T)
      probs = np.exp(out / temperature) / np.exp(out / temperature).sum(axis=0)  # shape (P, I, T)
      cum_probs = np.cumsum(probs, axis=0)  # shape (P, I, T)
      u = rs.rand(I, T)  # shape (I, T)
      return np.argmax(cum_probs > u, axis=0)
