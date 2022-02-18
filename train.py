import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime

from coconet import Net, save_heatmaps, harmonize_melody_and_save_midi
from hparams import I, T, P, MIN_MIDI_PITCH, MAX_MIDI_PITCH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load training data
data = np.load('Jsb16thSeparated.npz', encoding='bytes', allow_pickle=True)

# data augmentation
all_tracks = []
for y in data['train']:
  for i in range(-3, 4):
    all_tracks.append(y + i)

# construct training data
train_tracks = []

for track in all_tracks:
  track = track.transpose()
  cut = 0
  while cut < track.shape[1] - T:
    if (track[:, cut:cut + T] > 0).all():
      train_tracks.append(track[:, cut:cut + T] - MIN_MIDI_PITCH)
    cut += T

train_tracks = np.array(train_tracks).astype(int)

# get test sample
test_sample = data['valid'][0].transpose()[:, :T]
test_sample_melody = test_sample[0]

if __name__ == '__main__':
  batch_size = 16
  n_layers = 64
  hidden_size = 128
  n_train_steps = 80000
  save_every = n_train_steps // 10
  show_every = max(1, n_train_steps // 1000)
  softmax = F.softmax

  model = Net(n_layers, hidden_size).to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
  losses = []

  model.train()
  N = batch_size

  for i in range(n_train_steps):

    # tensor of shape (N, I, T)
    C = np.random.randint(2, size=(N, I, T))

    # batch is an np array of shape (N, I, T), entries are integers in [0, P)
    indices = np.random.choice(train_tracks.shape[0], size=N)
    batch = train_tracks[indices]

    # targets is of shape (N*I*T)
    targets = batch.reshape(-1)
    targets = torch.tensor(targets).to(device)

    # x is of shape (N, I, T, P)

    batch = batch.reshape(N * I * T)
    x = np.zeros((N * I * T, P))
    r = np.arange(N * I * T)
    x[r, batch] = 1
    x = x.reshape(N, I, T, P)
    x = torch.tensor(x).type(torch.FloatTensor).to(device)

    C2 = torch.tensor(C).type(torch.FloatTensor).to(device)
    out = model(x, C2)
    out = out.view(N * I * T, P)
    loss = loss_fn(out, targets)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 500 == 0:
      now_date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
      print(f'{now_date_time} | step: {i} | loss: {loss.item()}')
      D0 = np.ones((1, T))
      D1 = np.zeros((3, T))
      D = np.concatenate([D0, D1], axis=0).astype(int)
      y = np.random.randint(P, size=(I, T))
      y[0, :] = np.array(test_sample_melody - MIN_MIDI_PITCH)
      save_heatmaps(model, y, D, i, device)
      if i % 5000 == 0:
        harmonize_melody_and_save_midi(test_sample_melody, i, model, device)
      model.train()

    # adjust learning rate
    if i % 5000 == 0 and i > 0:
      for g in optimizer.param_groups:
        g['lr'] *= .75

  torch.save(model.state_dict(), 'pretrained.pt')
