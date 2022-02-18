import torch
from tqdm import tqdm
import argparse
from coconet import Net, generate_random_chorale_and_save

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='Generate from Coconet.')

if __name__ == '__main__':
  parser.add_argument('--num_gen', type=int, default=1, help='Number of MIDIs to generate.')
  parser.add_argument('--name_offset', type=int, default=0, help='The number offset when generating multiple files.')
  parser.add_argument('--no_render', dest='render', action='store_false')
  args = parser.parse_args()
  n_layers = 64
  hidden_size = 128
  model = Net(n_layers, hidden_size).to(device)
  model.load_state_dict(torch.load('pretrained.pt'))

  if args.num_gen == 1:
    generate_random_chorale_and_save(model, device, render_midi=args.render)
  else:
    for i in tqdm(range(args.num_gen)):
      generate_random_chorale_and_save(model, device, name=i + args.name_offset, render_midi=args.render)
