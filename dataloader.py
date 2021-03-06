import sys
import numpy, random, pdb, math, pickle, glob, time, os, re
import torch
from utils import collate_dense_tensors


class DataLoader:
    def __init__(self, fname, opt, dataset='simulator', single_shard=False):
        if opt.debug:
            single_shard = True
        self.opt = opt
        self.random = random.Random()
        self.random.seed(12345)  # use this so that the same batches will always be picked

        if dataset == 'i80' or dataset == 'us101':
            data_dir = f'traffic-data/state-action-cost/data_{dataset}_v0'
            if single_shard:
                # quick load for debugging
                data_files = [f'{next(os.walk(data_dir))[1][0]}/']
            else:
                data_files = next(os.walk(data_dir))[1]

            self.images = []
            self.actions = []
            self.costs = []
            self.states = []
            self.raw_states = []
            self.ids = []
            self.ego_car_images = []
            for df in data_files:
                combined_data_path = f'{data_dir}/{df}/all_data.pth'
                if os.path.isfile(combined_data_path):
                    print(f'[loading data shard: {combined_data_path}]')
                    data = torch.load(combined_data_path)
                    self.images += data.get('images')
                    self.actions += data.get('actions')
                    self.costs += data.get('costs')
                    self.states += data.get('states')
                    self.raw_states += data.get('raw_states')
                    self.ids += data.get('ids')
                    self.ego_car_images += data.get('ego_car')
                else:
                    print(data_dir)
                    images = []
                    actions = []
                    costs = []
                    states = []
                    raw_states = []
                    ids = glob.glob(f'{data_dir}/{df}/car*.pkl')
                    ids.sort()
                    ego_car_images = []
                    for f in ids:
                        print(f'[loading {f}]')
                        fd = pickle.load(open(f, 'rb'))
                        Ta = fd['actions'].size(0)
                        Tp = fd['pixel_proximity_cost'].size(0)
                        Tl = fd['lane_cost'].size(0)
                        # assert Ta == Tp == Tl  # TODO Check why there are more costs than actions
                        # if not(Ta == Tp == Tl): pdb.set_trace()
                        images.append(fd['images'])
                        actions.append(fd['actions'])
                        costs.append(torch.cat((
                            fd.get('pixel_proximity_cost')[:Ta].view(-1, 1),
                            fd.get('lane_cost')[:Ta].view(-1, 1),
                        ), 1),)
                        states.append(fd['states'])
                        raw_states.append(fd['raw_states'])
                        ego_car_images.append(fd['ego_car'])

                    print(f'Saving {combined_data_path} to disk')
                    torch.save({
                        'images': images,
                        'actions': actions,
                        'costs': costs,
                        'states': states,
                        'raw_states': raw_states,
                        'ids': ids,
                        'ego_car': ego_car_images,
                    }, combined_data_path)
                    self.images += images
                    self.actions += actions
                    self.costs += costs
                    self.states += states
                    self.raw_states += raw_states
                    self.ids += ids
                    self.ego_car_images += ego_car_images
        else:
            assert False, 'Data set not supported'

        self.n_episodes = len(self.images)
        print(f'Number of episodes: {self.n_episodes}')
        splits_path = data_dir + '/splits.pth'
        if os.path.exists(splits_path):
            print(f'[loading data splits: {splits_path}]')
            self.splits = torch.load(splits_path)
            self.train_indx = self.splits.get('train_indx')
            self.valid_indx = self.splits.get('valid_indx')
            self.test_indx = self.splits.get('test_indx')
        else:
            print('[generating data splits]')
            rgn = numpy.random.RandomState(0)
            perm = rgn.permutation(self.n_episodes)
            n_train = int(math.floor(self.n_episodes * 0.8))
            n_valid = int(math.floor(self.n_episodes * 0.1))
            self.train_indx = perm[0 : n_train]
            self.valid_indx = perm[n_train : n_train + n_valid]
            self.test_indx = perm[n_train + n_valid :]
            torch.save(dict(
                train_indx=self.train_indx,
                valid_indx=self.valid_indx,
                test_indx=self.test_indx,
            ), splits_path)

        stats_path = data_dir + '/data_stats.pth'
        if os.path.isfile(stats_path):
            print(f'[loading data stats: {stats_path}]')
            stats = torch.load(stats_path)
            self.a_mean = stats.get('a_mean')
            self.a_std = stats.get('a_std')
            self.s_mean = stats.get('s_mean')
            self.s_std = stats.get('s_std')
        else:
            print('[computing action stats]')
            all_actions = []
            for i in self.train_indx:
                all_actions.append(self.actions[i])
            all_actions = torch.cat(all_actions, 0)
            self.a_mean = torch.mean(all_actions, 0)
            self.a_std = torch.std(all_actions, 0)
            print('[computing state stats]')
            all_states = []
            for i in self.train_indx:
                all_states.append(self.states[i][:, 0])
            all_states = torch.cat(all_states, 0)
            self.s_mean = torch.mean(all_states, 0)
            self.s_std = torch.std(all_states, 0)
            torch.save({'a_mean': self.a_mean,
                        'a_std': self.a_std,
                        's_mean': self.s_mean,
                        's_std': self.s_std}, stats_path)

        car_sizes_path = data_dir + '/car_sizes.pth'
        print(f'[loading car sizes: {car_sizes_path}]')
        self.car_sizes = torch.load(car_sizes_path)

    # get batch to use for forward modeling
    # a sequence of ncond given states, a sequence of npred actions,
    # and a sequence of npred states to be predicted
    def get_batch_fm(self, split, npred=-1, cuda=True):

        # Choose the correct device
        device = torch.device('cuda') if cuda else torch.device('cpu')

        if split == 'train':
            indx = self.train_indx
        elif split == 'valid':
            indx = self.valid_indx
        elif split == 'test':
            indx = self.test_indx

        if npred == -1:
            npred = self.opt.npred

        images, states, raw_states, actions, costs, ids, sizes, ego_cars = [], [], [], [], [], [], [], []
        nb = 0
        T = self.opt.ncond + npred
        t0 = self.opt.ncond
        t1 = T
        while nb < self.opt.batch_size:
            s = self.random.choice(indx)
            # min is important since sometimes numbers do not align causing issues in stack operation below
            episode_length = min(self.images[s].size(0), self.states[s].size(0))
            if episode_length >= T:
                t = self.random.randint(0, episode_length - T)
                images.append(self.images[s][t : t + T].to(device))
                actions.append(self.actions[s][t : t + T].to(device))
                states.append(self.states[s][t : t + T, 0].to(device))  # discard 6 neighbouring cars
                costs.append(self.costs[s][t : t + T].to(device))
                ids.append(self.ids[s])
                ego_cars.append(self.ego_car_images[s].to(device))
                splits = self.ids[s].split('/')
                time_slot = splits[-2]
                car_id = int(re.findall(r'car(\d+).pkl', splits[-1])[0])
                size = self.car_sizes[time_slot][car_id]
                sizes.append([size[0], size[1]])
                raw_states.append(collate_raw_states(self.raw_states[s][t : t + T], reference_state_idx=t0-1).to(device))
                nb += 1

        # Pile up stuff
        images  = torch.stack(images)
        states  = torch.stack(states)
        raw_states = collate_dense_tensors(raw_states)
        actions = torch.stack(actions)
        sizes   = torch.tensor(sizes)
        ego_cars = torch.stack(ego_cars)

        # Normalise actions, state_vectors, state_images
        # Zeming asks: Is it really reasonable to normalize the state vectors?
        # It feels like you lose a lot of information (like velocities between batches, etc)
        if not self.opt.debug:
            actions = self.normalise_action(actions)
            states = self.normalise_state_vector(states)
        images = self.normalise_state_image(images)
        ego_cars = self.normalise_state_image(ego_cars)

        costs = torch.stack(costs)

        # |-----ncond-----||------------npred------------||
        # ^                ^                              ^
        # 0               t0                             t1
        input_images  = images [:,   :t0].float().contiguous()
        input_states  = states [:,   :t0].float().contiguous()
        target_images = images [:, t0:t1].float().contiguous()
        target_states = states [:, t0:t1].float().contiguous()
        if self.opt.object_space:
            input_states  = raw_states[:,   :t0].float().contiguous()
            target_states = raw_states[:, t0:t1].float().contiguous()
        target_costs  = costs  [:, t0:t1].float().contiguous()
        t0 -= 1; t1 -= 1
        actions       = actions[:, t0:t1].float().contiguous()
        # input_actions = actions[:, :t0].float().contiguous()
        ego_cars = ego_cars.float().contiguous()
        #          n_cond                      n_pred
        # <---------------------><---------------------------------->
        # .                     ..                                  .
        # +---------------------+.                                  .  ^          ^
        # |i|i|i|i|i|i|i|i|i|i|i|.  3 × 117 × 24                    .  |          |
        # +---------------------+.                                  .  | inputs   |
        # +---------------------+.                                  .  |          |
        # |s|s|s|s|s|s|s|s|s|s|s|.  4                               .  |          |
        # +---------------------+.                                  .  v          |
        # .                   +-----------------------------------+ .  ^          |
        # .                2  |a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a| .  | actions  |
        # .                   +-----------------------------------+ .  v          |
        # .                     +-----------------------------------+  ^          | tensors
        # .       3 × 117 × 24  |i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|  |          |
        # .                     +-----------------------------------+  |          |
        # .                     +-----------------------------------+  |          |
        # .                  4  |s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|  | targets  |
        # .                     +-----------------------------------+  |          |
        # .                     +-----------------------------------+  |          |
        # .                  2  |c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|  |          |
        # .                     +-----------------------------------+  v          v
        # +---------------------------------------------------------+             ^
        # |                           car_id                        |             | string
        # +---------------------------------------------------------+             v
        # +---------------------------------------------------------+             ^
        # |                          car_size                       |  2          | tensor
        # +---------------------------------------------------------+             v

        # When object_space is True, s is replaced by a `ncars x 9` tensor, where the 9 is
        # x, y, length, width, dx, dy, vx, vy, in_frame

        return [input_images, input_states, ego_cars], actions, [target_images, target_states, target_costs], ids, sizes

    @staticmethod
    def normalise_state_image(images):
        return images.float().div_(255.0)

    def normalise_state_vector(self, states):
        shape = (1, 1, 4) if states.dim() == 3 else (1, 4)  # dim = 3: state sequence, dim = 2: single state
        states -= self.s_mean.view(*shape).expand(states.size()).to(states.device)
        states /= (1e-8 + self.s_std.view(*shape).expand(states.size())).to(states.device)
        return states

    def normalise_action(self, actions):
        actions -= self.a_mean.view(1, 1, 2).expand(actions.size()).to(actions.device)
        actions /= (1e-8 + self.a_std.view(1, 1, 2).expand(actions.size())).to(actions.device)
        return actions


def collate_raw_states(raw_states, reference_state_idx=0):
    def filter_by(inds, values, func):
        mask = func(inds, values)
        return inds[mask], values[mask]

    raw_states = [filter_by(torch.LongTensor(i), v, lambda _, vv: (vv[:, -1] == 1))
                  for i, v in raw_states]
    catind = torch.cat([x[0] for x in raw_states], dim=0)
    minind = catind.min()
    maxind = catind.max()
    raw_states = [(i - minind, v.to(torch.float32)) for i, v in raw_states]
    car2ind = torch.zeros(maxind-minind+1, dtype=torch.int64) - 1
    reference_state = raw_states[reference_state_idx]
    car2ind.scatter_(0, reference_state[0], torch.arange(reference_state[0].shape[0]))

    collated_states = torch.zeros(len(raw_states), *(reference_state[1].shape))
    for idx, (i, v) in enumerate(raw_states):
        i = car2ind[i]
        i, v = filter_by(i, v, lambda ii, vv: ii >= 0)
        collated_states[idx].index_copy_(0, i, v)

    return collated_states


if __name__ == '__main__':
    # Create some dummy options
    class DataSettings:
        debug = False
        batch_size = 4
        npred = 20
        ncond = 10
    # Instantiate data set object
    d = DataLoader(None, opt=DataSettings, dataset='i80')
    # Retrieve first training batch
    x = d.get_batch_fm('train', cuda=False)
