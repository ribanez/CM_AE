import torch
import torch.utils.data
import h5py


class Dataset_CM(torch.utils.data.Dataset):

    def __init__(self, filename):
        super(Dataset_CM, self).__init__()

        self.h5pyfile = h5py.File(filename, 'r')
        self.num_proteins, self.max_sequence_len = self.h5pyfile['contact_map'].shape

    def __getitem__(self, index):
        cm = torch.Tensor(self.h5pyfile['contact_map'][index, :, :])

        return cm

    def __len__(self):
        return self.num_proteins

    def merge_samples_to_minibatch(samples):
        samples_list = []
        for s in samples:
            samples_list.append(s)
        # sort according to length of aa sequence
        samples_list.sort(key=lambda x: len(x[0]), reverse=True)
        return zip(*samples_list)


def contruct_dataloader_from_disk(filename, minibatch_size):
    return torch.utils.data.DataLoader(Dataset_CM(filename),
                                       batch_size=minibatch_size,
                                       shuffle=True,
                                       collate_fn=Dataset_CM.merge_samples_to_minibatch)
