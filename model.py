import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


class DeepFam(nn.Module):    
    def __init__(self, FLAGS):
        super(DeepFam, self).__init__()
        
        # number of features from convolutional motif detector
        self.n_convfeatures = 0
        # list of features extracted from each variable-length filters
        self.motif_features = []

        for len_w, num_w in zip(FLAGS.window_lengths, FLAGS.num_windows):
            # each motif feature is extracted through conv -> one-max pooling
            motifs = nn.Sequential( nn.Conv2d(1, num_w, (len_w, FLAGS.charset_size)),
                                    #nn.BatchNorm2d(num_w),
                                    n n.ReLU(),
                                    nn.MaxPool2d( (FLAGS.seq_len - len_w + 1, 1), 
                                                   stride=(1, FLAGS.seq_len)) )
            motifs.apply(weights_init)
            self.motif_features.append(motifs)
            self.n_convfeatures += num_w

        # features from variable length motifs are combined using dense layer
        self.hidden = nn.Sequential( nn.Dropout(0.2),
                                     nn.Linear(self.n_convfeatures, FLAGS.num_hidden),
                                     nn.BatchNorm1d(FLAGS.num_hidden),
                                     nn.ReLU() )
        self.hidden.apply(weights_init)
        
        # classifier using one depth dense layer
        self.classifier = nn.Linear( FLAGS.num_hidden, FLAGS.num_classes )
        self.classifier.apply(weights_init)


    def forward(self, x):
        features = []

        # extract features from each convolutional motif detector
        for m_feature in self.motif_features :
            features.append( m_feature(x) )

        out = torch.cat( features, 1 )
        out = out.view(-1, out.size(1))
        out = self.hidden(out)
        out = self.classifier(out)

        return out

if __name__ == '__main__':
    pass
